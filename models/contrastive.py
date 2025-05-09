import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

# --- 1. Contrastive Dataset with Feature Dropout ---
class ContrastiveFeatureDataset(Dataset):
    def __init__(self, X, mask_prob=0.2):
        self.X = X.astype(np.float32)
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        def augment(x):
            x_aug = x.copy()
            mask = np.random.rand(len(x)) < self.mask_prob
            x_aug[mask] = 0  # or noise
            return x_aug
        return augment(x), augment(x)  # (x_i, x_j)

# --- 2. Encoder + Projection Head ---
class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, proj_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1), h  # Return both z (proj) and h (embed)

# --- 3. NT-Xent Contrastive Loss ---
def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim /= temperature

    batch_size = z_i.size(0)
    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, -9e15)

    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    numerator = torch.exp(positives)
    denominator = torch.exp(sim).sum(dim=1)
    loss = -torch.log(numerator / denominator).mean()
    return loss

# --- 4. Pretrain Contrastive Encoder ---
def pretrain_contrastive(X, epochs=100, batch_size=128):
    dataset = ContrastiveFeatureDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ContrastiveEncoder(input_dim=X.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_i, x_j in loader:
            x_i, x_j = x_i.cuda(), x_j.cuda()
            z_i, _ = model(x_i)
            z_j, _ = model(x_j)
            loss = nt_xent_loss(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Contrastive Loss: {total_loss / len(loader):.4f}")
    return model

# --- 5. Full Pipeline ---
if __name__ == "__main__":
    df = pd.read_csv("../data/merged_train.csv")
    y_time = df["OS_YEARS"].values
    y_event = df["OS_STATUS"].values.astype(bool)
    X = df.drop(columns=["OS_YEARS", "OS_STATUS"]).values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_val, y_time_tr, y_time_val, y_event_tr, y_event_val = train_test_split(
        X_scaled, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42
    )

    model = pretrain_contrastive(X_train, epochs=100)
    model.eval()
    with torch.no_grad():
        emb_train = model.encoder(torch.tensor(X_train, dtype=torch.float32).cuda()).cpu().numpy()
        emb_val = model.encoder(torch.tensor(X_val, dtype=torch.float32).cuda()).cpu().numpy()

    y_train_struct = Surv.from_arrays(y_event_tr, y_time_tr)
    y_val_struct = Surv.from_arrays(y_event_val, y_time_val)

    rsf = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10,
                               max_features="sqrt", random_state=42, n_jobs=-1)
    rsf.fit(emb_train, y_train_struct)

    pred_train = rsf.predict(emb_train)
    pred_val = rsf.predict(emb_val)

    c_train = concordance_index_ipcw(y_train_struct, y_train_struct, pred_train, tau=7)[0]
    c_val = concordance_index_ipcw(y_train_struct, y_val_struct, pred_val, tau=7)[0]
    print(f"✅ C-index (train): {c_train:.4f}, C-index (val): {c_val:.4f}")
