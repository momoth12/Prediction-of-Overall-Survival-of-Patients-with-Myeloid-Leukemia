import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw


# --- Dataset with Feature Dropout for Augmentation ---
class ContrastiveSurvivalDataset(Dataset):
    def __init__(self, X, mask_prob=0.2):
        self.X = X.astype(np.float32)
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        mask1 = np.random.rand(len(x)) < self.mask_prob
        mask2 = np.random.rand(len(x)) < self.mask_prob
        x1 = x.copy()
        x2 = x.copy()
        x1[mask1] = 0.0
        x2[mask2] = 0.0
        return x1, x2


# --- Encoder with Projection Head ---
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return h, z


# --- NT-Xent Contrastive Loss ---
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    batch_size = z1.size(0)
    labels = torch.arange(batch_size).cuda()
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).cuda()
    sim = sim.masked_fill(mask, float('-inf'))
    sim = sim - torch.max(sim, dim=1, keepdim=True)[0]  # stability

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_indices = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask
    pos_log_prob = log_prob[pos_indices].view(2 * batch_size, -1).squeeze()
    loss = -pos_log_prob.mean()
    return loss


# --- Train Encoder with Contrastive Loss ---
def train_contrastive_encoder(X, epochs=100, batch_size=128):
    dataset = ContrastiveSurvivalDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Encoder(input_dim=X.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x1, x2 in loader:
            x1, x2 = x1.cuda(), x2.cuda()
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = contrastive_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model


# --- Embedding Extraction ---
def get_embeddings(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X.astype(np.float32)).cuda()
        h, _ = model(X_tensor)
        return h.cpu().numpy()


# --- Main Pipeline ---
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

    encoder = train_contrastive_encoder(X_train, epochs=100)
    emb_train = get_embeddings(encoder, X_train)
    emb_val = get_embeddings(encoder, X_val)

    y_train_struct = Surv.from_arrays(y_event_tr, y_time_tr)
    y_val_struct = Surv.from_arrays(y_event_val, y_time_val)

    rsf = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10,
                               max_features="sqrt", random_state=42, n_jobs=-1)
    rsf.fit(emb_train, y_train_struct)

    pred_train = rsf.predict(emb_train)
    pred_val = rsf.predict(emb_val)

    c_train = concordance_index_ipcw(y_train_struct, y_train_struct, pred_train, tau=7)[0]
    c_val = concordance_index_ipcw(y_train_struct, y_val_struct, pred_val, tau=7)[0]
    print(f"\n✅ C-index (train): {c_train:.4f}, C-index (val): {c_val:.4f}")