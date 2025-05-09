import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

# --- Dataset with SimCLR-style augmentations ---
class ContrastiveDataset(Dataset):
    def __init__(self, X, dropout_rate=0.2):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.dropout_rate = dropout_rate

    def augment(self, x):
        mask = torch.rand_like(x) < self.dropout_rate
        x_aug = x.clone()
        x_aug[mask] = 0
        return x_aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return self.augment(x), self.augment(x)

# --- Encoder with projection head ---
class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, proj_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, F.normalize(z, dim=1)

# --- NT-Xent loss ---
def contrastive_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim /= temperature

    n = z1.size(0)
    labels = torch.arange(n, device=z.device)
    labels = torch.cat([labels + n, labels], dim=0)

    mask = torch.eye(2 * n, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)

    sim_exp = torch.exp(sim)
    sim_sum = sim_exp.sum(dim=1)
    sim_pos = torch.exp(F.cosine_similarity(z, torch.roll(z, n, 0)) / temperature)

    loss = -torch.log(sim_pos / sim_sum)
    return loss.mean()

# --- Train SimCLR Encoder ---
def train_contrastive_encoder(X, epochs=100, batch_size=128):
    dataset = ContrastiveDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = ContrastiveEncoder(X.shape[1]).cuda()
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

# --- Extract embeddings ---
def get_embeddings(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X.astype(np.float32)).cuda()
        h, _ = model(X_tensor)
        return h.cpu().numpy()

# --- Main ---
if __name__ == "__main__":
    df = pd.read_csv("../data/merged_train.csv")
    y_time = df["OS_YEARS"].values
    y_event = df["OS_STATUS"].values.astype(bool)
    X = df.drop(columns=["OS_YEARS", "OS_STATUS"]).values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Split for validation only
    X_train, X_val, y_time_tr, y_time_val, y_event_tr, y_event_val = train_test_split(
        X_scaled, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42
    )

    # Train contrastive encoder on training split
    encoder = train_contrastive_encoder(X_train, epochs=100)
    emb_train = get_embeddings(encoder, X_train)
    emb_val = get_embeddings(encoder, X_val)

    y_train_struct = Surv.from_arrays(y_event_tr, y_time_tr)
    y_val_struct = Surv.from_arrays(y_event_val, y_time_val)

    rsf = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10,
                               max_features="sqrt", n_jobs=-1, random_state=42)
    rsf.fit(emb_train, y_train_struct)

    pred_train = rsf.predict(emb_train)
    pred_val = rsf.predict(emb_val)
    c_train = concordance_index_ipcw(y_train_struct, y_train_struct, pred_train, tau=7)[0]
    c_val = concordance_index_ipcw(y_train_struct, y_val_struct, pred_val, tau=7)[0]
    print(f"✅ C-index (train): {c_train:.4f}, C-index (val): {c_val:.4f}")

    # --- Retrain on full data & generate submission ---
    encoder_full = train_contrastive_encoder(X_scaled, epochs=100)
    emb_full = get_embeddings(encoder_full, X_scaled)
    y_full_struct = Surv.from_arrays(y_event, y_time)
    rsf_final = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10,
                                     max_features="sqrt", n_jobs=-1, random_state=42)
    rsf_final.fit(emb_full, y_full_struct)

    # Test set
    df_test = pd.read_csv("../data/merged_test.csv")
    X_test = scaler.transform(df_test.drop(columns=["ID"]).values)
    emb_test = get_embeddings(encoder_full, X_test)

    submission = pd.DataFrame({
        "ID": df_test["ID"],
        "risk_score": rsf_final.predict(emb_test)
    })
    submission.to_csv("../submissions/submission_simclr_rsf.csv", index=False)
    print("✅ Submission file saved as ../submissions/submission_simclr_rsf.csv")
