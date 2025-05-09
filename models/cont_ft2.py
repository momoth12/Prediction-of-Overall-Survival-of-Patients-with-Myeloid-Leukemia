import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

# --- Dataset ---
class SurvivalContrastiveDataset(Dataset):
    def __init__(self, X, y_time, y_event):
        self.X = X
        self.y_time = y_time
        self.y_event = y_event

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_time[idx], self.y_event[idx]

# --- Encoder + Projection Head ---
class Encoder(nn.Module):
    def __init__(self, input_dim, proj_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.proj = nn.Linear(64, proj_dim)

    def forward(self, x):
        z = self.model(x)
        return z, self.proj(z)

# --- Contrastive Loss (NT-Xent) ---
def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    N = z.shape[0]
    mask = torch.eye(N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    targets = torch.arange(N, device=z.device)
    targets = (targets + N//2) % N
    loss = nn.functional.cross_entropy(sim, targets)
    return loss

# --- Load Data ---
df = pd.read_csv("../data/merged_train.csv")
y_time = df["OS_YEARS"].values
y_event = df["OS_STATUS"].values.astype(bool)
X = df.drop(columns=["OS_YEARS", "OS_STATUS"]).values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_val, y_time_tr, y_time_val, y_event_tr, y_event_val = train_test_split(X, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42)

# --- Training ---
device = "cuda"
dataset = SurvivalContrastiveDataset(X_train, y_time_tr, y_event_tr)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = Encoder(X.shape[1]).to(device)
predictor = nn.Linear(64, 1).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=1e-3)
mse_loss = nn.MSELoss()

for epoch in range(100):
    model.train()
    total_mse, total_contrastive = 0, 0
    for Xb, tb, eb in loader:
        Xb = Xb.float().to(device)
        t_true = torch.tensor(tb, dtype=torch.float32).unsqueeze(1).to(device)

        z, z_proj = model(Xb)
        t_pred = predictor(z)

        # SimCLR-style contrastive loss (view = noise + dropout)
        noise = torch.randn_like(Xb) * 0.1
        z2, z2_proj = model(Xb + noise)
        contrastive = nt_xent_loss(z_proj, z2_proj)

        # MSE loss for time
        mse = mse_loss(t_pred, t_true)

        loss = mse + 0.1 * contrastive
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_mse += mse.item()
        total_contrastive += contrastive.item()

    print(f"Epoch {epoch+1}, MSE: {total_mse:.4f}, Contrastive: {total_contrastive:.4f}")

# --- RSF on frozen embeddings ---
model.eval()
with torch.no_grad():
    z_train = model.model(torch.tensor(X_train).float().to(device)).cpu().numpy()
    z_val = model.model(torch.tensor(X_val).float().to(device)).cpu().numpy()

y_train_struct = Surv.from_arrays(y_event_tr, y_time_tr)
y_val_struct = Surv.from_arrays(y_event_val, y_time_val)

from sksurv.ensemble import RandomSurvivalForest
rsf = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=42)
rsf.fit(z_train, y_train_struct)

pred_train = rsf.predict(z_train)
pred_val = rsf.predict(z_val)

c_train = concordance_index_ipcw(y_train_struct, y_train_struct, pred_train, tau=7)[0]
c_val = concordance_index_ipcw(y_train_struct, y_val_struct, pred_val, tau=7)[0]
print(f"✅ C-index (train): {c_train:.4f}, C-index (val): {c_val:.4f}")
