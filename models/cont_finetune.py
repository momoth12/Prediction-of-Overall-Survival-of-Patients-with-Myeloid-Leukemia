import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

# --- Dataset for survival with float labels ---
class SurvivalDataset(Dataset):
    def __init__(self, X, y_time, y_event):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_time = torch.tensor(y_time, dtype=torch.float32)
        self.y_event = torch.tensor(y_event, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_time[idx], self.y_event[idx]

# --- Encoder definition matching SimCLR ---
class SurvivalMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z

# --- Load and preprocess data ---
df = pd.read_csv("../data/merged_train.csv")
y_time = df['OS_YEARS'].values
y_event = df['OS_STATUS'].values.astype(bool)
X = df.drop(columns=['OS_YEARS', 'OS_STATUS']).values

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_train, X_val, y_time_tr, y_time_val, y_event_tr, y_event_val = train_test_split(
    X_scaled, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42
)

train_dataset = SurvivalDataset(X_train, y_time_tr, y_event_tr)
val_dataset = SurvivalDataset(X_val, y_time_val, y_event_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# --- Model init & pretrained encoder loading ---
model = SurvivalMLP(X.shape[1]).cuda()
model.encoder.load_state_dict(torch.load("simclr_encoder.pt"))  # load pretrained encoder

# Unfreeze encoder for fine-tuning
for param in model.encoder.parameters():
    param.requires_grad = True

# --- Optimizer and loss ---
optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
loss_fn = nn.MSELoss()

# --- Training loop ---
for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for xb, yb_time, yb_event in train_loader:
        xb = xb.cuda()
        yb = torch.tensor(yb_time, dtype=torch.float32).view(-1, 1).cuda()

        pred, _ = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch}/100, Loss: {total_loss/len(train_loader):.4f}")

# --- Get embeddings for RSF ---
def get_embeddings(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        _, embeddings = model(X_tensor)
        return embeddings.cpu().numpy()

emb_train = get_embeddings(model, X_train)
emb_val = get_embeddings(model, X_val)

# --- Train RSF ---
from sksurv.ensemble import RandomSurvivalForest
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
