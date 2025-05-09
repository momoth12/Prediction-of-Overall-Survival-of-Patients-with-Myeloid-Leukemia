import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

# --- 1. Dataset Class with Denoising Masking ---
class DenoisingFeatureDataset(Dataset):
    def __init__(self, X, mask_prob=0.33, noise_std=0.1):
        self.X = X.astype(np.float32)
        self.mask_prob = mask_prob
        self.feature_means = self.X.mean(axis=0)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        mask = np.random.rand(len(x)) < self.mask_prob
        x_masked = x.copy()
        x_masked[mask] = self.feature_means[mask] + np.random.normal(0, self.noise_std, size=mask.sum())
        return x_masked, x, mask.astype(np.float32)

# --- 2. Transformer Encoder for Tabular Data ---
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=128, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(emb_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # shape: (batch_size, 1, emb_dim)
        z = self.encoder(x)  # shape: (batch_size, 1, emb_dim)
        z = z.squeeze(1)
        return self.decoder(z), z

# --- 3. Train Transformer Autoencoder ---
def train_transformer_autoencoder(X, mask_prob=0.15, epochs=200, batch_size=64):
    dataset = DenoisingFeatureDataset(X, mask_prob)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TransformerEncoder(X.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss(reduction='none')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_masked, x_true, mask in loader:
            x_masked, x_true, mask = x_masked.cuda(), x_true.cuda(), mask.cuda()
            x_pred, _ = model(x_masked)
            loss = (loss_fn(x_pred, x_true) * mask).sum() / (mask.sum() + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model

# --- 4. Extract Embeddings ---
def get_embeddings(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X.astype(np.float32)).cuda()
        embeddings = model.encoder(model.embedding(X_tensor).unsqueeze(1)).squeeze(1).cpu().numpy()
    return embeddings

# --- 5. Main ---
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

    model = train_transformer_autoencoder(X_train, mask_prob=0.33, epochs=300)
    emb_train = get_embeddings(model, X_train)
    emb_val = get_embeddings(model, X_val)

    y_train_struct = Surv.from_arrays(y_event_tr, y_time_tr)
    y_val_struct = Surv.from_arrays(y_event_val, y_time_val)

    rsf = ExtraSurvivalTrees(
        n_estimators=204,
        max_features="sqrt",
        min_samples_split=3,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    )
    rsf.fit(emb_train, y_train_struct)

    pred_train = rsf.predict(emb_train)
    pred_val = rsf.predict(emb_val)

    c_train = concordance_index_ipcw(y_train_struct, y_train_struct, pred_train, tau=7)[0]
    c_val = concordance_index_ipcw(y_train_struct, y_val_struct, pred_val, tau=7)[0]
    print(f"✅ C-index (train): {c_train:.4f}, C-index (val): {c_val:.4f}")
