import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees

from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
import os

# --- 1. Dataset Class with Feature Masking ---
class MaskedFeatureDataset(Dataset):
    def __init__(self, X, mask_prob=0.1):
        self.X = X.astype(np.float32)
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        mask = np.random.rand(len(x)) < self.mask_prob
        x_masked = x.copy()
        x_masked[mask] = 0
        return x_masked, x, mask.astype(np.float32)

# --- 2. MLP Autoencoder ---
class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# --- 3. Train Autoencoder ---
def train_autoencoder(X, mask_prob=0.1, epochs=50, batch_size=64):
    dataset = MaskedFeatureDataset(X, mask_prob)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MaskedAutoencoder(X.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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

# --- 4. Embedding extraction ---
def get_embeddings(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X.astype(np.float32)).cuda()
        embeddings = model.encoder(X_tensor).cpu().numpy()
    return embeddings

# --- 5. Main pipeline ---
if __name__ == "__main__":
    # Load train data
    df = pd.read_csv("../data/merged_train.csv")
    y_time = df['OS_YEARS'].values
    y_event = df['OS_STATUS'].values.astype(bool)
    X = df.drop(columns=[col for col in ['ID', 'OS_YEARS', 'OS_STATUS'] if col in df.columns]).values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_val, y_time_tr, y_time_val, y_event_tr, y_event_val = train_test_split(
        X_scaled, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42
    )

    # Train autoencoder
    model = train_autoencoder(X_train, mask_prob=0.35, epochs=100)
    emb_train = get_embeddings(model, X_train)
    emb_val = get_embeddings(model, X_val)

    y_train_struct = Surv.from_arrays(y_event_tr, y_time_tr)
    y_val_struct = Surv.from_arrays(y_event_val, y_time_val)

    # RSF on embeddings
    rsf = ExtraSurvivalTrees(
    n_estimators=204,
    max_features="sqrt",
    min_samples_split=3,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42)
    rsf.fit(emb_train, y_train_struct)

    pred_train = rsf.predict(emb_train)
    pred_val = rsf.predict(emb_val)

    c_train = concordance_index_ipcw(y_train_struct, y_train_struct, pred_train, tau=7)[0]
    c_val = concordance_index_ipcw(y_train_struct, y_val_struct, pred_val, tau=7)[0]
    print(f"\n✅ C-index (train): {c_train:.4f}, C-index (val): {c_val:.4f}")

    # --- Retrain on full train data ---
    emb_full = get_embeddings(model, X_scaled)
    y_full_struct = Surv.from_arrays(y_event, y_time)
    rsf_final = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10,
                                     max_features="sqrt", random_state=42, n_jobs=-1)
    rsf_final.fit(emb_full, y_full_struct)

    # --- Predict on test set ---
    df_test = pd.read_csv("../data/merged_test.csv")
    X_test = df_test.drop(columns=["ID"]).values
    X_test_scaled = scaler.transform(X_test)
    emb_test = get_embeddings(model, X_test_scaled)

    risk_scores = rsf_final.predict(emb_test)
    submission = pd.DataFrame({
        "ID": df_test["ID"],
        "risk_score": risk_scores
    })
    os.makedirs("../submissions", exist_ok=True)
    sub_path = "../submissions/submission_model_ae_rsf.csv"
    submission.to_csv(sub_path, index=False)
    print(f"📤 Submission saved as {sub_path}")
