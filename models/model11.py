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

# --- 1. Dataset Class with Important Feature Masking ---
class MaskedFeatureDataset(Dataset):
    def __init__(self, X, important_idx, noise_std=0.1):
        self.X = X.astype(np.float32)
        self.important_idx = important_idx
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        x_masked = x.copy()
        mask = np.zeros_like(x)
        noise = np.random.normal(0, self.noise_std, size=len(self.important_idx)).astype(np.float32)
        x_masked[self.important_idx] += noise
        mask[self.important_idx] = 1.0
        return x_masked, x, mask

# --- 2. Masked Autoencoder ---
class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# --- 3. Training Function ---
def train_autoencoder(X, important_idx, epochs=100, batch_size=64, lr=1e-3):
    dataset = MaskedFeatureDataset(X, important_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MaskedAutoencoder(X.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        embeddings = model.encoder(X_tensor).cpu().numpy()
    return embeddings

# --- 5. Main script ---
if __name__ == "__main__":
    df = pd.read_csv("../data/merged_train.csv")
    drop_cols = [col for col in ["ID", "OS_YEARS", "OS_STATUS"] if col in df.columns]
    X = df.drop(columns=drop_cols).values
    y_time = df["OS_YEARS"].values
    y_event = df["OS_STATUS"].values.astype(bool)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Train-val split
    X_train, X_val, y_time_tr, y_time_val, y_event_tr, y_event_val = train_test_split(
        X_scaled, y_time, y_event, test_size=0.2, stratify=y_event, random_state=42
    )

    # Select most important features using variance (placeholder, replace with importances if available)
    variances = np.var(X_train, axis=0)
    important_idx = np.argsort(variances)[-30:]  # Top 30 features by variance

    model = train_autoencoder(X_train, important_idx, epochs=100)
    emb_train = get_embeddings(model, X_train)
    emb_val = get_embeddings(model, X_val)

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

    # Submission
    df_test = pd.read_csv("../data/merged_test.csv")
    test_ids = df_test["ID"] if "ID" in df_test.columns else np.arange(len(df_test))
    X_test = df_test.drop(columns=["ID"] if "ID" in df_test.columns else []).values
    X_test_scaled = scaler.transform(X_test)
    emb_test = get_embeddings(model, X_test_scaled)
    pred_test = rsf.predict(emb_test)

    submission = pd.DataFrame({"ID": test_ids, "risk_score": pred_test})
    submission_path = "../submissions/submission_model11_denoise_imp.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ Submission saved: {submission_path}")
