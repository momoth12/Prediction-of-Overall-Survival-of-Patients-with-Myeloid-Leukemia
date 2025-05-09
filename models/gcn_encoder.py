import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

# --- 1. GCN Encoder ---
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# --- 2. Create Patient Graph (KNN) ---
def build_patient_graph(X, k=10):
    sim = cosine_similarity(X)
    edges = []
    for i in range(len(X)):
        top_k = np.argsort(-sim[i])[1:k+1]
        for j in top_k:
            edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

# --- 3. Load & Prepare ---
df = pd.read_csv("../data/merged_train.csv")
y_time = df["OS_YEARS"].values
y_event = df["OS_STATUS"].values.astype(bool)
X = df.drop(columns=["OS_YEARS", "OS_STATUS"]).values

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float)

edge_index = build_patient_graph(X_scaled, k=10)

# --- 4. Train/Val Split ---
idx_train, idx_val = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y_event, random_state=42)
y_struct = Surv.from_arrays(y_event, y_time)

# --- 5. Train GCN ---
gcn = GCNEncoder(input_dim=X.shape[1]).cuda()
optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=1e-4)

X_tensor = X_tensor.cuda()
edge_index = edge_index.cuda()

for epoch in range(100):
    gcn.train()
    optimizer.zero_grad()
    out = gcn(X_tensor, edge_index)
    loss = out.norm(p=2, dim=1).mean()  # simple embedding regularization
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# --- 6. Extract Embeddings ---
gcn.eval()
with torch.no_grad():
    embeddings = gcn(X_tensor, edge_index).cpu().numpy()

# --- 7. Fit RSF ---
rsf = ExtraSurvivalTrees(
    n_estimators=204,
    max_features="sqrt",
    min_samples_split=3,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)
rsf.fit(embeddings[idx_train], y_struct[idx_train])

pred_train = rsf.predict(embeddings[idx_train])
pred_val = rsf.predict(embeddings[idx_val])

c_train = concordance_index_ipcw(y_struct[idx_train], y_struct[idx_train], pred_train, tau=7)[0]
c_val = concordance_index_ipcw(y_struct[idx_train], y_struct[idx_val], pred_val, tau=7)[0]
print(f"✅ C-index (train): {c_train:.4f}, C-index (val): {c_val:.4f}")
