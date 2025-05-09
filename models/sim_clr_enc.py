import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- SimCLR Dataset with Feature Dropout Augmentations ---
class SimCLRDataset(Dataset):
    def __init__(self, X, drop_prob=0.2):
        self.X = X.astype(np.float32)
        self.drop_prob = drop_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        def augment(x):
            mask = np.random.rand(len(x)) < self.drop_prob
            x_aug = x.copy()
            x_aug[mask] = 0  # or add noise
            return x_aug
        return augment(x), augment(x)

# --- Simple MLP Encoder ---
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, projection_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)

# --- NT-Xent Contrastive Loss ---
def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim_matrix = sim_matrix / temperature

    labels = torch.arange(batch_size).cuda()
    labels = torch.cat([labels, labels], dim=0)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# --- Training Loop ---
def train_simclr(X, epochs=100, batch_size=64, drop_prob=0.2):
    dataset = SimCLRDataset(X, drop_prob)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Encoder(input_dim=X.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x1, x2 in loader:
            x1, x2 = x1.cuda(), x2.cuda()
            z1, z2 = model(x1), model(x2)
            loss = contrastive_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    # 🔥 Save the encoder
    torch.save(model.backbone.state_dict(), "simclr_encoder.pt")
    print("✅ Pretrained encoder saved as simclr_encoder.pt")

    return model

# --- Run Pretraining ---
if __name__ == "__main__":
    df = pd.read_csv("../data/merged_train.csv")  # full training features
    X = df.drop(columns=["OS_YEARS", "OS_STATUS"], errors="ignore").values
    X = StandardScaler().fit_transform(X)
    train_simclr(X, epochs=100)
