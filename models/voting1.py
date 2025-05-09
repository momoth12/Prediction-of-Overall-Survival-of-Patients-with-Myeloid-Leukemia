import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees
from sksurv.metrics import concordance_index_ipcw
from model10dropout import train_autoencoder, get_embeddings
from model_4 import extract_cytogenetics_features

# --- Load and merge data ---
clinical = pd.read_csv("../data/clinical_train.csv")
target = pd.read_csv("../data/target_train.csv")
molecular = pd.read_csv("../data/molecular_train.csv")

# Add cytogenetics and pathway features
clinical = extract_cytogenetics_features(clinical)
data = clinical.merge(target, on="ID").dropna(subset=["OS_STATUS", "OS_YEARS"])

# Define top genes and pathways
top_genes = molecular["GENE"].value_counts().head(15).index.tolist()
mutation_count = molecular.groupby("ID").size().rename("MUTATION_COUNT")
avg_vaf = molecular.groupby("ID")["VAF"].mean().rename("VAF_AVG")
gene_flags = (
    molecular[molecular["GENE"].isin(top_genes)]
    .assign(flag=1)
    .pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
)

pathway_genes = {
    "EPIGENETIC": ['TET2', 'DNMT3A', 'IDH1', 'IDH2'],
    "RTK": ['FLT3', 'KIT', 'NRAS', 'KRAS'],
    "TRANSCRIPTION": ['RUNX1', 'CEBPA', 'ETV6'],
    "TUMOR_SUPPRESSOR": ['TP53', 'WT1', 'NPM1']
}
for name, group in pathway_genes.items():
    data[f"PATHWAY_{name}"] = molecular[molecular["GENE"].isin(group)] \
        .groupby("ID").size().reindex(data["ID"], fill_value=0).gt(0).astype(int).values

pathway_cols = [col for col in data.columns if col.startswith("PATHWAY_")]
mol_feats = pd.concat([mutation_count, avg_vaf, gene_flags], axis=1).reset_index()
merged = data.merge(mol_feats, on="ID", how="left")

# --- Prepare features and labels ---
mol_cols = ['MUTATION_COUNT', 'VAF_AVG'] + top_genes + pathway_cols
clinical_cols = ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT']
cyto_cols = [col for col in merged.columns if col.startswith("CYTO_")]
X_raw = merged[clinical_cols + cyto_cols + mol_cols].fillna(0).values
scaler = StandardScaler().fit(X_raw)
X_scaled = scaler.transform(X_raw)
y = Surv.from_arrays(merged["OS_STATUS"].astype(bool), merged["OS_YEARS"])

# --- Train/val split ---
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=merged["OS_STATUS"], random_state=42
)

# --- Train RSF on raw features ---
rsf = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=42)
rsf.fit(X_train, y_train)

# --- Train Autoencoder + EST ---
autoencoder = train_autoencoder(X_train, mask_prob=0.33, epochs=200)
emb_train = get_embeddings(autoencoder, X_train)
emb_val = get_embeddings(autoencoder, X_val)

est = ExtraSurvivalTrees(n_estimators=204, max_features="sqrt", min_samples_split=3, min_samples_leaf=3, n_jobs=-1, random_state=42)
est.fit(emb_train, y_train)

# --- Cluster and evaluate model per cluster ---
X_cluster = np.vstack([X_train, X_val])
y_cluster = np.concatenate([y_train, y_val])
clusters = KMeans(n_clusters=4, random_state=42).fit_predict(X_cluster)

model_choices = []
for i in range(4):
    idx = clusters == i
    x_cl = X_cluster[idx]
    emb_cl = get_embeddings(autoencoder, x_cl)
    y_cl = y_cluster[idx]
    if len(y_cl) < 10:  # Skip small clusters
        model_choices.append("RSF")
        continue
    rsf_pred = rsf.predict(x_cl)
    est_pred = est.predict(emb_cl)
    c_rsf = concordance_index_ipcw(y_cl, y_cl, rsf_pred, tau=7)[0]
    c_est = concordance_index_ipcw(y_cl, y_cl, est_pred, tau=7)[0]
    model_choices.append("EST" if c_est > c_rsf else "RSF")

# --- Prepare test set ---
clinical_test = pd.read_csv("../data/clinical_test.csv")
molecular_test = pd.read_csv("../data/molecular_test.csv")
clinical_test = extract_cytogenetics_features(clinical_test)
for name, group in pathway_genes.items():
    clinical_test[f"PATHWAY_{name}"] = molecular_test[molecular_test["GENE"].isin(group)] \
        .groupby("ID").size().reindex(clinical_test["ID"], fill_value=0).gt(0).astype(int).values
mutation_count_test = molecular_test.groupby("ID").size().rename("MUTATION_COUNT")
avg_vaf_test = molecular_test.groupby("ID")["VAF"].mean().rename("VAF_AVG")
gene_flags_test = (
    molecular_test[molecular_test["GENE"].isin(top_genes)]
    .assign(flag=1)
    .pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
)
mol_feats_test = pd.concat([mutation_count_test, avg_vaf_test, gene_flags_test], axis=1).reset_index()
test_merged = clinical_test.merge(mol_feats_test, on="ID", how="left")
X_test = test_merged[clinical_cols + cyto_cols + mol_cols].fillna(0).values
X_test_scaled = scaler.transform(X_test)
test_clusters = KMeans(n_clusters=4, random_state=42).fit(X_cluster).predict(X_test_scaled)
emb_test = get_embeddings(autoencoder, X_test_scaled)

# --- Predict with best model per cluster ---
test_preds = []
for i, x in enumerate(X_test_scaled):
    cluster = test_clusters[i]
    if model_choices[cluster] == "RSF":
        test_preds.append(rsf.predict(x.reshape(1, -1))[0])
    else:
        test_preds.append(est.predict(emb_test[i].reshape(1, -1))[0])

# --- Save submission ---
submission = pd.DataFrame({
    "ID": clinical_test["ID"],
    "risk_score": test_preds
})
submission_path = "../submissions/submission_voting_clustered.csv"
submission.to_csv(submission_path, index=False)
print(f"✅ Submission saved to {submission_path}")
