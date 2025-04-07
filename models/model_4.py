# model4.py: RSF + CYTOGENETICS + Gene Pathway Features + Feature Importance Plot

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt

# --- CYTOGENETICS Feature Extraction ---
cyto_markers = [
    "+8", "-7", "-5", "del(5q)", "del(7q)", "inv(3)", "t(3;3)", "t(6;9)",
    "complex", "monosomy 7", "trisomy 8", "t(8;21)", "inv(16)", "t(15;17)"
]

def clean_cyto_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\d\+\-\(\);]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_cytogenetics_features(df):
    cyto_text = df["CYTOGENETICS"].fillna("").apply(clean_cyto_text)
    for marker in cyto_markers:
        colname = f"CYTO_{marker.replace('(', '').replace(')', '').replace(';', '_').replace('+', 'plus').replace('-', 'minus')}"
        df[colname] = cyto_text.apply(lambda s: int(marker.lower() in s))
    df["CYTO_COMPLEX_KARYOTYPE"] = cyto_text.apply(lambda s: int(len(re.findall(r'del|inv|t\(|\+|\-', s)) >= 3))

    return df

# --- Load data ---
clinical = pd.read_csv("../data/clinical_train.csv")
target = pd.read_csv("../data/target_train.csv")
molecular = pd.read_csv("../data/molecular_train.csv")
clinical = extract_cytogenetics_features(clinical)
data = clinical.merge(target, on="ID").dropna(subset=["OS_STATUS", "OS_YEARS"])

# --- Clinical features ---
encoder = OneHotEncoder(handle_unknown='ignore')
X_cat = encoder.fit_transform(data[["CENTER"]]).toarray()
X_num = SimpleImputer(strategy="mean").fit_transform(data[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
cyto_cols = [col for col in data.columns if col.startswith("CYTO_")]
X_cyto = data[cyto_cols].values
X_clinical = np.hstack([X_num, X_cat, X_cyto])

# --- Molecular features ---
top_genes = molecular["GENE"].value_counts().head(10).index.tolist()
mutation_count = molecular.groupby("ID").size().rename("MUTATION_COUNT")
avg_vaf = molecular.groupby("ID")["VAF"].mean().rename("VAF_AVG")
gene_flags = (
    molecular[molecular["GENE"].isin(top_genes)]
    .assign(flag=1)
    .pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
)

# --- Pathway features ---
epigenetic = ['TET2', 'DNMT3A', 'IDH1', 'IDH2']
rtk = ['FLT3', 'KIT', 'NRAS', 'KRAS']
transcription = ['RUNX1', 'CEBPA', 'ETV6']
tumor_sup = ['TP53', 'WT1', 'NPM1']

for name, group in {
    "EPIGENETIC": epigenetic,
    "RTK": rtk,
    "TRANSCRIPTION": transcription,
    "TUMOR_SUPPRESSOR": tumor_sup
}.items():
    data[f"PATHWAY_{name}"] = molecular[molecular["GENE"].isin(group)] \
        .groupby("ID").size().reindex(data["ID"], fill_value=0).gt(0).astype(int).values

pathway_cols = [col for col in data.columns if col.startswith("PATHWAY_")]
mol_feats = pd.concat([mutation_count, avg_vaf, gene_flags], axis=1).reset_index()
merged = data.merge(mol_feats, on="ID", how="left")
mol_cols = ['MUTATION_COUNT', 'VAF_AVG'] + top_genes + pathway_cols
X_mol = SimpleImputer(strategy="mean").fit_transform(merged[mol_cols])
X = np.hstack([X_clinical, X_mol])
y_struct = Surv.from_arrays(event=merged["OS_STATUS"].astype(bool), time=merged["OS_YEARS"])

# --- Train/val split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y_struct, test_size=0.2, random_state=42, stratify=merged["OS_STATUS"]
)

def evaluate_ipcw(y_train, y_val, pred_train, pred_val, name):
    cidx_train = concordance_index_ipcw(y_train, y_train, pred_train, tau=7)[0]
    cidx_val = concordance_index_ipcw(y_train, y_val, pred_val, tau=7)[0]
    print(f"[{name}] IPCW C-index (train): {cidx_train:.4f}")
    print(f"[{name}] IPCW C-index (val)  : {cidx_val:.4f}")

# --- Train RSF ---
rsf = RandomSurvivalForest(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=10,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
rsf.fit(X_train, y_train)
evaluate_ipcw(y_train, y_val, rsf.predict(X_train), rsf.predict(X_val), "RSF Tuned")

# --- Retrain on all data ---
rsf_final = RandomSurvivalForest(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=10,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
rsf_final.fit(X, y_struct)

# --- Feature Importance Plot ---
feature_names = (
    [f"NUM_{col}" for col in ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]] +
    list(encoder.get_feature_names_out(["CENTER"])) +
    cyto_cols + mol_cols
)
from sklearn.inspection import permutation_importance

def ipcw_scorer(estimator, X, y):
    return concordance_index_ipcw(y_train, y, estimator.predict(X), tau=7)[0]

result = permutation_importance(
    rsf_final,
    X_val,
    y_val,
    n_repeats=5,
    random_state=42,
    scoring=ipcw_scorer
)

importances = result.importances_mean
indices = np.argsort(importances)[::-1][:20]


plt.figure(figsize=(10, 6))
plt.title("Top 20 Feature Importances (RSF)")
plt.barh(range(len(indices)), importances[indices][::-1], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# --- Prepare test set ---
clinical_test = pd.read_csv("../data/clinical_test.csv")
molecular_test = pd.read_csv("../data/molecular_test.csv")
clinical_test = extract_cytogenetics_features(clinical_test)

X_cat_test = encoder.transform(clinical_test[["CENTER"]]).toarray()
X_num_test = SimpleImputer(strategy="mean").fit_transform(clinical_test[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
X_cyto_test = clinical_test[cyto_cols].values
X_clinical_test = np.hstack([X_num_test, X_cat_test, X_cyto_test])

for name, group in {
    "EPIGENETIC": epigenetic,
    "RTK": rtk,
    "TRANSCRIPTION": transcription,
    "TUMOR_SUPPRESSOR": tumor_sup
}.items():
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
X_mol_test = SimpleImputer(strategy="mean").fit_transform(test_merged[mol_cols])
X_test_final = np.hstack([X_clinical_test, X_mol_test])

# --- Predict and save ---
risk_scores = rsf_final.predict(X_test_final)
submission = pd.DataFrame({
    "ID": clinical_test["ID"],
    "risk_score": risk_scores
})
submission_dir = "../submissions/submission_model_4_pathways.csv"
submission.to_csv(submission_dir, index=False)
print(f"✅ Submission saved as {submission_dir}")
