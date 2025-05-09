import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

# ------------------------- CYTOGENETICS -------------------------
cyto_markers = [
    "+8", "-7", "-5", "del(5q)", "del(7q)", "inv(3)", "t(3;3)", "t(6;9)",
    "complex", "monosomy 7", "trisomy 8", "t(8;21)", "inv(16)", "t(15;17)",
    "del(17)", "t(16;16)"
]

def clean_cyto_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r"[^\w\d\+\-\(\);]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_cytogenetics_features(df):
    cyto_text = df["CYTOGENETICS"].fillna("").apply(clean_cyto_text)
    for marker in cyto_markers:
        colname = f"CYTO_{marker.replace('(', '').replace(')', '').replace(';', '_').replace('+','plus').replace('-', 'minus')}"
        df[colname] = cyto_text.str.contains(re.escape(marker.lower())).astype(int)
    df["CYTO_COMPLEX_KARYOTYPE"] = cyto_text.apply(lambda s: int(len(re.findall(r'del|inv|t\(|\+|\-', s)) >= 3))
    return df

# ------------------------- Load and Merge -------------------------
clinical = extract_cytogenetics_features(pd.read_csv("../data/clinical_train.csv"))
target = pd.read_csv("../data/target_train.csv")
molecular = pd.read_csv("../data/molecular_train.csv")
data = clinical.merge(target, on="ID").dropna(subset=["OS_STATUS", "OS_YEARS"])

# ------------------------- Clinical Features -------------------------
encoder = OneHotEncoder(handle_unknown="ignore")
X_cat = encoder.fit_transform(data[["CENTER"]]).toarray()
X_num = SimpleImputer().fit_transform(data[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
cyto_cols = [col for col in data.columns if col.startswith("CYTO_")]
X_cyto = data[cyto_cols].values
X_clinical = np.hstack([X_num, X_cat, X_cyto])

# ------------------------- Genetic Features -------------------------
top_genes = molecular["GENE"].value_counts().head(30).index.tolist()
mutation_count = molecular.groupby("ID").size().rename("MUTATION_COUNT")
avg_vaf = molecular.groupby("ID")["VAF"].mean().rename("VAF_AVG")
gene_flags = (
    molecular[molecular["GENE"].isin(top_genes)]
    .assign(flag=1)
    .pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
)
gene_vaf = (
    molecular[molecular["GENE"].isin(top_genes)]
    .pivot_table(index="ID", columns="GENE", values="VAF", aggfunc="mean")
    .add_prefix("VAF_")
)

# ------------------------- Pathways -------------------------
pathways = {
    "EPIGENETIC": ['TET2', 'DNMT3A', 'IDH1', 'IDH2'],
    "RTK": ['FLT3', 'KIT', 'NRAS', 'KRAS'],
    "TRANSCRIPTION": ['RUNX1', 'CEBPA', 'ETV6'],
    "TUMOR_SUPPRESSOR": ['TP53', 'WT1', 'NPM1']
}

for name, genes in pathways.items():
    mask = molecular["GENE"].isin(genes)
    path_data = molecular[mask].groupby("ID")
    data[f"PATHWAY_{name}_MUT_COUNT"] = path_data.size().reindex(data["ID"], fill_value=0)
    data[f"PATHWAY_{name}_VAF_MEAN"] = path_data["VAF"].mean().reindex(data["ID"], fill_value=0)
    data[f"PATHWAY_{name}_FRAC_MUT"] = (
        path_data["GENE"].nunique().reindex(data["ID"], fill_value=0) / len(genes)
    )
    data[f"PATHWAY_{name}_INTERACTION"] = data[f"PATHWAY_{name}_MUT_COUNT"] * data[f"PATHWAY_{name}_VAF_MEAN"]

# ------------------------- Mutation Interactions -------------------------
co_occurrence = molecular[molecular["GENE"].isin(["FLT3", "NPM1"])].assign(flag=1)
co_occurrence = co_occurrence.pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
co_occurrence["FLT3_NPM1"] = ((co_occurrence.get("FLT3", 0) > 0) & (co_occurrence.get("NPM1", 0) > 0)).astype(int)

# ------------------------- Merge -------------------------
mol_feats = pd.concat([mutation_count, avg_vaf, gene_flags, gene_vaf, co_occurrence], axis=1).reset_index()
merged = data.merge(mol_feats, on="ID", how="left")

mol_cols = (
    ["MUTATION_COUNT", "VAF_AVG"]
    + top_genes
    + [f"VAF_{g}" for g in top_genes]
    + [c for c in data.columns if c.startswith("PATHWAY_")]
    + ["FLT3_NPM1"]
)
X_mol = SimpleImputer().fit_transform(merged[mol_cols])
X = np.hstack([X_clinical, X_mol])
y_struct = Surv.from_arrays(event=merged["OS_STATUS"].astype(bool), time=merged["OS_YEARS"])

# ------------------------- Train/Val Split -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_struct, test_size=0.2, random_state=42, stratify=merged["OS_STATUS"]
)

# ------------------------- Train RSF -------------------------
rsf = RandomSurvivalForest(
    n_estimators=300, min_samples_split=5, min_samples_leaf=10,
    max_features="sqrt", n_jobs=-1, random_state=42
)
rsf.fit(X_train, y_train)

def evaluate_ipcw(y_train, y_val, pred_train, pred_val, name):
    cidx_train = concordance_index_ipcw(y_train, y_train, pred_train, tau=7)[0]
    cidx_val = concordance_index_ipcw(y_train, y_val, pred_val, tau=7)[0]
    print(f"[{name}] IPCW C-index (train): {cidx_train:.4f}")
    print(f"[{name}] IPCW C-index (val)  : {cidx_val:.4f}")

evaluate_ipcw(y_train, y_val, rsf.predict(X_train), rsf.predict(X_val), "RSF Tuned")

# ------------------------- Retrain on Full -------------------------
rsf_final = RandomSurvivalForest(
    n_estimators=300, min_samples_split=5, min_samples_leaf=10,
    max_features="sqrt", n_jobs=-1, random_state=42
)
rsf_final.fit(X, y_struct)

# ------------------------- Submission -------------------------
clinical_test = extract_cytogenetics_features(pd.read_csv("../data/clinical_test.csv"))
molecular_test = pd.read_csv("../data/molecular_test.csv")

# Repeat the same preprocessing for test
X_cat_test = encoder.transform(clinical_test[["CENTER"]]).toarray()
X_num_test = SimpleImputer().fit_transform(clinical_test[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
X_cyto_test = clinical_test[cyto_cols].values
X_clinical_test = np.hstack([X_num_test, X_cat_test, X_cyto_test])

# Pathway and interaction features
for name, genes in pathways.items():
    mask = molecular_test["GENE"].isin(genes)
    path_data = molecular_test[mask].groupby("ID")
    clinical_test[f"PATHWAY_{name}_MUT_COUNT"] = path_data.size().reindex(clinical_test["ID"], fill_value=0)
    clinical_test[f"PATHWAY_{name}_VAF_MEAN"] = path_data["VAF"].mean().reindex(clinical_test["ID"], fill_value=0)
    clinical_test[f"PATHWAY_{name}_FRAC_MUT"] = (
        path_data["GENE"].nunique().reindex(clinical_test["ID"], fill_value=0) / len(genes)
    )
    clinical_test[f"PATHWAY_{name}_INTERACTION"] = (
        clinical_test[f"PATHWAY_{name}_MUT_COUNT"] * clinical_test[f"PATHWAY_{name}_VAF_MEAN"]
    )

# Genetic
gene_flags_test = (
    molecular_test[molecular_test["GENE"].isin(top_genes)]
    .assign(flag=1)
    .pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
)
gene_vaf_test = (
    molecular_test[molecular_test["GENE"].isin(top_genes)]
    .pivot_table(index="ID", columns="GENE", values="VAF", aggfunc="mean")
    .add_prefix("VAF_")
)
mutation_count_test = molecular_test.groupby("ID").size().rename("MUTATION_COUNT")
avg_vaf_test = molecular_test.groupby("ID")["VAF"].mean().rename("VAF_AVG")

co_occ_test = molecular_test[molecular_test["GENE"].isin(["FLT3", "NPM1"])].assign(flag=1)
co_occ_test = co_occ_test.pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
co_occ_test["FLT3_NPM1"] = ((co_occ_test.get("FLT3", 0) > 0) & (co_occ_test.get("NPM1", 0) > 0)).astype(int)

mol_feats_test = pd.concat([
    mutation_count_test, avg_vaf_test,
    gene_flags_test, gene_vaf_test, co_occ_test
], axis=1).reset_index()

test_merged = clinical_test.merge(mol_feats_test, on="ID", how="left")
X_mol_test = SimpleImputer().fit_transform(test_merged[mol_cols])
X_test_final = np.hstack([X_clinical_test, X_mol_test])

risk_scores = rsf_final.predict(X_test_final)
submission = pd.DataFrame({"ID": clinical_test["ID"], "risk_score": risk_scores})
submission.to_csv("../submissions/submission_enhanced_model4.csv", index=False)
print("✅ Submission saved: submission_enhanced_model4.csv")
