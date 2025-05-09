import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
import re
import matplotlib.pyplot as plt

# --- Cytogenetics ---
cyto_markers = ["+8", "-7", "-5", "del(5q)", "del(7q)", "inv(3)", "t(3;3)", "t(6;9)", 
                "complex", "monosomy 7", "trisomy 8", "t(8;21)", "inv(16)", "t(15;17)"]

def clean_cyto_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r"[^\w\d\+\-\(\);]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_cytogenetics_features(df):
    cyto_text = df["CYTOGENETICS"].fillna("").apply(clean_cyto_text)
    for marker in cyto_markers:
        colname = f"CYTO_{marker.replace('(', '').replace(')', '').replace(';', '_').replace('+', 'plus').replace('-', 'minus')}"
        df[colname] = cyto_text.apply(lambda s: int(marker.lower() in s))
    df["CYTO_COMPLEX_KARYOTYPE"] = cyto_text.apply(lambda s: int(len(re.findall(r'del|inv|t\(|\+|\-', s)) >= 3))
    return df

# --- Load Data ---
clinical = extract_cytogenetics_features(pd.read_csv("../data/clinical_train.csv"))
target = pd.read_csv("../data/target_train.csv")
molecular = pd.read_csv("../data/molecular_train.csv")
data = clinical.merge(target, on="ID").dropna(subset=["OS_STATUS", "OS_YEARS"])

# --- Pathway Genes ---
pathway_dict = {
    "EPIGENETIC": ['TET2', 'DNMT3A', 'IDH1', 'IDH2'],
    "RTK": ['FLT3', 'KIT', 'NRAS', 'KRAS'],
    "TRANSCRIPTION": ['RUNX1', 'CEBPA', 'ETV6'],
    "TUMOR_SUPPRESSOR": ['TP53', 'WT1', 'NPM1']
}
all_pathway_genes = sum(pathway_dict.values(), [])

# --- Gene Flags, VAF, Mutation Count ---
top_genes = molecular["GENE"].value_counts().head(25).index.tolist()
gene_flags = molecular[molecular["GENE"].isin(top_genes)].assign(flag=1).pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
mutation_count = molecular.groupby("ID").size().rename("MUTATION_COUNT")
vaf_mean = molecular.groupby("ID")["VAF"].mean().rename("VAF_MEAN")

# --- Pairwise Co-Mutations ---
co_occurrence = gene_flags.copy()
for pair in [("FLT3", "NPM1"), ("TP53", "RUNX1")]:
    if all(g in gene_flags.columns for g in pair):
        co_occurrence[f"{pair[0]}_{pair[1]}"] = (gene_flags[pair[0]] == 1) & (gene_flags[pair[1]] == 1)

# --- Pathway Aggregates ---
pathway_feats = pd.DataFrame(index=data["ID"])
for name, genes in pathway_dict.items():
    df_sub = molecular[molecular["GENE"].isin(genes)]
    pathway_feats[f"PATHWAY_{name}_MEAN_VAF"] = df_sub.groupby("ID")["VAF"].mean()
    pathway_feats[f"PATHWAY_{name}_FRAC_MUT"] = df_sub.groupby("ID")["GENE"].nunique() / len(genes)
    pathway_feats[f"PATHWAY_{name}_COUNT"] = df_sub.groupby("ID").size()
    pathway_feats[f"PATHWAY_{name}_INTERACTION"] = pathway_feats[f"PATHWAY_{name}_MEAN_VAF"] * pathway_feats[f"PATHWAY_{name}_COUNT"]

# --- Merge All ---
mol_feats = pd.concat([gene_flags, co_occurrence, mutation_count, vaf_mean, pathway_feats], axis=1).reset_index()
merged = data.merge(mol_feats, on="ID", how="left")
mol_cols = mol_feats.columns.drop("ID").tolist()

# --- Clinical Features ---
encoder = OneHotEncoder(handle_unknown='ignore')
X_cat = encoder.fit_transform(merged[["CENTER"]]).toarray()
X_num = SimpleImputer(strategy="mean").fit_transform(merged[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
cyto_cols = [col for col in merged.columns if col.startswith("CYTO_")]
X_cyto = merged[cyto_cols].values
X_clinical = np.hstack([X_num, X_cat, X_cyto])

# --- Combine All ---
X_mol = SimpleImputer(strategy="mean").fit_transform(merged[mol_cols])
X = np.hstack([X_clinical, X_mol])
y_struct = Surv.from_arrays(event=merged["OS_STATUS"].astype(bool), time=merged["OS_YEARS"])

# --- Train/Val ---
X_train, X_val, y_train, y_val = train_test_split(X, y_struct, test_size=0.2, stratify=merged["OS_STATUS"], random_state=42)

def evaluate_ipcw(y_train, y_val, pred_train, pred_val, name):
    cidx_train = concordance_index_ipcw(y_train, y_train, pred_train, tau=7)[0]
    cidx_val = concordance_index_ipcw(y_train, y_val, pred_val, tau=7)[0]
    print(f"[{name}] IPCW C-index (train): {cidx_train:.4f}")
    print(f"[{name}] IPCW C-index (val)  : {cidx_val:.4f}")

# --- RSF ---
rsf = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10,
                           max_features="sqrt", n_jobs=-1, random_state=42)
rsf.fit(X_train, y_train)
evaluate_ipcw(y_train, y_val, rsf.predict(X_train), rsf.predict(X_val), "RSF")

# --- Retrain on all ---
rsf_final = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10,
                                 max_features="sqrt", n_jobs=-1, random_state=42)
rsf_final.fit(X, y_struct)

# --- Submission ---
clinical_test = extract_cytogenetics_features(pd.read_csv("../data/clinical_test.csv"))
molecular_test = pd.read_csv("../data/molecular_test.csv")
test = clinical_test.copy()

# Clinical test
X_cat_test = encoder.transform(test[["CENTER"]]).toarray()
X_num_test = SimpleImputer().fit_transform(test[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
X_cyto_test = test[cyto_cols].values
X_clinical_test = np.hstack([X_num_test, X_cat_test, X_cyto_test])

# Molecular test
gene_flags_test = molecular_test[molecular_test["GENE"].isin(top_genes)].assign(flag=1).pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
co_occurrence_test = gene_flags_test.copy()
for pair in [("FLT3", "NPM1"), ("TP53", "RUNX1")]:
    if all(g in gene_flags_test.columns for g in pair):
        co_occurrence_test[f"{pair[0]}_{pair[1]}"] = (gene_flags_test[pair[0]] == 1) & (gene_flags_test[pair[1]] == 1)

mutation_count_test = molecular_test.groupby("ID").size().rename("MUTATION_COUNT")
vaf_mean_test = molecular_test.groupby("ID")["VAF"].mean().rename("VAF_MEAN")

pathway_feats_test = pd.DataFrame(index=test["ID"])
for name, genes in pathway_dict.items():
    df_sub = molecular_test[molecular_test["GENE"].isin(genes)]
    pathway_feats_test[f"PATHWAY_{name}_MEAN_VAF"] = df_sub.groupby("ID")["VAF"].mean()
    pathway_feats_test[f"PATHWAY_{name}_FRAC_MUT"] = df_sub.groupby("ID")["GENE"].nunique() / len(genes)
    pathway_feats_test[f"PATHWAY_{name}_COUNT"] = df_sub.groupby("ID").size()
    pathway_feats_test[f"PATHWAY_{name}_INTERACTION"] = pathway_feats_test[f"PATHWAY_{name}_MEAN_VAF"] * pathway_feats_test[f"PATHWAY_{name}_COUNT"]

mol_feats_test = pd.concat([gene_flags_test, co_occurrence_test, mutation_count_test, vaf_mean_test, pathway_feats_test], axis=1).reset_index()
test_merged = test.merge(mol_feats_test, on="ID", how="left")
X_mol_test = SimpleImputer().fit_transform(test_merged[mol_cols])
X_test = np.hstack([X_clinical_test, X_mol_test])

# Predict & Save
submission = pd.DataFrame({
    "ID": test["ID"],
    "risk_score": rsf_final.predict(X_test)
})
submission_path = "../submissions/submission_model4bis.csv"
submission.to_csv(submission_path, index=False)
print(f"✅ Submission saved as {submission_path}")
