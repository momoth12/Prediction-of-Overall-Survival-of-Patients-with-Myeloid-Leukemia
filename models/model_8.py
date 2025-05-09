# clustered_rsf.py: RSF with Clustering + CYTOGENETICS + Pathway & Gene-Level Features

import numpy as np
import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

# --- CYTOGENETICS Feature Extraction ---
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
        df[colname] = cyto_text.str.contains(re.escape(marker.lower())).astype(int)
    df["CYTO_COMPLEX_KARYOTYPE"] = cyto_text.apply(lambda s: int(len(re.findall(r'del|inv|t\(|\+|\-', s)) >= 3))
    return df

# --- Load & merge ---
clinical = extract_cytogenetics_features(pd.read_csv("../data/clinical_train.csv"))
target = pd.read_csv("../data/target_train.csv")
molecular = pd.read_csv("../data/molecular_train.csv")
data = clinical.merge(target, on="ID").dropna(subset=["OS_STATUS", "OS_YEARS"])

# --- Clinical Features ---
encoder = OneHotEncoder(handle_unknown="ignore")
X_cat = encoder.fit_transform(data[["CENTER"]]).toarray()
X_num = SimpleImputer().fit_transform(data[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
cyto_cols = [col for col in data.columns if col.startswith("CYTO_")]
X_cyto = data[cyto_cols].values
X_clinical = np.hstack([X_num, X_cat, X_cyto])

# --- Molecular + Pathway Features ---
top_genes = molecular["GENE"].value_counts().head(25).index.tolist()
mutation_count = molecular.groupby("ID").size().rename("MUTATION_COUNT")
avg_vaf = molecular.groupby("ID")["VAF"].mean().rename("VAF_AVG")
max_vaf = molecular.groupby("ID")["VAF"].max().rename("VAF_MAX")
gene_vaf = molecular[molecular["GENE"].isin(top_genes)].pivot_table(index="ID", columns="GENE", values="VAF", aggfunc="mean")
gene_flags = gene_vaf.notna().astype(int)

# Pathway definitions
epigenetic = ['TET2', 'DNMT3A', 'IDH1', 'IDH2']
rtk = ['FLT3', 'KIT', 'NRAS', 'KRAS']
transcription = ['RUNX1', 'CEBPA', 'ETV6']
tumor_sup = ['TP53', 'WT1', 'NPM1']
pathway_dict = {
    "EPIGENETIC": epigenetic,
    "RTK": rtk,
    "TRANSCRIPTION": transcription,
    "TUMOR_SUPPRESSOR": tumor_sup
}

for name, genes in pathway_dict.items():
    subset = molecular[molecular["GENE"].isin(genes)]
    data[f"{name}_BINARY"] = subset.groupby("ID").size().reindex(data["ID"], fill_value=0).gt(0).astype(int).values
    data[f"{name}_COUNT"] = subset.groupby("ID").size().reindex(data["ID"], fill_value=0).values
    data[f"{name}_VAF"] = subset.groupby("ID")["VAF"].mean().reindex(data["ID"], fill_value=0).values
    data[f"{name}_INTERACT"] = data[f"{name}_COUNT"] * data[f"{name}_VAF"]
    total = np.array([len(set(gene_flags.columns) & set(genes))] * len(data))
    data[f"{name}_FRACTION"] = (data[f"{name}_COUNT"] / total).fillna(0).values

pathway_cols = [col for col in data.columns if any(col.startswith(p) for p in pathway_dict.keys())]
mol_feats = pd.concat([mutation_count, avg_vaf, max_vaf, gene_vaf, gene_flags], axis=1).reset_index()
merged = data.merge(mol_feats, on="ID", how="left")
mol_cols = ["MUTATION_COUNT", "VAF_AVG", "VAF_MAX"] + list(gene_vaf.columns) + list(gene_flags.columns) + pathway_cols
X_mol = SimpleImputer().fit_transform(merged[mol_cols])
X_all = np.hstack([X_clinical, X_mol])
y_struct = Surv.from_arrays(event=merged["OS_STATUS"].astype(bool), time=merged["OS_YEARS"])

# --- Cluster & train separate RSF ---
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_all)

all_preds = np.zeros(len(y_struct))
all_test_preds = []

clinical_test = extract_cytogenetics_features(pd.read_csv("../data/clinical_test.csv"))
molecular_test = pd.read_csv("../data/molecular_test.csv")
X_cat_test = encoder.transform(clinical_test[["CENTER"]]).toarray()
X_num_test = SimpleImputer().fit_transform(clinical_test[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
X_cyto_test = clinical_test[cyto_cols].values
X_clinical_test = np.hstack([X_num_test, X_cat_test, X_cyto_test])

for name, genes in pathway_dict.items():
    subset = molecular_test[molecular_test["GENE"].isin(genes)]
    clinical_test[f"{name}_BINARY"] = subset.groupby("ID").size().reindex(clinical_test["ID"], fill_value=0).gt(0).astype(int).values
    clinical_test[f"{name}_COUNT"] = subset.groupby("ID").size().reindex(clinical_test["ID"], fill_value=0).values
    clinical_test[f"{name}_VAF"] = subset.groupby("ID")["VAF"].mean().reindex(clinical_test["ID"], fill_value=0).values
    clinical_test[f"{name}_INTERACT"] = clinical_test[f"{name}_COUNT"] * clinical_test[f"{name}_VAF"]
    total = np.array([len(set(gene_flags.columns) & set(genes))] * len(clinical_test))
    clinical_test[f"{name}_FRACTION"] = (clinical_test[f"{name}_COUNT"] / total).fillna(0).values

mutation_count_test = molecular_test.groupby("ID").size().rename("MUTATION_COUNT")
avg_vaf_test = molecular_test.groupby("ID")["VAF"].mean().rename("VAF_AVG")
max_vaf_test = molecular_test.groupby("ID")["VAF"].max().rename("VAF_MAX")
gene_vaf_test = molecular_test[molecular_test["GENE"].isin(top_genes)].pivot_table(index="ID", columns="GENE", values="VAF", aggfunc="mean")
gene_flags_test = gene_vaf_test.notna().astype(int)
mol_feats_test = pd.concat([mutation_count_test, avg_vaf_test, max_vaf_test, gene_vaf_test, gene_flags_test], axis=1).reset_index()
test_merged = clinical_test.merge(mol_feats_test, on="ID", how="left")
X_mol_test = SimpleImputer().fit_transform(test_merged[mol_cols])
X_test_final = np.hstack([X_clinical_test, X_mol_test])

for c in range(3):
    mask = clusters == c
    X_c, y_c = X_all[mask], y_struct[mask]
    rsf = RandomSurvivalForest(n_estimators=300, min_samples_split=5, min_samples_leaf=10,
                               max_features="sqrt", random_state=42, n_jobs=-1)
    rsf.fit(X_c, y_c)
    all_preds[mask] = rsf.predict(X_c)
    all_test_preds.append(rsf.predict(X_test_final))

cidx = concordance_index_ipcw(y_struct, y_struct, all_preds, tau=7)[0]
print(f"✅ Clustered RSF IPCW C-index: {cidx:.4f}")

# Average test risk scores from each cluster model
submission = pd.DataFrame({
    "ID": clinical_test["ID"],
    "risk_score": np.mean(all_test_preds, axis=0)
})
submission_file = "../submissions/submission_model8_clustered.csv"
submission.to_csv(submission_file, index=False)
print(f"✅ Submission saved as {submission_file}")
