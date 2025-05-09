import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# --- CYTOGENETICS Feature Extraction ---
cyto_markers = [
    "+8", "-7", "-5", "del(5q)", "del(7q)", "inv(3)", "t(3;3)", "t(6;9)",
    "complex", "monosomy 7", "trisomy 8", "t(8;21)", "inv(16)", "t(15;17)"
]

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

# --- Load datasets ---
clinical = extract_cytogenetics_features(pd.read_csv("../data/clinical_train.csv"))
target = pd.read_csv("../data/target_train.csv")
molecular = pd.read_csv("../data/molecular_train.csv")
data = clinical.merge(target, on="ID").dropna(subset=["OS_STATUS", "OS_YEARS"])

# --- Encode clinical features ---
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = encoder.fit_transform(data[["CENTER"]])
X_num = SimpleImputer().fit_transform(data[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
cyto_cols = [col for col in data.columns if col.startswith("CYTO_")]
X_cyto = data[cyto_cols].values
X_clinical = np.hstack([X_num, X_cat, X_cyto])
clinical_feature_names = (
    [f"NUM_{col}" for col in ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]] +
    list(encoder.get_feature_names_out(["CENTER"])) +
    cyto_cols
)

# --- Molecular features ---
top_genes = molecular["GENE"].value_counts().head(30).index.tolist()
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
pathway_dict = {
    "EPIGENETIC": epigenetic,
    "RTK": rtk,
    "TRANSCRIPTION": transcription,
    "TUMOR_SUPPRESSOR": tumor_sup
}
for name, genes in pathway_dict.items():
    data[f"PATHWAY_{name}"] = molecular[molecular["GENE"].isin(genes)] \
        .groupby("ID").size().reindex(data["ID"], fill_value=0).gt(0).astype(int).values

# --- Merge molecular features ---
mol_feats = pd.concat([mutation_count, avg_vaf, gene_flags], axis=1).reset_index()
merged = data.merge(mol_feats, on="ID", how="left")
pathway_cols = [col for col in data.columns if col.startswith("PATHWAY_")]
mol_cols = ["MUTATION_COUNT", "VAF_AVG"] + top_genes + pathway_cols
X_mol = SimpleImputer().fit_transform(merged[mol_cols])

# --- Final feature matrix ---
X = np.hstack([X_clinical, X_mol])
final_features = clinical_feature_names + mol_cols
df_full = pd.DataFrame(X, columns=final_features)
df_full["OS_STATUS"] = merged["OS_STATUS"].values
df_full["OS_YEARS"] = merged["OS_YEARS"].values

df_full.to_csv("../data/merged_train.csv", index=False)
print("✅ Saved: ../data/merged_train.csv")

# --- Test set (optional) ---
clinical_test = extract_cytogenetics_features(pd.read_csv("../data/clinical_test.csv"))
molecular_test = pd.read_csv("../data/molecular_test.csv")

X_cat_test = encoder.transform(clinical_test[["CENTER"]])
X_num_test = SimpleImputer().fit_transform(clinical_test[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
X_cyto_test = clinical_test[cyto_cols].values
X_clinical_test = np.hstack([X_num_test, X_cat_test, X_cyto_test])

for name, genes in pathway_dict.items():
    clinical_test[f"PATHWAY_{name}"] = molecular_test[molecular_test["GENE"].isin(genes)] \
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
X_mol_test = SimpleImputer().fit_transform(test_merged[mol_cols])
X_test_final = np.hstack([X_clinical_test, X_mol_test])
df_test = pd.DataFrame(X_test_final, columns=final_features)
df_test["ID"] = clinical_test["ID"]
df_test.to_csv("../data/merged_test.csv", index=False)
print("✅ Saved: ../data/merged_test.csv")
