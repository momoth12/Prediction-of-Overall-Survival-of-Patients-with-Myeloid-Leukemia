import optuna
import numpy as np
import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sksurv.ensemble import ExtraSurvivalTrees
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from tqdm import tqdm

# ========== DATA LOADING & PREPROCESSING ==========
cyto_markers = ["+8", "-7", "-5", "del(5q)", "del(7q)", "inv(3)", "t(3;3)", "t(6;9)",
                "complex", "monosomy 7", "trisomy 8", "t(8;21)", "inv(16)", "t(15;17)"]

def clean_cyto_text(text):
    if pd.isna(text): return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\d\+\-\(\);]", " ", text.lower())).strip()

def extract_cytogenetics_features(df):
    cyto_text = df["CYTOGENETICS"].fillna("").apply(clean_cyto_text)
    for marker in cyto_markers:
        safe_marker = marker.replace("(", "").replace(")", "").replace(";", "_").replace("+", "plus").replace("-", "minus")
        df[f"CYTO_{safe_marker}"] = cyto_text.apply(lambda s: int(marker.lower() in s))
    df["CYTO_COMPLEX_KARYOTYPE"] = cyto_text.apply(lambda s: int(len(re.findall(r'del|inv|t\(|\+|\-', s)) >= 3))
    return df

# --- Load and preprocess ---
clinical = extract_cytogenetics_features(pd.read_csv("../data/clinical_train.csv"))
target = pd.read_csv("../data/target_train.csv")
molecular = pd.read_csv("../data/molecular_train.csv")
data = clinical.merge(target, on="ID").dropna(subset=["OS_STATUS", "OS_YEARS"])

# --- Features ---
encoder = OneHotEncoder(handle_unknown='ignore')
X_cat = encoder.fit_transform(data[["CENTER"]]).toarray()
X_num = SimpleImputer().fit_transform(data[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
cyto_cols = [col for col in data.columns if col.startswith("CYTO_")]
X_cyto = data[cyto_cols].values
X_clinical = np.hstack([X_num, X_cat, X_cyto])

top_genes = molecular["GENE"].value_counts().head(15).index.tolist()
mutation_count = molecular.groupby("ID").size().rename("MUTATION_COUNT")
avg_vaf = molecular.groupby("ID")["VAF"].mean().rename("VAF_AVG")
gene_flags = molecular[molecular["GENE"].isin(top_genes)].assign(flag=1).pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)

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

mol_feats = pd.concat([mutation_count, avg_vaf, gene_flags], axis=1).reset_index()
merged = data.merge(mol_feats, on="ID", how="left")
mol_cols = ["MUTATION_COUNT", "VAF_AVG"] + top_genes + [col for col in data.columns if col.startswith("PATHWAY_")]
X_mol = SimpleImputer().fit_transform(merged[mol_cols])

X = np.hstack([X_clinical, X_mol])
y = Surv.from_arrays(merged["OS_STATUS"].astype(bool), merged["OS_YEARS"])
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=merged["OS_STATUS"], test_size=0.2, random_state=42)

# ========== OPTUNA HYPERPARAMETER TUNING ==========

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
        "n_jobs": -1
    }

    model = ExtraSurvivalTrees(**params)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_val)

    try:
        score = concordance_index_ipcw(y_train, y_val, preds_val, tau=7)[0]
    except:
        score = 0.0  # in case model fails to compute

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, show_progress_bar=True)

print("\n✅ Best C-index:", study.best_value)
print("🔧 Best Hyperparameters:", study.best_params)
