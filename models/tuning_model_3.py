
# rsf_tuning_cytogenetics.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, RegressorMixin
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
import re

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
    df["CYTO_COMPLEX_KARYOTYPE"] = cyto_text.apply(lambda s: int(len(re.findall(r"del|inv|t\(|\+|\-", s)) >= 3))
    return df

# --- Load and preprocess training data ---
clinical = pd.read_csv("../data/clinical_train.csv")
target = pd.read_csv("../data/target_train.csv")
molecular = pd.read_csv("../data/molecular_train.csv")
clinical = extract_cytogenetics_features(clinical)
data = clinical.merge(target, on="ID").dropna(subset=["OS_STATUS", "OS_YEARS"])

# Clinical preprocessing
encoder = OneHotEncoder(handle_unknown='ignore')
X_cat = encoder.fit_transform(data[["CENTER"]]).toarray()
X_num = SimpleImputer(strategy="mean").fit_transform(
    data[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
cyto_cols = [col for col in data.columns if col.startswith("CYTO_")]
X_cyto = data[cyto_cols].values
X_clinical = np.hstack([X_num, X_cat, X_cyto])

# Molecular feature engineering
top_genes = molecular["GENE"].value_counts().head(10).index.tolist()
mutation_count = molecular.groupby("ID").size().rename("MUTATION_COUNT")
avg_vaf = molecular.groupby("ID")["VAF"].mean().rename("VAF_AVG")
gene_flags = (
    molecular[molecular["GENE"].isin(top_genes)]
    .assign(flag=1)
    .pivot_table(index="ID", columns="GENE", values="flag", fill_value=0)
)
mol_feats = pd.concat([mutation_count, avg_vaf, gene_flags], axis=1).reset_index()
merged = data.merge(mol_feats, on="ID", how="left")
mol_cols = ['MUTATION_COUNT', 'VAF_AVG'] + top_genes
X_mol = SimpleImputer(strategy="mean").fit_transform(merged[mol_cols])
X = np.hstack([X_clinical, X_mol])
y_struct = Surv.from_arrays(event=merged["OS_STATUS"].astype(bool), time=merged["OS_YEARS"])

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_struct, test_size=0.2, random_state=42, stratify=merged["OS_STATUS"]
)

# --- RSF Wrapper for sklearn compatibility ---
class RSFWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, min_samples_split=10, min_samples_leaf=15, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=-1,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

# --- Custom scoring function using IPCW C-index ---
def ipcw_score(estimator, X, y_true):
    y_pred = estimator.predict(X)
    return concordance_index_ipcw(y_true, y_true, y_pred, tau=7)[0]

# --- Parameter grid ---
param_grid = {
    "n_estimators": [100, 200, 300],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [5, 10],
    "max_features": ["sqrt", "log2"]
}

# --- Grid Search ---
rsf_cv = GridSearchCV(
    estimator=RSFWrapper(),
    param_grid=param_grid,
    scoring=ipcw_score,
    cv=3,
    verbose=2,
    n_jobs=-1
)

rsf_cv.fit(X_train, y_train)

# --- Results ---
print("✅ Best params:", rsf_cv.best_params_)
print("✅ Best IPCW C-index (CV):", rsf_cv.best_score_)
