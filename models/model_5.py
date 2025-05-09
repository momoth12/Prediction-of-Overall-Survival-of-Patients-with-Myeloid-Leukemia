
# deepsurv_model_full_features.py — DeepSurv using clinical + CYTOGENETICS + gene pathway features

import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pycox.models import CoxPH
import torchtuples as tt
import torch
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
import warnings
warnings.filterwarnings("ignore")

# --- CYTOGENETICS processing ---
cyto_markers = [
    "+8", "-7", "-5", "del(5q)", "del(7q)", "inv(3)", "t(3;3)", "t(6;9)",
    "complex", "monosomy 7", "trisomy 8", "t(8;21)", "inv(16)", "t(15;17)"
]

def clean_cyto_text(text):
    if pd.isna(text): return ""
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

# --- Load and preprocess ---
clinical = pd.read_csv("../data/clinical_train.csv")
target = pd.read_csv("../data/target_train.csv")
molecular = pd.read_csv("../data/molecular_train.csv")
clinical = extract_cytogenetics_features(clinical)
data = clinical.merge(target, on="ID").dropna(subset=["OS_STATUS", "OS_YEARS"])

# --- Clinical ---
encoder = OneHotEncoder(handle_unknown='ignore')
X_cat = encoder.fit_transform(data[["CENTER"]]).toarray()
X_num = SimpleImputer(strategy="mean").fit_transform(data[["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]])
cyto_cols = [col for col in data.columns if col.startswith("CYTO_")]
X_cyto = data[cyto_cols].values

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
    data[f"PATHWAY_{name}"] = molecular[molecular["GENE"].isin(group)]         .groupby("ID").size().reindex(data["ID"], fill_value=0).gt(0).astype(int).values
pathway_cols = [col for col in data.columns if col.startswith("PATHWAY_")]

mol_feats = pd.concat([mutation_count, avg_vaf, gene_flags], axis=1).reset_index()
merged = data.merge(mol_feats, on="ID", how="left")
mol_cols = ['MUTATION_COUNT', 'VAF_AVG'] + top_genes + pathway_cols
X_mol = SimpleImputer(strategy="mean").fit_transform(merged[mol_cols])

# --- Combine all features ---
X_all = np.hstack([X_num, X_cat, X_cyto, X_mol])
X_all = X_all.astype("float32")
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

# --- Labels ---
durations = merged["OS_YEARS"].values
events = merged["OS_STATUS"].astype(bool).values

# --- Train/Test split ---
X_train, X_val, y_train_dur, y_val_dur, y_train_evt, y_val_evt = train_test_split(
    X_all, durations, events, test_size=0.2, random_state=42, stratify=events)

train_data = (X_train, (y_train_dur, y_train_evt))
val_data = (X_val, (y_val_dur, y_val_evt))
# --- Define DeepSurv Model ---
net = torch.nn.Sequential(
    torch.nn.Linear(X_all.shape[1], 64),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(64),
    torch.nn.Dropout(0.1),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
)

model = CoxPH(net, tt.optim.Adam)
batch_size = 128
epochs = 200
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

# --- Fit ---
model.fit(train_data[0], train_data[1], batch_size, epochs, callbacks, verbose, val_data=val_data)

# --- Evaluate ---
surv = Surv.from_arrays(event=y_train_evt, time=y_train_dur)
surv_val = Surv.from_arrays(event=y_val_evt, time=y_val_dur)
pred_train = -model.predict(X_train).reshape(-1)
pred_val = -model.predict(X_val).reshape(-1)

c_train = concordance_index_ipcw(surv, surv, pred_train, tau=7)[0]
c_val = concordance_index_ipcw(surv, surv_val, pred_val, tau=7)[0]

print(f"[DeepSurv-Full] IPCW C-index (train): {c_train:.4f}")
print(f"[DeepSurv-Full] IPCW C-index (val)  : {c_val:.4f}")
