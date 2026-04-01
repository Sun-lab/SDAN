"""
SEA_AD_spectra_preprocess.py
Prepare SEA_AD dataset for Spectra training:
 - Load SEA_AD data
 - Label Dementia vs No Dementia
 - Split donors into train/val/test
 - Perform QC and DE-gene selection
 - Save filtered AnnData for Spectra
"""

import os, math, warnings
import numpy as np
import pandas as pd
import scanpy as sc

from SDAN.args import parse_args
from SDAN.preprocess import qc, construct_gene_list

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(888)
sc.settings.verbosity = 2
sc.settings.set_figure_params(figsize=(8, 6), dpi=80)

# ---------------------------------------------------------------------
# Paths and args
# ---------------------------------------------------------------------
args = parse_args()
d = "./SEA_AD/"
adata_path = f"{d}data/{args.cell_type}.h5ad"
donor_xlsx = f"{d}data/sea-ad_cohort_donor_metadata_020624.xlsx"
os.makedirs(f"{d}output_comparsion/Spectra/", exist_ok=True)


# ---------------------------------------------------------------------
# Load and label data
# ---------------------------------------------------------------------
print("[INFO] Loading SEA_AD data...")
data = sc.read(adata_path)
data.var_names = data.var["feature_name"].astype(str)

mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t").iloc[:, 0].astype(str)
data = data[:, ~data.var_names.isin(mito)].copy()
sc.pp.filter_genes(data, min_cells=math.ceil(0.02 * data.n_obs))

meta = pd.read_excel(donor_xlsx)
T = meta.loc[meta["Cognitive Status"] == "Dementia", "Donor ID"].astype(str)
C = meta.loc[meta["Cognitive Status"] == "No dementia", "Donor ID"].astype(str)

donors = data.obs["donor_id"].astype(str)
T = pd.Index(T).intersection(donors.unique())
C = pd.Index(C).intersection(donors.unique())

data.obs["cell_type"] = pd.Categorical(
    np.where(donors.isin(T), "Dementia", "No dementia"),
    categories=["Dementia", "No dementia"]
)

print("[INFO] Label counts:")
for k, v in data.obs["cell_type"].value_counts().items():
    print(f"  {k:<12}: {v:>6}")

# ---------------------------------------------------------------------
# Split by donor
# ---------------------------------------------------------------------
print("[INFO] Splitting by donor...")
Te = pd.Index(pd.concat([
    pd.Series(T).sample(n=math.floor(0.5 * len(T))),
    pd.Series(C).sample(n=math.floor(0.5 * len(C)))
]).unique())
All = T.union(C)
TrPool = All.difference(Te)

VaT_n = min(len(T), math.floor(0.1 * len(T)))
VaC_n = min(len(C), math.floor(0.1 * len(C)))
Va = pd.Index(pd.concat([
    pd.Series(list(T)).sample(n=VaT_n),
    pd.Series(list(C)).sample(n=VaC_n)
]).unique())
Tr = TrPool.difference(Va)

donor_ids = data.obs["donor_id"].astype(str)
train_data = data[donor_ids.isin(Tr)].copy()
val_data   = data[donor_ids.isin(Va)].copy()
test_data  = data[donor_ids.isin(Te)].copy()
print(f"[INFO] Split complete: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")

# ---------------------------------------------------------------------
# QC and DE-gene selection
# ---------------------------------------------------------------------
print("[INFO] Running QC and DE selection...")
train_data_qc = qc(train_data.copy())
X = np.asarray(train_data_qc.X, dtype=np.float64)
X[~np.isfinite(X)] = 0.0
# cap = np.percentile(X, 99.9)
# train_data_qc.X = np.clip(X, 0, cap)
train_data_qc.X = X

gene_list = construct_gene_list(train_data_qc,
                                train_data_qc.obs["cell_type"].cat.categories.values,
                                n_top_genes=args.n_top_genes,
                                alpha=0.05)

gene_list = pd.Index(gene_list.astype(str))
# np.save(f"{d}output_comparsion/Spectra/gene_list_{args.cell_type}.npy", np.array(gene_list))
# print(f"[SAVED] gene_list_{args.cell_type}.npy ({len(gene_list)} genes)")

# Save both .npy and .txt versions for compatibility
np.save(f"{d}output_comparsion/Spectra/gene_list_{args.cell_type}.npy", np.array(gene_list))
pd.Series(gene_list).to_csv(f"{d}output_comparsion/Spectra/gene_list_{args.cell_type}.txt",
                            index=False, header=False)
print(f"[SAVED] gene_list_{args.cell_type}.npy and .txt ({len(gene_list)} genes)")

# ---------------------------------------------------------------------
# Save filtered .h5ad
# ---------------------------------------------------------------------
for name, ad in zip(["train", "val", "test"], [train_data, val_data, test_data]):
    subset = ad[:, [g for g in gene_list if g in ad.var_names]].copy()
    subset.write(f"{d}output_comparsion/Spectra/{name}_{args.cell_type}_filtered.h5ad")
    print(f"[SAVED] {name}_{args.cell_type}_filtered.h5ad")

print("[INFO][Spectra Preprocess] Done.")
