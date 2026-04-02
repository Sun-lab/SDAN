# Su_2020_spectra_preprocess.py
"""
Select and save top 2000 genes for CD4 or CD8 using the same Su_2020 structure.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import math

from SDAN.preprocess import qc, construct_gene_list
from SDAN.args import parse_args

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------- Reproducibility ----------------
np.random.seed(888)
torch.manual_seed(888)

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(figsize=(8, 6), dpi=80, facecolor='white')

# ---------------- Parse args ----------------
args = parse_args()

# ---------------- Paths ----------------
d = "./Su_2020/"
cell_type_str = args.cell_type
data_dir = f"{d}gex_{cell_type_str}.mtx.gz"
genes_dir = f"{d}gex_{cell_type_str}_genes.txt"
meta_ind_dir = f"{d}Table_S1.xlsx"
meta_cell_dir = f"{d}cell_info_{cell_type_str}.csv"

OUT_DIR = os.path.join(d, "output_comparison/Spectra/")
os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(os.path.join(d, "figures/"), exist_ok=True)

# ---------------- Load data ----------------
print("[LOAD] Reading data ...", flush=True)
data = sc.read(data_dir)
gene_names = pd.read_csv(genes_dir, header=None)
meta_cell = pd.read_csv(meta_cell_dir)
cell_names = meta_cell["V1"]
meta_ind = pd.read_excel(meta_ind_dir, sheet_name="S1.1 Patient Clinical Data")

data.var["gene_symbols"] = gene_names.values
data.var_names = gene_names.squeeze()
data.obs["barcode"] = cell_names.values
data.obs_names = cell_names.squeeze()

# ---------------- Preprocess (QC) ----------------
adata_qc = data.copy()
if not hasattr(adata_qc.X, "toarray"):  # already dense
    adata_qc.X = np.asarray(adata_qc.X, dtype=np.float64)
qc(adata_qc)
data = adata_qc  # use QC’d data going forward

# ---- confirm that data truly contains the QC-processed, log-normalized expression matrix ----
print("[CHECK] QC transformation verification:")
# show range of values before and after QC
print("Before QC: min =", np.min(data.raw.X) if hasattr(data, "raw") and data.raw is not None else "N/A",
      "max =", np.max(data.raw.X) if hasattr(data, "raw") and data.raw is not None else "N/A")
print("After QC:  min =", np.min(data.X), "max =", np.max(data.X))
# confirm assignment worked
print("data is adata_qc?", data is adata_qc)

# ---------------- Filter mitochondrial genes ----------------
gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")
mito_col = "hgnc_symbol" if "hgnc_symbol" in gene_mito.columns else gene_mito.columns[0]
data = data[:, ~data.var_names.isin(gene_mito[mito_col].astype(str))]
print(f"[INFO] After removing mitochondrial genes: {data.shape[1]} genes")

# ---------------- Keep genes expressed ≥ 2% cells ----------------
data_nonzero_prop = (data.X != 0).sum(axis=0) / data.shape[0]
data = data[:, np.array(data_nonzero_prop > 0.02)]
print(f"[INFO] After filtering by 2% expression: {data.shape[1]} genes remain")

# ---------------- Assign severity labels ----------------
print("\n[STEP] Assigning severity labels ...", flush=True)
meta_ind["Who Ordinal Scale"].replace("1 or 2", 2, inplace=True)
meta_ind_WOS = meta_ind.groupby("Study Subject ID")["Who Ordinal Scale"].max()
mild_ind = meta_ind_WOS[meta_ind_WOS <= 2].index.to_series()
severe_ind = meta_ind_WOS[meta_ind_WOS >= 5].index.to_series()

data.obs["cell_type"] = np.select(
    [(meta_cell["individual"].isin(mild_ind)), (meta_cell["individual"].isin(severe_ind))],
    ["mild", "severe"],
    default="moderate"
)
data.obs["individual"] = meta_cell["individual"].values

print(f"[INFO] Assigned cell_type labels: {data.obs['cell_type'].unique().tolist()}")
print("[INFO] cell_type distribution:")
print(data.obs["cell_type"].value_counts())

# ---------------- Select top 2000 genes ----------------
print("\n[STEP] Selecting top 2000 genes ...", flush=True)
if not pd.api.types.is_categorical_dtype(data.obs["cell_type"]):
    data.obs["cell_type"] = data.obs["cell_type"].astype("category")

cell_type_list = data.obs["cell_type"].cat.categories.values
print(f"[DEBUG] cell_type_list = {cell_type_list}")

n_top = getattr(args, "n_top_genes", 2000)

# ---- Call construct_gene_list ----
# gene_list = construct_gene_list(data, cell_type_list, n_top_genes=n_top, alpha=0.05)
gene_list = construct_gene_list(data.copy(), cell_type_list, n_top_genes=args.n_top_genes, alpha=0.05)
gene_list = pd.Index(gene_list.astype(str))

# ---- Debugging info ----
print(f"[DEBUG] Raw gene_list length before cap: {len(gene_list)}")
if len(gene_list) > n_top:
    print(f"[WARN] More than {n_top} genes ({len(gene_list)}) passed alpha filter. Truncating to {n_top}.")
    gene_list = gene_list[:n_top]

print(f"[INFO] Selected top {len(gene_list)} genes (final)")

# ---------------- Save gene list ----------------
gene_list_path = os.path.join(OUT_DIR, f"gene_list_{cell_type_str}.txt")
pd.Series(gene_list).to_csv(gene_list_path, index=False, header=False)
print(f"[SAVED] {gene_list_path}")

# ---------------- Preview ----------------
print("\n[PREVIEW] Top 10 genes:")
print(gene_list[:10].tolist())
