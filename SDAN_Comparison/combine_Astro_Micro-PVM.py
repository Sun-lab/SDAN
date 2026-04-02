"""
combine_Astro_Micro-PVM.py
Combine SEA_AD Astro and Micro-PVM datasets into one AnnData.
Ensures consistent gene naming (feature_name) and metadata alignment
for downstream Spectra / GNN / sciRED training.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

# ---------------- 1. Paths ----------------
BASE = Path("./SEA_AD")
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "output_comparison" / "Spectra"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASTRO_PATH = DATA_DIR / "Astro.h5ad"
MICRO_PATH = DATA_DIR / "Micro-PVM.h5ad"
OUT_PATH = OUT_DIR / "combined_Astro_Micro-PVM.h5ad"

np.random.seed(888)

print("[INFO] Input files:")
print(f"  ASTRO : {ASTRO_PATH}")
print(f"  MICRO : {MICRO_PATH}")

# ---------------- 2. Load datasets ----------------
astro = sc.read(ASTRO_PATH)
micro = sc.read(MICRO_PATH)
print(f"[INFO] astro shape : {astro.shape}")
print(f"[INFO] micro shape : {micro.shape}")

# ---------------- 3. Standardize gene names ----------------
def standardize_gene_names(adata, label):
    """Convert Ensembl IDs to gene symbols, uppercase, remove version suffix."""
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str)
    adata.var_names = adata.var_names.str.replace(r"\.\d+$", "", regex=True).str.upper()
    adata.var_names_make_unique()
    print(f"[INFO] {label}: standardized {adata.n_vars} gene names")
    return adata

astro = standardize_gene_names(astro, "Astro")
micro = standardize_gene_names(micro, "Micro-PVM")

# ---------------- 4. Verify gene consistency ----------------
astro_genes = pd.Index(astro.var_names)
micro_genes = pd.Index(micro.var_names)

same_length = len(astro_genes) == len(micro_genes)
same_set = set(astro_genes) == set(micro_genes)
same_order = astro_genes.equals(micro_genes)

print("\n[CHECK] Cross-dataset comparison:")
print(f"  Same number of genes? {same_length}")
print(f"  Same set (ignoring order)? {same_set}")
print(f"  Same order (exact)? {same_order}")

if not same_order:
    print("[WARN] Gene lists differ between Astro and Micro-PVM — aligning by intersection.")
    common_genes = astro_genes.intersection(micro_genes)
    print(f"[INFO] Common genes retained: {len(common_genes)}")
    astro = astro[:, common_genes].copy()
    micro = micro[:, common_genes].copy()
else:
    print("[INFO] Gene names match perfectly — using all genes.")
    common_genes = astro_genes

# ---------------- 5. Add metadata ----------------
astro.obs["cell_label"] = "Astro"
micro.obs["cell_label"] = "Micro-PVM"

astro.obs_names = [f"astro_{i}" for i in range(astro.n_obs)]
micro.obs_names = [f"micro_{i}" for i in range(micro.n_obs)]

# ---------------- 6. Combine datasets ----------------
print("\n[COMBINE] Concatenating Astro and Micro-PVM...")
combined = astro.concatenate(
    micro,
    batch_key="batch",
    batch_categories=["Astro", "Micro-PVM"],
    uns_merge="unique"
)
print(f"[INFO] Combined shape: {combined.shape}")

# ---------------- 7. Finalize metadata ----------------
combined.obs["cell_label"] = combined.obs["batch"].astype(str)
combined.obs["cell_type"] = combined.obs["cell_label"]

if "donor_id" in combined.obs.columns:
    print(f"[INFO] Unique donors: {len(combined.obs['donor_id'].unique())}")

print("\n[INFO] Cell type distribution:")
print(combined.obs["cell_type"].value_counts())

# ---------------- 8. Save ----------------
print(f"\n[SAVE] Writing combined AnnData to: {OUT_PATH}")
combined.write(OUT_PATH, compression="gzip")
print("[INFO] Done — combined Astro + Micro-PVM dataset saved successfully.")
