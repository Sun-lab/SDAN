# combine_cd4_cd8.py
"""
The script loads CD4 and CD8 matrices, verifies identical gene lists, labels each dataset, 
concatenates them cell-wise, and saves a single AnnData file (combined_cd4_cd8.h5ad) 
containing both cell types with consistent gene alignment and metadata.

"""

import pandas as pd
import scanpy as sc
from pathlib import Path

# 0. Flexible base path — auto-detect regardless of where you run the script
# This resolves to the folder containing this Python file
BASE = Path(__file__).resolve().parent

# Go up to the dataset root if script is inside Su_2020
if BASE.name != "Su_2020":
    BASE = BASE / "Su_2020"

OUT_DIR = BASE / "output_comparsion" / "Spectra"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "combined_cd4_cd8.h5ad"

# File paths relative to BASE
CD4_MTX = BASE / "gex_cd4_BL.mtx.gz"
CD8_MTX = BASE / "gex_cd8_BL.mtx.gz"
CD4_TXT = BASE / "gex_cd4_BL_genes.txt"
CD8_TXT = BASE / "gex_cd8_BL_genes.txt"


# 1. Load expression matrices (for info only)
print("[LOAD] Reading .mtx.gz matrices...")
cd4 = sc.read_mtx(CD4_MTX)
cd8 = sc.read_mtx(CD8_MTX)
print(f"[INFO] cd4 shape: {cd4.shape}, cd8 shape: {cd8.shape}")

# 2. Helper function to read gene list safely
def read_gene_list(path):
    with open(path, "r") as f:
        lines = f.readlines()
    genes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", "\t").split("\t")
        genes.append(parts[0].strip())
    return pd.Series(genes, dtype=str)

# 3. Load .txt gene lists
genes_cd4 = read_gene_list(CD4_TXT)
genes_cd8 = read_gene_list(CD8_TXT)

print(f"[INFO] Loaded gene files:")
print(f"  CD4: {len(genes_cd4)} genes")
print(f"  CD8: {len(genes_cd8)} genes")

# 4. Compare CD4 vs CD8 gene lists
same_length = len(genes_cd4) == len(genes_cd8)
same_set = set(genes_cd4) == set(genes_cd8)
same_order = genes_cd4.equals(genes_cd8)

print("\n[CHECK] Cross-dataset comparison:")
print(f"  Same number of genes? {same_length}")
print(f"  Same gene set (ignoring order)? {same_set}")
print(f"  Same order (exact position match)? {same_order}")

print("\n[Preview] First 10 genes (CD4 | CD8):")
for i in range(30):
    print(f"{i+1:>3d}. {genes_cd4[i]}  |  {genes_cd8[i]}")

if not same_order:
    print("\n[WARN] Differences found between CD4 and CD8 gene lists:")
    diff_cd4 = set(genes_cd4) - set(genes_cd8)
    diff_cd8 = set(genes_cd8) - set(genes_cd4)
    print(f"  Unique to CD4: {len(diff_cd4)} | Unique to CD8: {len(diff_cd8)}")

    mismatches = (genes_cd4 != genes_cd8).sum()
    print(f"  Number of mismatched positions: {mismatches}")

    if mismatches > 0:
        print("  Example mismatches:")
        for i in range(min(10, len(genes_cd4))):
            if genes_cd4[i] != genes_cd8[i]:
                print(f"    idx {i}: cd4={genes_cd4[i]}, cd8={genes_cd8[i]}")
else:
    print("\n[SUCCESS] CD4 and CD8 have identical gene lists (same names and order).")


cd4.var_names = genes_cd4
cd8.var_names = genes_cd8
print("\n[INFO] Assigned verified gene names to cd4 and cd8 AnnData objects.")

cd4.obs["cell_label"] = "cd4"
cd8.obs["cell_label"] = "cd8"

# give unique cell IDs
cd4.obs_names = [f"cd4_{i}" for i in range(cd4.n_obs)]
cd8.obs_names = [f"cd8_{i}" for i in range(cd8.n_obs)]

# 5. Combine datasets
print("\n[COMBINE] Concatenating CD4 and CD8 datasets...")
combined = cd4.concatenate(
    cd8,
    batch_key="batch",
    batch_categories=["cd4", "cd8"],
    uns_merge="unique"
)
print(f"[INFO] Combined shape: {combined.shape}")

# # Double-check label column
# combined.obs["cell_label"] = combined.obs["batch"].astype(str)

# # 6. Save combined dataset
# print(f"[SAVE] Writing combined AnnData to: {OUT_PATH}")
# combined.write(OUT_PATH)
# Double-check label column

combined.obs["cell_label"] = combined.obs["batch"].astype(str)
# Add this line — ensures Spectra recognizes the CD4/CD8 labels
combined.obs["cell_type"] = combined.obs["cell_label"]
# 6. Save combined dataset
print(f"[SAVE] Writing combined AnnData to: {OUT_PATH}")
combined.write(OUT_PATH)
