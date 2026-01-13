# Su_2020_spectra_evaluation.py
"""
Final workflow:
- Split 50/50 by individual.
- Load pre-trained Spectra union gene list and loadings from output_v2_union/
- Align/pad genes in CD4 or CD8 dataset to the pretrained union gene list
- Project train/test sets into Spectra latent space
- Evaluate classifiers (LR/RF)
"""

# ---------------- Threading locks ----------------
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------- Imports ----------------
import os, math, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
import torch

# ---------------- SDAN imports ----------------
from SDAN.args import parse_args
from SDAN.preprocess import qc
from SDAN.evaluation import eval_and_save as _eval_and_save
from SDAN.evaluation import set_eval_labels

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(888)
torch.manual_seed(888)

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(figsize=(8, 6), dpi=80, facecolor="white")

# ---------------- Arguments ----------------
args = parse_args()
print(f"[DEBUG] args={args}", flush=True)

# ---------------- Paths ----------------
BASE = "./Su_2020/"
OUT_DIR = os.path.join(BASE, "output_v2_4k/")
PRETRAIN_DIR = os.path.join(BASE, "output_v2_4k/")
os.makedirs(OUT_DIR, exist_ok=True)

cell_type_str = args.cell_type
data_dir = f"{BASE}gex_{cell_type_str}.mtx.gz"
genes_dir = f"{BASE}gex_{cell_type_str}_genes.txt"
meta_ind_dir = f"{BASE}Table_S1.xlsx"
meta_cell_dir = f"{BASE}cell_info_{cell_type_str}.csv"

# ---------------- Load data ----------------
print("[LOAD] Reading data ...")
data = sc.read(data_dir)
gene_names = pd.read_csv(genes_dir, header=None).iloc[:, 0].astype(str).to_numpy()
if len(gene_names) != data.n_vars:
    raise ValueError(f"genes length {len(gene_names)} != n_vars {data.n_vars}")
data.var["gene_symbols"] = gene_names
data.var_names = pd.Index(gene_names)

meta_cell = pd.read_csv(meta_cell_dir)
cell_names = meta_cell["V1"].astype(str).to_numpy()
if len(cell_names) != data.n_obs:
    raise ValueError(f"cells length {len(cell_names)} != n_obs {data.n_obs}")
data.obs["barcode"] = cell_names
data.obs_names = pd.Index(cell_names)
meta_ind = pd.read_excel(meta_ind_dir, sheet_name="S1.1 Patient Clinical Data")

# ---------------- Label assignment ----------------
wos = meta_ind["Who Ordinal Scale"].astype(str).str.replace("1 or 2", "2", regex=False)
wos = pd.to_numeric(wos, errors="coerce")
meta_ind = meta_ind.assign(WOS=wos)

meta_ind_WOS = meta_ind.groupby("Study Subject ID")["WOS"].max().dropna()
mild_ind = meta_ind_WOS[meta_ind_WOS <= 2].index.to_series()
severe_ind = meta_ind_WOS[meta_ind_WOS >= 5].index.to_series()

data.obs["cell_type"] = np.select(
    [(meta_cell["individual"].isin(mild_ind)), (meta_cell["individual"].isin(severe_ind))],
    ["mild", "severe"],
    default="moderate",
)
data.obs["individual"] = meta_cell["individual"].values

# ---------------- Split train/test ----------------
test_ind = pd.concat([
    mild_ind.sample(n=math.floor(0.5 * len(mild_ind))),
    severe_ind.sample(n=math.floor(0.5 * len(severe_ind))),
])
train_ind = pd.concat([mild_ind, severe_ind]).drop(test_ind.index)

train_cell_id = meta_cell[meta_cell["individual"].isin(train_ind)]["V1"]
test_cell_id = meta_cell[meta_cell["individual"].isin(test_ind)]["V1"]
val_cell_id = train_cell_id.sample(n=math.floor(0.1 * len(train_cell_id)))
train_cell_id = train_cell_id.drop(val_cell_id.index)

train_data = data[train_cell_id].copy()
test_data = data[test_cell_id].copy()
val_data = data[val_cell_id].copy()

# ---------------- Load pretrained Spectra gene list ----------------
gene_list_path = os.path.join(PRETRAIN_DIR, "gene_list_cd4_cd8_union.txt")
pretrain_path = os.path.join(PRETRAIN_DIR, "train_s_cd4_cd8_BL_Spectra_union.npy")

if not os.path.exists(gene_list_path):
    raise FileNotFoundError(f"[ERROR] Pretrained gene list not found: {gene_list_path}")
if not os.path.exists(pretrain_path):
    raise FileNotFoundError(f"[ERROR] Pretrained Spectra loadings not found: {pretrain_path}")

gene_list = pd.read_csv(gene_list_path, header=None).iloc[:, 0].astype(str)
gene_list = pd.Index(gene_list)
print(f"[LOAD] Using pretrained Spectra gene list ({len(gene_list)} genes)")

# ---------------- Pad dataset to pretrained genes ----------------
def _subset_or_pad_in_order(adata, genes):
    """Subset existing genes, and add all-zero columns for missing ones."""
    data_genes = pd.Index(adata.var_names.astype(str))
    shared_genes = [g for g in genes if g in data_genes]
    missing_genes = [g for g in genes if g not in data_genes]

    adata_sub = adata[:, shared_genes].copy()
    X_sub = adata_sub.X
    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()
    X_sub = np.asarray(X_sub, dtype=np.float64)
    if X_sub.ndim == 1:
        X_sub = X_sub.reshape(-1, 1)

    if len(missing_genes) > 0:
        print(f"[WARN] {len(missing_genes)} genes missing; adding zero columns.")
        zeros = np.zeros((adata_sub.n_obs, len(missing_genes)), dtype=np.float64)
        X_full = np.concatenate([X_sub, zeros], axis=1)
        adata_sub = AnnData(X=X_full, obs=adata_sub.obs.copy())
        adata_sub.var_names = pd.Index(shared_genes + missing_genes)
    else:
        print("[INFO] All pretrained genes found in dataset.")
        adata_sub.X = X_sub

    adata_sub = adata_sub[:, genes].copy()
    print(f"[STATS] Shared genes: {len(shared_genes)} / {len(genes)} ({len(shared_genes)/len(genes):.2%})")
    return adata_sub

train_data = _subset_or_pad_in_order(train_data, gene_list)
test_data = _subset_or_pad_in_order(test_data, gene_list)
print(f"[DEBUG INFO] shapes after freeze: train={train_data.shape}, test={test_data.shape}")

# ---------------- Spectra Projection ----------------
def run_spectra():
    print("[Spectra] Using precomputed union Spectra model.", flush=True)

    for _ad in (train_data, test_data):
        _ad.var["mt"] = _ad.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(_ad, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        sc.pp.normalize_total(_ad, target_sum=1e4)
        sc.pp.log1p(_ad)
        if sp.issparse(_ad.X):
            _ad.X = _ad.X.toarray()

    Xtr = np.asarray(train_data.X, dtype=np.float64)
    Xte = np.asarray(test_data.X, dtype=np.float64)
    print(f"[Spectra] Data shapes: train={Xtr.shape}, test={Xte.shape}")

    S = np.load(pretrain_path)
    print(f"[Spectra] Loaded precomputed train_s: {S.shape}")

    if S.shape[0] != Xtr.shape[1]:
        raise ValueError(f"[ERROR] train_s gene dimension mismatch: S={S.shape[0]} vs data={Xtr.shape[1]}")

    # Projection (genes × components) @ (cells × genes) = cells × components
    Ztr_full = Xtr @ S
    Zte_full = Xte @ S
    print(f"[Spectra] Projected shapes: train={Ztr_full.shape}, test={Zte_full.shape}")

    # Save reduced data
    prog_names = [f"Spectra_{i}" for i in range(S.shape[1])]
    tag = f"{args.cell_type}_Spectra_union"

    train_reduced_adata = AnnData(X=Ztr_full)
    train_reduced_adata.obs = train_data.obs.copy()
    train_reduced_adata.var = pd.DataFrame(index=prog_names)
    train_reduced_adata.write(os.path.join(OUT_DIR, f"train_reduced_{tag}.h5ad"))

    test_reduced_adata = AnnData(X=Zte_full)
    test_reduced_adata.obs = test_data.obs.copy()
    test_reduced_adata.var = pd.DataFrame(index=prog_names)
    test_reduced_adata.write(os.path.join(OUT_DIR, f"test_reduced_{tag}.h5ad"))

    # Evaluation
    set_eval_labels(pos_label="severe", neg_label="mild")
    cell_type_list = sorted(pd.unique(train_data.obs["cell_type"]))
    _eval_and_save(
        train_reduced_adata,
        test_reduced_adata,
        "Spectra_union",
        os.path.join(OUT_DIR, f"results_classifiers_{args.cell_type}_Spectra_union.csv"),
        os.path.join(OUT_DIR, f"logreg_feature_importance_{args.cell_type}_Spectra_union.csv"),
        cell_type_list,
        mild_ind=mild_ind,
        severe_ind=severe_ind,
    )

    print(f"[DEBUG][Spectra] done ({tag})", flush=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    print(f"[info] Backend selected: {args.backend}")
    if args.backend == "Spectra":
        run_spectra()
    else:
        raise ValueError("Unsupported backend. Use 'Spectra' for pretrained projection.")
