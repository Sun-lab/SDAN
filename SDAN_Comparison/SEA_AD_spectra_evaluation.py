"""
SEA_AD_spectra_evaluation.py
Evaluate pretrained Spectra union model on SEA_AD data (Dementia vs No Dementia).

Steps:
 - Load SEA_AD .h5ad dataset for given cell type
 - Split donors 50/50 into train/test
 - Load pretrained union Spectra gene list and loadings
 - Align/pad genes to match pretrained union
 - Project train/test sets into Spectra latent space
 - Evaluate classifiers (LR / RF)
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
import torch
from anndata import AnnData

from SDAN.args import parse_args
from SDAN.preprocess import qc
from SDAN.evaluation import eval_and_save as _eval_and_save
from SDAN.evaluation import set_eval_labels

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(888)
torch.manual_seed(888)
sc.settings.verbosity = 2
sc.settings.set_figure_params(figsize=(8, 6), dpi=80, facecolor="white")

# ------------------------- Helpers -------------------------
def ensure_individual_col(ad):
    """
    Ensure that ad.obs contains an 'individual' column.
    Tries to map from donor_id / Donor ID / participant, etc.
    Falls back to obs_names if nothing found.
    """
    if "individual" not in ad.obs.columns:
        candidates = ["donor_id", "Donor ID", "participant", "subject_id", "sample_id"]
        for c in candidates:
            if c in ad.obs.columns:
                ad.obs["individual"] = ad.obs[c].astype(str)
                return
        ad.obs["individual"] = ad.obs_names.astype(str)


# ---------------- Args & Paths ----------------
args = parse_args()
print(f"[INFO] args={args}", flush=True)

BASE = "./SEA_AD/"
OUT_DIR = os.path.join(BASE, "output_comparison/Spectra/")
PRETRAIN_DIR = OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

cell_type_str = args.cell_type
adata_path = f"{BASE}data/{cell_type_str}.h5ad"
meta_path = f"{BASE}data/sea-ad_cohort_donor_metadata_020624.xlsx"

# ---------------- Load data ----------------
print("[LOAD] Reading SEA_AD data ...")
data = sc.read(adata_path)
data.var_names = data.var["feature_name"].astype(str)

mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t").iloc[:, 0].astype(str)
data = data[:, ~data.var_names.isin(mito)].copy()
sc.pp.filter_genes(data, min_cells=math.ceil(0.02 * data.n_obs))

# ---------------- Label assignment ----------------
meta = pd.read_excel(meta_path)
T = meta.loc[meta["Cognitive Status"] == "Dementia", "Donor ID"].astype(str)
C = meta.loc[meta["Cognitive Status"] == "No dementia", "Donor ID"].astype(str)
donors = data.obs["donor_id"].astype(str)

T = pd.Index(T).intersection(donors.unique())
C = pd.Index(C).intersection(donors.unique())

data.obs["cell_type"] = np.where(
    donors.isin(T), "Dementia", "No dementia"
)
data.obs["individual"] = donors

print("[INFO] Label counts:")
for k, v in data.obs["cell_type"].value_counts().items():
    print(f"  {k:<12}: {v:>6}")

# ---------------- Split by donor ----------------
print("[INFO] Splitting by donor ...")
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

donor_ids = donors
train_data = data[donor_ids.isin(Tr)].copy()
test_data = data[donor_ids.isin(Te)].copy()
print(f"[INFO] Split complete: train={train_data.shape}, test={test_data.shape}")

# ---------------- Load pretrained Spectra ----------------
#gene_list_path = os.path.join(PRETRAIN_DIR, "gene_list_Astro_Micro-PVM_union.txt")
gene_list_path = os.path.join(PRETRAIN_DIR, "gene_list_Astro_Micro-PVM_union_filtered.txt")
pretrain_path  = os.path.join(PRETRAIN_DIR, "train_s_Astro_Micro-PVM_union.npy")

if not os.path.exists(gene_list_path):
    raise FileNotFoundError(f"[ERROR] Gene list not found: {gene_list_path}")
if not os.path.exists(pretrain_path):
    raise FileNotFoundError(f"[ERROR] Pretrained Spectra loadings not found: {pretrain_path}")

gene_list = pd.read_csv(gene_list_path, header=None).iloc[:, 0].astype(str)
gene_list = pd.Index(gene_list)
S = np.load(pretrain_path)
print(f"[INFO] Loaded pretrained Spectra loadings: {S.shape}")

# ---------------- Subset or pad genes ----------------
def _subset_or_pad_in_order(adata, genes):
    """Subset existing genes, pad zeros for missing ones."""
    data_genes = pd.Index(adata.var_names.astype(str))
    shared = [g for g in genes if g in data_genes]
    missing = [g for g in genes if g not in data_genes]
    adata_sub = adata[:, shared].copy()
    X = np.asarray(adata_sub.X.toarray() if sp.issparse(adata_sub.X) else adata_sub.X, dtype=np.float64)
    if missing:
        zeros = np.zeros((adata_sub.n_obs, len(missing)), dtype=np.float64)
        X = np.concatenate([X, zeros], axis=1)
        adata_sub = AnnData(X=X, obs=adata_sub.obs.copy())
        adata_sub.var_names = pd.Index(shared + missing)
        print(f"[WARN] {len(missing)} missing genes → padded zeros.")
    adata_sub = adata_sub[:, genes].copy()
    print(f"[INFO] Gene overlap: {len(shared)}/{len(genes)} ({len(shared)/len(genes):.2%})")
    return adata_sub

train_data = _subset_or_pad_in_order(train_data, gene_list)
test_data  = _subset_or_pad_in_order(test_data, gene_list)

# ---------------- Spectra Projection ----------------
def run_spectra():
    print("[Spectra] Projecting SEA_AD data using pretrained Spectra union model ...")

    for ad in (train_data, test_data):
        ad.var["mt"] = ad.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        if sp.issparse(ad.X):
            ad.X = ad.X.toarray()

    Xtr = np.asarray(train_data.X, dtype=np.float64)
    Xte = np.asarray(test_data.X, dtype=np.float64)
    print(f"[Spectra] Shapes: train={Xtr.shape}, test={Xte.shape}")

    if S.shape[0] != Xtr.shape[1]:
        raise ValueError(f"[ERROR] train_s gene dimension mismatch: S={S.shape[0]} vs data={Xtr.shape[1]}")

    # Projection: (cells × genes) @ (genes × components)
    Ztr_full = Xtr @ S
    Zte_full = Xte @ S
    print(f"[Spectra] Projected shapes: train={Ztr_full.shape}, test={Zte_full.shape}")

    prog_names = [f"Spectra_{i}" for i in range(S.shape[1])]
    tag = f"{args.cell_type}_Spectra_union"

    train_reduced_adata = AnnData(X=Ztr_full, obs=train_data.obs.copy(), var=pd.DataFrame(index=prog_names))
    test_reduced_adata  = AnnData(X=Zte_full,  obs=test_data.obs.copy(),  var=pd.DataFrame(index=prog_names))

    train_reduced_adata.write(os.path.join(OUT_DIR, f"train_reduced_{tag}.h5ad"))
    test_reduced_adata.write(os.path.join(OUT_DIR, f"test_reduced_{tag}.h5ad"))

    ensure_individual_col(train_reduced_adata)
    ensure_individual_col(test_reduced_adata)

    # ---------------- Evaluate ----------------
    set_eval_labels(pos_label="Dementia", neg_label="No dementia")
    cell_type_list = sorted(pd.unique(train_data.obs["cell_type"]))
    _eval_and_save(
        train_reduced_adata,
        test_reduced_adata,
        "Spectra_union",
        os.path.join(OUT_DIR, f"results_classifiers_{args.cell_type}_Spectra_union.csv"),
        os.path.join(OUT_DIR, f"logreg_feature_importance_{args.cell_type}_Spectra_union.csv"),
        cell_type_list,
        mild_ind=C,
        severe_ind=T,
    )

    print(f"[INFO][Spectra Evaluation] done ({tag})", flush=True)

# ---------------- Main ----------------
if __name__ == "__main__":
    if args.backend.lower() == "spectra":
        run_spectra()
    else:
        raise ValueError("Only Spectra backend supported in this script.")
