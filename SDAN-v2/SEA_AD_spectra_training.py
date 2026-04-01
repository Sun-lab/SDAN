"""
SEA_AD_spectra_training.py
Train Spectra using SEA_AD Dementia vs No Dementia filtered datasets (Astro + Micro-PVM union).
Dataset: SEA-AD (Seattle Alzheimer's Disease) single-cell RNA-seq data.
"""

# ---------- Setup ----------
import os, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from SDAN.preprocess import construct_gene_graph
from SDAN.args import parse_args

# Thread control for reproducibility
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false"
})
warnings.simplefilter("ignore", FutureWarning)
np.random.seed(888)
torch.manual_seed(888)

try:
    import Spectra
except ImportError:
    Spectra = None


# ---------- Args & Paths ----------
args = parse_args()
BASE = "./SEA_AD/"
OUT_DIR = os.path.join(BASE, "output_comparsion/Spectra/")
os.makedirs(OUT_DIR, exist_ok=True)

cell_type_str = args.cell_type
combined_path = os.path.join(OUT_DIR, "combined_Astro_Micro-PVM.h5ad")

# ---------- Union Gene List ----------
Astro = pd.read_csv(f"{OUT_DIR}gene_list_Astro.txt", header=None)[0].astype(str)
Micro_PVM = pd.read_csv(f"{OUT_DIR}gene_list_Micro-PVM.txt", header=None)[0].astype(str)
union_genes = pd.Index(Astro).union(Micro_PVM)

# Clean format but only save the final union list
union_genes = union_genes.str.replace(r"\.\d+$", "", regex=True).str.upper()
pd.Series(union_genes).to_csv(
    f"{OUT_DIR}gene_list_Astro_Micro-PVM_union.txt",
    index=False, header=False
)

print(f"[INFO] Astro list size: {len(Astro)}")
print(f"[INFO] Micro-PVM list size: {len(Micro_PVM)}")
print(f"[INFO] Union size: {len(union_genes)}")

# ---------- Load Combined Data ----------
adata = sc.read_h5ad(combined_path)
print(f"[INFO] Combined dataset:")
print(f"       File: {combined_path}")
print(f"       Shape: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"       cell_types = {adata.obs['cell_type'].unique().tolist()}")

# ---------- Filter once only (no feature_name alignment, no overlap check) ----------
adata = adata[:, adata.var_names.isin(union_genes)].copy()
adata = adata[:, ~adata.var_names.duplicated()].copy()

print(f"[INFO] Filtered dataset: {adata.n_obs} cells × {adata.n_vars} genes")

# ---------- Save ----------
filtered_path = f"{OUT_DIR}combined_Astro_Micro-PVM_union_filtered.h5ad"
adata.write(filtered_path)
print(f"[SAVED] Filtered AnnData → {filtered_path}")

# ---------- Reload & Check ----------
adata = sc.read_h5ad(filtered_path)
print(f"[INFO] Reloaded dataset for training:")
print(f"       File: {filtered_path}")
print(f"       Shape: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"       cell_types = {adata.obs['cell_type'].unique().tolist()}")

# ---------- Spectra Training ----------
def run_spectra():
    if Spectra is None:
        raise ImportError("Spectra not installed.")

    print("\n[Spectra] Training Spectra model (Astro + Micro-PVM union)...", flush=True)

    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X = np.asarray(adata.X, dtype=np.float64)
    labels = adata.obs["cell_type"].astype(str).values
    print(f"[INFO] Found cell types: {np.unique(labels)}")

    # ---------- Build adjacency ----------
    print("[Spectra] Building adjacency ...", flush=True)
    edge_index = construct_gene_graph(adata.var_names)

    sp_adj = to_scipy_sparse_matrix(edge_index, num_nodes=X.shape[1])
    sp_adj = ((sp_adj + sp_adj.T) > 0).astype(np.float64)
    sp_adj.setdiag(0)
    sp_adj.eliminate_zeros()

    A = sp_adj.toarray().astype(np.float64)
    print(f"[Spectra] Adjacency shape: {A.shape}, nnz={sp_adj.nnz}")

    # ---------- Configure L ----------
    L_total = int(args.spectra_L)
    L = {
        "global": L_total // 2,
        "Astro": L_total // 4,
        "Micro-PVM": L_total // 4
    }
    adj_dict = {"global": A, "Astro": A, "Micro-PVM": A}

    # ---------- Train Spectra ----------
    model = Spectra.SPECTRA_Model(
        X=X,
        labels=labels,
        L=L,
        adj_matrix=adj_dict,
        gs_dict=None,
        lam=args.spectra_lam,
        delta=args.spectra_delta,
        rho=args.spectra_rho,
        kappa=None,
        use_cell_types=True,
        vocab=list(adata.var_names.astype(str)),
    )

    print("[Spectra] Training ...", flush=True)

    np.seterr(divide="raise", over="raise", invalid="raise", under="ignore")
    model.train(X=X, labels=labels, num_epochs=int(args.spectra_epochs))

    # ---------- Save outputs ----------
    W = model.return_factors().astype(np.float64)
    S = W.T

    print(f"[Spectra] Finished training; train_s shape: {S.shape}")

    tag = f"{cell_type_str}_union"

    np.save(os.path.join(OUT_DIR, f"train_s_{tag}.npy"), S)
    pd.DataFrame(
        S,
        index=adata.var_names,
        columns=[f"Spectra_{i}" for i in range(S.shape[1])]
    ).to_csv(os.path.join(OUT_DIR, f"train_s_{tag}.csv"))

    print(f"[SAVED] train_s_{tag}.npy and train_s_{tag}.csv")


# ---------- Main ----------
if __name__ == "__main__":
    if args.backend.lower() == "spectra":
        run_spectra()
    else:
        raise ValueError("Only Spectra backend is supported.")
