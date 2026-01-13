"""
Su_2020_v2.py — unified version for GNN, and sciRED backends.
Compatible with args.py and preprocess.py.
"""

# ------------------------- Imports & setup -------------------------
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import os, math, warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch_geometric.utils import to_scipy_sparse_matrix

from SDAN.model import pipeline
from SDAN.preprocess import qc, construct_gene_graph, construct_gene_list
from SDAN.args import parse_args
from SDAN.evaluation import set_eval_labels, eval_and_save as _eval_and_save

try:
    import Spectra
except Exception:
    Spectra = None

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(888)
torch.manual_seed(888)
# sc.settings.verbosity = 3
sc.settings.verbosity = 1

try:
    sc.logging.print_header()
except Exception as e:
    print(f"[WARN] Skipped Scanpy header due to: {type(e).__name__}: {e}")

sc.settings.set_figure_params(figsize=(8, 6), dpi=80, facecolor="white")

# ------------------------- Paths & args -------------------------
args = parse_args()
d = "./Su_2020/"
cell_type_str = args.cell_type
data_dir = f"{d}gex_{cell_type_str}.mtx.gz"
genes_dir = f"{d}gex_{cell_type_str}_genes.txt"
meta_ind_dir = f"{d}Table_S1.xlsx"
meta_cell_dir = f"{d}cell_info_{cell_type_str}.csv"
os.makedirs(f"{d}output_v2/", exist_ok=True)


# ------------------------- Load and label -------------------------
def load_and_label_data():
    """Load Su_2020 data and assign mild/severe labels."""
    print("[INFO] Loading raw data...")
    data = sc.read(data_dir, cache=True)
    gene_names = pd.read_csv(genes_dir, header=None).iloc[:, 0].astype(str).to_numpy()
    if len(gene_names) != data.n_vars:
        raise ValueError(f"[ERROR] genes length {len(gene_names)} != n_vars {data.n_vars}")
    meta_cell = pd.read_csv(meta_cell_dir)
    meta_ind = pd.read_excel(meta_ind_dir, sheet_name="S1.1 Patient Clinical Data")

    data.var["gene_symbols"] = gene_names
    data.var_names = pd.Index(gene_names)
    data.obs["barcode"] = meta_cell["V1"].astype(str).to_numpy()
    data.obs_names = pd.Index(data.obs["barcode"])

    gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")
    mito_col = "hgnc_symbol" if "hgnc_symbol" in gene_mito.columns else gene_mito.columns[0]
    data = data[:, ~data.var_names.isin(gene_mito[mito_col])]

    data_nonzero_prop = (data.X != 0).sum(axis=0) / data.shape[0]
    data = data[:, data_nonzero_prop > 0.02]

    wos = meta_ind['Who Ordinal Scale'].astype(str).str.replace('1 or 2', '2', regex=False)
    wos = pd.to_numeric(wos, errors='coerce')
    meta_ind = meta_ind.assign(WOS=wos)
    meta_ind_WOS = meta_ind.groupby('Study Subject ID')['WOS'].max().dropna()
    mild_ind = meta_ind_WOS[meta_ind_WOS <= 2].index.to_series()
    severe_ind = meta_ind_WOS[meta_ind_WOS >= 5].index.to_series()

    data.obs["cell_type"] = np.select(
        [(meta_cell["individual"].isin(mild_ind)),
         (meta_cell["individual"].isin(severe_ind))],
        ["mild", "severe"],
        default="moderate",
    )
    data.obs["individual"] = meta_cell["individual"].values
    print("[INFO] Label counts:")
    for k, v in data.obs["cell_type"].value_counts().items():
        print(f"  {k:<9}: {v:>6}")
    return data, meta_cell, mild_ind, severe_ind


# ------------------------- Split -------------------------
def split_by_individual(data, meta_cell, mild_ind, severe_ind):
    print("[INFO] Splitting by individual...")
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
    val_data = data[val_cell_id].copy()
    test_data = data[test_cell_id].copy()
    return train_data, val_data, test_data


# ------------------------- DE-gene selection -------------------------
def get_de_gene_list(train_data, cell_type_list, tag):
    out_path = f"{d}output_v2/gene_list_{tag}.npy"
    print("[INFO] Constructing DE gene list...")
    X = train_data.X.toarray() if sp.issparse(train_data.X) else np.asarray(train_data.X, dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    cap = np.percentile(X, 99.9)
    X = np.clip(X, 0, cap)
    train_data.X = X
    gene_list = construct_gene_list(train_data.copy(), cell_type_list, n_top_genes=args.n_top_genes, alpha=0.05)
    gene_list = pd.Index(gene_list.astype(str))
    np.save(out_path, np.array(gene_list))
    print(f"[SAVED] {out_path}")
    return gene_list


# ------------------------- Run GNN -------------------------
def run_gnn():
    print("[INFO] Running GNN backend ...")
    data, meta_cell, mild_ind, severe_ind = load_and_label_data()
    qc(data)
    train_data, val_data, test_data = split_by_individual(data, meta_cell, mild_ind, severe_ind)
    gene_list = get_de_gene_list(train_data, ["mild", "severe"], args.cell_type)
    print(f"[INFO] The number of DE genes: {len(gene_list)}")

    for ad in [train_data, val_data, test_data]:
        ad._inplace_subset_var([g for g in gene_list if g in ad.var_names])
        ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")

    args.mc_weight = args.graph_weight
    args.o_weight = args.graph_weight
    tag = f"{args.cell_type}_{args.graph_weight:.1f}"
    out_dir = f"{d}output_v2/"

    print("[INFO] Launching SDAN GNN pipeline...")
    for ad in [train_data, val_data, test_data]:
        if sp.issparse(ad.X):
            ad.X = ad.X.toarray().astype(np.float32)
        else:
            ad.X = ad.X.astype(np.float32)
    torch.set_default_dtype(torch.float32)

    (train_GNN, val_GNN, test_GNN), _, cell_type_list, gene_list = pipeline(
        [train_data, val_data, test_data], args, d, tag)

    train_s = torch.tensor(np.load(f"{d}output/train_s_{tag}.npy"))
    np.save(f"{out_dir}train_s_{args.cell_type}_GNN.npy", train_s.detach().cpu().numpy())
    # print(f"[INFO] train_s shape: {train_s.shape}")

    print("[INFO] Projecting cell embeddings using train_s...")
    Ztr = train_GNN.x.t().detach().cpu().numpy() @ train_s.detach().cpu().numpy()
    Zte = test_GNN.x.t().detach().cpu().numpy() @ train_s.detach().cpu().numpy()

    train_reduced_adata = AnnData(X=Ztr, obs=train_data.obs.copy())
    test_reduced_adata = AnnData(X=Zte, obs=test_data.obs.copy())
    train_reduced_adata.write(f"{out_dir}train_reduced_{args.cell_type}_GNN.h5ad")
    test_reduced_adata.write(f"{out_dir}test_reduced_{args.cell_type}_GNN.h5ad")
    # print(f"[INFO] Reduced embedding shapes:\n  train_reduced : {train_reduced_adata.X.shape}\n  test_reduced  : {test_reduced_adata.X.shape}")
    # print(f"[SAVED] train/test reduced embeddings for {args.cell_type}_GNN")

    # Define consistent label order
    cell_type_list = sorted(pd.unique(train_data.obs["cell_type"]))
    print(f"[INFO] Cell type list: {cell_type_list}")

    # Set which label is positive/negative
    set_eval_labels(pos_label="severe", neg_label="mild")

    _eval_and_save(
        AnnData(X=Ztr, obs=train_data.obs.copy()),
        AnnData(X=Zte, obs=test_data.obs.copy()),
        "GNN",
        f"{out_dir}results_classifiers_{args.cell_type}_GNN.csv",
        f"{out_dir}logreg_feature_importance_{args.cell_type}_GNN.csv",
        sorted(pd.unique(train_data.obs["cell_type"])),
        mild_ind,
        severe_ind,
    )
    print(f"[INFO] train_s shape : {train_s.shape}")
    print(f"[INFO] train_reduced : {train_reduced_adata.X.shape}")
    print(f"[INFO] test_reduced  : {test_reduced_adata.X.shape}")
    print(f"[SAVED] {out_dir}train_s_{tag}.npy")
    print(f"[SAVED] {out_dir}train_reduced_{tag}.h5ad")
    print(f"[SAVED] {out_dir}test_reduced_{tag}.h5ad")
    print("[INFO][GNN] done.")

# ------------------------- Run sciRED -------------------------
def run_scired():
    print("[INFO] Running sciRED backend (raw counts, no normalization)...")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SkPipe
    from sciRED import glm as sc_glm
    from sciRED import rotations as rot

    data, meta_cell, mild_ind, severe_ind = load_and_label_data()
    train_data, _, test_data = split_by_individual(data, meta_cell, mild_ind, severe_ind)
    gene_list = get_de_gene_list(train_data, ["mild", "severe"], args.cell_type)

    train_data = train_data[:, [g for g in gene_list if g in train_data.var_names]].copy()
    test_data = test_data[:, [g for g in gene_list if g in test_data.var_names]].copy()

    def _dense64(adata):
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        return X.astype(np.float64, copy=False)

    Xtr_counts = _dense64(train_data)
    Xte_counts = _dense64(test_data)
    genes = np.asarray(train_data.var_names)
    G = Xtr_counts.shape[1]

    if "protocol" in train_data.obs.columns:
        prot_tr = pd.get_dummies(train_data.obs["protocol"], drop_first=False).to_numpy(dtype=np.float64)
        prot_te = pd.get_dummies(test_data.obs["protocol"], drop_first=False).to_numpy(dtype=np.float64)
    else:
        prot_tr = np.empty((Xtr_counts.shape[0], 0), dtype=np.float64)
        prot_te = np.empty((Xte_counts.shape[0], 0), dtype=np.float64)

    lib_tr = Xtr_counts.sum(axis=1).reshape(-1, 1)
    lib_te = Xte_counts.sum(axis=1).reshape(-1, 1)
    Dtr = np.column_stack([np.ones((Xtr_counts.shape[0], 1)), lib_tr, prot_tr])
    Dte = np.column_stack([np.ones((Xte_counts.shape[0], 1)), lib_te, prot_te])

    print("[sciRED] GLM residuals (train)...")
    rtr = sc_glm.poissonGLM(y=Xtr_counts, x=Dtr)
    Ytr = rtr["resid_pearson"]
    print("[sciRED] GLM residuals (test)...")
    rte = sc_glm.poissonGLM(y=Xte_counts, x=Dte)
    Yte = rte["resid_pearson"]

    if Ytr.shape[1] != G:
        if Ytr.shape[0] == G:
            Ytr = Ytr.T
        else:
            raise ValueError(f"[sciRED] Unexpected Ytr shape {Ytr.shape}")
    if Yte.shape[1] != G:
        if Yte.shape[0] == G:
            Yte = Yte.T
        else:
            raise ValueError(f"[sciRED] Unexpected Yte shape {Yte.shape}")

    k = int(getattr(args, "n_comp", 40))
    pipe = SkPipe([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=min(k, G), random_state=888))
    ])
    Ztr_pca = pipe.fit_transform(Ytr)
    Zte_pca = pipe.transform(Yte)
    L_pca = pipe.named_steps["pca"].components_.T
    vr = rot.varimax(L_pca)
    L_varimax = vr["rotloading"]
    Ztr_full = rot.get_rotated_scores(Ztr_pca, vr["rotmat"])
    Zte_full = rot.get_rotated_scores(Zte_pca, vr["rotmat"])

    out_dir = f"{d}output_v2/"
    tag = f"{args.cell_type}_sciRED"
    np.save(f"{out_dir}train_s_{tag}.npy", L_varimax.astype(np.float64))
    pd.DataFrame(L_varimax, index=genes,
                 columns=[f"sciRED_{i}" for i in range(L_varimax.shape[1])]) \
        .to_csv(f"{out_dir}varimax_loading_{tag}.csv")

    prog_names = [f"sciRED_{i}" for i in range(L_varimax.shape[1])]
    train_reduced_adata = AnnData(X=Ztr_full, obs=train_data.obs.copy(),
                                  var=pd.DataFrame(index=prog_names))
    test_reduced_adata = AnnData(X=Zte_full, obs=test_data.obs.copy(),
                                 var=pd.DataFrame(index=prog_names))
    train_reduced_adata.write(f"{out_dir}train_reduced_{tag}.h5ad")
    test_reduced_adata.write(f"{out_dir}test_reduced_{tag}.h5ad")
    # print(f"[INFO] Reduced embedding shapes:\n  train_reduced : {train_reduced_adata.X.shape}\n  test_reduced  : {test_reduced_adata.X.shape}")
    # print(f"[SAVED] {out_dir}train_s_{tag}.npy and reduced .h5ad files")

    # Define consistent label order
    cell_type_list = sorted(pd.unique(train_data.obs["cell_type"]))
    print(f"[INFO] Cell type list: {cell_type_list}")

    # Set which label is positive/negative
    set_eval_labels(pos_label="severe", neg_label="mild")
    
    _eval_and_save(
        train_reduced_adata,
        test_reduced_adata,
        "sciRED",
        f"{out_dir}results_classifiers_{args.cell_type}_sciRED.csv",
        f"{out_dir}logreg_feature_importance_{args.cell_type}_sciRED.csv",
        sorted(pd.unique(train_data.obs["cell_type"])),
        mild_ind,
        severe_ind,
    )
    print(f"[INFO] train_s shape : {L_varimax.shape}")
    print(f"[INFO] train_reduced : {train_reduced_adata.X.shape}")
    print(f"[INFO] test_reduced  : {test_reduced_adata.X.shape}")
    print(f"[SAVED] {out_dir}train_s_{tag}.npy")
    print(f"[SAVED] {out_dir}train_reduced_{tag}.h5ad")
    print(f"[SAVED] {out_dir}test_reduced_{tag}.h5ad")
    print(f"[INFO][sciRED] done ({tag}).")


# ------------------------- Dispatch -------------------------
BACKENDS = {"GNN": run_gnn, "sciRED": run_scired}

if __name__ == "__main__":
    backend = args.backend
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Choose from {list(BACKENDS.keys())}")
    BACKENDS[backend]()
