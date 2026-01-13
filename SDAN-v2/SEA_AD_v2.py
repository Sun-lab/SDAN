"""
SEA_AD_v2.py — unified pipeline for GNN and sciRED backends.
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

from SDAN.args import parse_args
from SDAN.model import pipeline
from SDAN.preprocess import qc, construct_gene_graph, construct_gene_list
from SDAN.evaluation import set_eval_labels, eval_and_save as _eval_and_save

try:
    import Spectra
except Exception:
    Spectra = None

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(888)
torch.manual_seed(888)
sc.settings.verbosity = 1
sc.settings.set_figure_params(figsize=(8, 6), dpi=80, facecolor="white")

# ------------------------- Paths & args -------------------------
args = parse_args()
d = "./SEA_AD/"
adata_path = f"{d}data/{args.cell_type}.h5ad"
donor_xlsx = f"{d}data/sea-ad_cohort_donor_metadata_020624.xlsx"
os.makedirs(f"{d}output_v2/", exist_ok=True)
# os.makedirs(f"{d}figures/", exist_ok=True)

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

# ------------------------- Load and label -------------------------
def load_and_label_data():
    """Load SEA_AD dataset, remove mito genes, and assign Dementia vs No Dementia labels."""
    print("[INFO] Loading raw data...")
    data = sc.read(adata_path, cache=True)
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

    return data, T, C


# ------------------------- Split by donor -------------------------
def split_by_donor(data, T, C):
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

    donors = data.obs["donor_id"].astype(str)
    train_data = data[donors.isin(Tr)].copy()
    val_data = data[donors.isin(Va)].copy()
    test_data = data[donors.isin(Te)].copy()

    print(f"[INFO] Split complete: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
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
    data, T, C = load_and_label_data()
    qc(data)
    train_data, val_data, test_data = split_by_donor(data, T, C)
    gene_list = get_de_gene_list(train_data, train_data.obs["cell_type"].cat.categories.values,
                                 tag=f"{args.cell_type}_{args.graph_weight}")   
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
        [train_data, val_data, test_data], args, d, tag
    )

    train_s = torch.tensor(np.load(f"{d}output/train_s_{tag}.npy"))
    np.save(f"{out_dir}train_s_{args.cell_type}_GNN.npy", train_s.detach().cpu().numpy())

    Ztr = train_GNN.x.t().detach().cpu().numpy() @ train_s.detach().cpu().numpy()
    Zte = test_GNN.x.t().detach().cpu().numpy() @ train_s.detach().cpu().numpy()

    train_reduced_adata = AnnData(X=Ztr, obs=train_data.obs.copy())
    test_reduced_adata = AnnData(X=Zte, obs=test_data.obs.copy())
    train_reduced_adata.write(f"{out_dir}train_reduced_{args.cell_type}_GNN.h5ad")
    test_reduced_adata.write(f"{out_dir}test_reduced_{args.cell_type}_GNN.h5ad")

    ensure_individual_col(train_reduced_adata)
    ensure_individual_col(test_reduced_adata)

    # Define consistent label order
    cell_type_list = sorted(pd.unique(train_data.obs["cell_type"]))
    print(f"[INFO] Cell type list: {cell_type_list}")

    set_eval_labels(pos_label="Dementia", neg_label="No dementia")
    _eval_and_save(
        train_reduced_adata,
        test_reduced_adata,
        "GNN",
        f"{out_dir}results_classifiers_{args.cell_type}_GNN.csv",
        f"{out_dir}logreg_feature_importance_{args.cell_type}_GNN.csv",
        sorted(pd.unique(train_data.obs["cell_type"])),
        C, T
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

    data, T, C = load_and_label_data()
    train_data, _, test_data = split_by_donor(data, T, C)
    gene_list = get_de_gene_list(train_data,
                                 train_data.obs["cell_type"].cat.categories.values,
                                 tag=args.cell_type)
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
    ensure_individual_col(train_reduced_adata)
    ensure_individual_col(test_reduced_adata)

    # Define consistent label order
    cell_type_list = sorted(pd.unique(train_data.obs["cell_type"]))
    print(f"[INFO] Cell type list: {cell_type_list}")
    
    set_eval_labels(pos_label="Dementia", neg_label="No dementia")
    _eval_and_save(
        train_reduced_adata, test_reduced_adata,
        "sciRED",
        f"{out_dir}results_classifiers_{args.cell_type}_sciRED.csv",
        f"{out_dir}logreg_feature_importance_{args.cell_type}_sciRED.csv",
        sorted(pd.unique(train_data.obs["cell_type"])),
        C, T,
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

