"""
Yost_2019_v2.py
ALL outputs saved into:
./Yost_2019/output_v2/
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scipy.sparse as sp
import warnings
from anndata import AnnData

from SDAN.model import pipeline
from SDAN.args import parse_args
from SDAN.utils import s2name
from SDAN.evaluation import set_eval_labels, eval_and_save as _eval_and_save
from SDAN.preprocess import construct_gene_list

warnings.simplefilter("ignore", FutureWarning)

np.random.seed(888)
torch.manual_seed(888)

sc.settings.verbosity = 3
sc.settings.set_figure_params(figsize=(8, 6), dpi=80)

TRAIN_ROOT = "./SF_2018/"
TEST_ROOT  = "./Yost_2019/"
OUTROOT    = "./Yost_2019/output_v2/"
SDAN_OUT   = f"{OUTROOT}output/"   
os.makedirs(OUTROOT, exist_ok=True)

def track_counts(X, name):
    X = np.asarray(X)
    rs = X.sum(axis=1)
    cs = X.sum(axis=0)
    print(f"\n[TRACK] {name}: {X.shape[0]} × {X.shape[1]}")
    print("  Row-sum quantiles:", np.quantile(rs, [0.1, 0.5, 0.9]))
    print("  Col-sum quantiles:", np.quantile(cs, [0.1, 0.5, 0.9]))


def median_first20(X):
    """
    Median expression of first 20 genes.
    """
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    med = np.median(X[:, :20], axis=0)
    print("\n[TRACK] Median expression of first 20 genes:\n", med)
    return med

def check_rawness(adata, name):
    X = adata.X
    X = X.data if sp.issparse(X) else X
    is_integer = np.all(np.isclose(X, np.round(X)))
    print(f"\n[CHECK] {name}")
    print("  integer only:", is_integer)
    print("  max value:", X.max())
    return is_integer

def prepare_shared_gene_list_from_filtered(train_data, test_data):
    out_path = f"{OUTROOT}shared_gene_list.txt"

    if os.path.exists(out_path):
        print("[INFO] Using cached shared gene list.")
        return pd.Index(pd.read_csv(out_path, header=None).squeeze())

    train_genes = pd.Index(train_data.var_names)
    test_genes  = pd.Index(test_data.var_names)
    shared_genes = train_genes[train_genes.isin(test_genes)]

    pd.Series(shared_genes).to_csv(out_path, index=False, header=False)
    print(f"[INFO] Shared genes (filtered): {len(shared_genes)}")
    return shared_genes

# ------------------------- DE-gene selection -------------------------
def get_de_gene_list(train_data, cell_type_list, tag, out_dir):
    """
    DE-based gene selection using Scanpy's rank_genes_groups with FDR correction.
    """
    out_path = os.path.join(out_dir, f"gene_list_{tag}.npy")  

    print("[INFO] Constructing DE gene list...")
    data_tmp = train_data.copy()

    X = (
        data_tmp.X.toarray()
        if hasattr(data_tmp.X, "toarray")
        else np.asarray(data_tmp.X)
    ).astype(np.float64, copy=False)

    X[~np.isfinite(X)] = 0.0
    data_tmp.X = X

    gene_list = construct_gene_list(
        data_tmp,
        cell_type_list,
        n_top_genes=args.n_top_genes,
        method="fdr_bh", 
        alpha=0.05,
    )

    gene_list = pd.Index(gene_list.astype(str))
    np.save(out_path, np.array(gene_list))

    print(f"[INFO] Number of selected genes: {len(gene_list)}")
    print(f"[SAVED] {out_path}")

    return gene_list

# -------------- Parse command-line arguments --------------
args = parse_args()

def run_gnn():
    args = parse_args()
    cell_type = args.cell_type
    tag = f"{cell_type}_GNN_{args.graph_weight}"

    # ---------------- Load SF2018 ----------------
    train_data = sc.read(f"{TRAIN_ROOT}data/{cell_type}_tpm.tsv.gz").transpose()
    gene_names = pd.read_csv(f"{TRAIN_ROOT}data/{cell_type}_gene_info.tsv", sep="\t")
    meta_cell  = pd.read_csv(f"{TRAIN_ROOT}data/{cell_type}_cell_info.tsv", sep="\t")

    train_data.var_names = gene_names["gene"].values
    train_data.obs["cell_type"] = pd.Categorical(meta_cell["response"])
    train_data.obs["sample"]   = meta_cell["sample"].values

    # Check rawness and then reverse-engineer
    track_counts(train_data.X, "SF_2018 raw")
    median_first20(train_data.X)
    check_rawness(train_data, "SF_2018")
    train_data.X = np.expm1(
        train_data.X.A if sp.issparse(train_data.X) else train_data.X
    )
    track_counts(train_data.X, "SF_2018 raw by using expm1")
    median_first20(train_data.X)

    # ---------------- Load Yost2019 ----------------
    test_data = sc.read(f"{TEST_ROOT}data/yost_cd8_counts.tsv.gz").transpose()
    meta_cell_test = pd.read_csv(f"{TEST_ROOT}data/yost_cd8_meta.tsv", sep="\t")
    meta_ind_test  = pd.read_excel(
        f"{TEST_ROOT}data/41591_2019_522_MOESM2_ESM.xlsx",
        sheet_name="SuppTable1",
        skiprows=3,
        nrows=15,
    )

    # ---------------- Build labels on Yost 2019 ----------------
    meta_ind_test["Response"].replace({"Yes (CR)": "Yes"}, inplace=True)

    ind_Y = meta_ind_test.loc[meta_ind_test["Response"] == "Yes", "Patient"]
    ind_N = meta_ind_test.loc[meta_ind_test["Response"] == "No",  "Patient"]

    test_data.obs["cell_type"] = np.select(
        [meta_cell_test.patient.isin(ind_Y), meta_cell_test.patient.isin(ind_N)],
        ["Yes", "No"], default="Unknown",
    )
    test_data.obs["cell_type"] = test_data.obs.cell_type.astype("category")
    test_data.obs["individual"] = meta_cell_test["patient"].values

    track_counts(test_data.X, "Yost_2019 raw")
    median_first20(test_data.X)

   # ---------------- Gene Filtering ----------------
    gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")
    train_data = train_data[:, ~train_data.var_names.isin(gene_mito["hgnc_symbol"])]
    prop_nonzero = (train_data.X != 0).sum(axis=0) / train_data.shape[0]
    train_data = train_data[:, prop_nonzero > 0.02]

    gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")
    test_data = test_data[:, ~test_data.var_names.isin(gene_mito["hgnc_symbol"])]
    prop_nonzero = (test_data.X != 0).sum(axis=0) / test_data.shape[0]
    test_data = test_data[:, prop_nonzero > 0.02]

    track_counts(train_data.X, "SF_2018 filtered")
    median_first20(train_data.X)
    track_counts(test_data.X,  "Yost_2019 filtered")
    median_first20(test_data.X)

    # ---------------- Intersection ----------------
    shared_genes = prepare_shared_gene_list_from_filtered(train_data, test_data)
    train_data = train_data[:, shared_genes].copy()
    test_data  = test_data[:, shared_genes].copy()

    assert train_data.var_names.equals(test_data.var_names)

    # ---------------- Normalize AFTER alignment ----------------
    sc.pp.normalize_total(train_data, target_sum=1e4)
    sc.pp.log1p(train_data)
    sc.pp.normalize_total(test_data, target_sum=1e4)
    sc.pp.log1p(test_data)

    track_counts(train_data.X, "SF_2018 normalized")
    track_counts(test_data.X,  "Yost_2019 normalized")

    # ---------------- Train GNN using SF 2018 ----------------
    args.mc_weight = args.graph_weight
    args.o_weight  = args.graph_weight

    (train_GNN, val_GNN), (train_labels, val_labels), cell_type_list, gene_list = \
        pipeline(
            train_data,
            args,
            OUTROOT,
            tag,
            api=True,
        )

    model = torch.load(f"{OUTROOT}output/model_{tag}.pth", weights_only=False)

    train_s = torch.tensor(
        np.load(f"{OUTROOT}output/train_s_{tag}.npy"),
        dtype=torch.float32,
    )

    gene_list = pd.Index(
        pd.read_csv(f"{OUTROOT}output/gene_list_{tag}.txt", header=None).squeeze()
    )

    s2name(train_s, gene_list, tag, OUTROOT)

    # SUBSET train/test to EXACT same genes
    train_data = train_data[:, gene_list].copy()
    test_data  = test_data[:, gene_list].copy()

    assert train_data.n_vars == train_s.shape[0]
    assert train_data.var_names.equals(gene_list), "Gene list mismatch!"

    # ---------------- Project on Yost 2019 ----------------
    Xtr = train_data.X.toarray() if sp.issparse(train_data.X) else train_data.X
    Xte = test_data.X.toarray()  if sp.issparse(test_data.X)  else test_data.X

    Ztr = (torch.tensor(Xtr) @ train_s).numpy()
    Zte = (torch.tensor(Xte) @ train_s).numpy()

    # ---------------- Save reduced ----------------
    prog_names = [f"GNN_{i}" for i in range(train_s.shape[1])] 

    train_red = AnnData(
        Ztr,
        obs=train_data.obs.copy(),
        var=pd.DataFrame(index=prog_names),
    )
    train_red.write(f"{OUTROOT}train_reduced_{tag}.h5ad")

    test_red = AnnData(
        Zte,
        obs=test_data.obs.copy(),
        var=pd.DataFrame(index=prog_names),
    )
    test_red.write(f"{OUTROOT}test_reduced_{tag}.h5ad")

    # Training data: SF_2018 → keep NR / R
    train_mask = train_red.obs["cell_type"].isin(["NR", "R"])

    # Test data: Yost_2019 → keep Yes / No
    test_mask  = test_red.obs["cell_type"].isin(["Yes", "No"])

    train_red = train_red[train_mask].copy()
    test_red  = test_red[test_mask].copy()

    # Harmonize training labels
    train_red.obs["cell_type"] = train_red.obs["cell_type"].replace(
        {"NR": "No", "R": "Yes"}
    ).astype(str)

    # ---------------- Evaluation ----------------
    set_eval_labels(pos_label="Yes", neg_label="No")
    _eval_and_save(
        train_red,
        test_red,
        "GNN",
        f"{OUTROOT}results_classifiers_{tag}.csv",
        f"{OUTROOT}logreg_feature_importance_{tag}.csv",
        ["No", "Yes"],
        ind_N,
        ind_Y,
    )
    print("\n[INFO] GNN pipeline finished successfully.")
    print(f"[INFO] shared genes: {train_data.n_vars}")
    print(f"[INFO] train_s shape: {train_s.shape}")

def run_spectra():
    print("[INFO] Running Spectra backend (train on SF_2018 and test on Yost_2019)...")

    import Spectra
    from torch_geometric.utils import to_scipy_sparse_matrix
    from SDAN.preprocess import construct_gene_graph
    from SDAN.evaluation import set_eval_labels, eval_and_save as _eval_and_save

    # ------------------ Paths ------------------
    OUTROOT    = "./Yost_2019/output_v2/"
    os.makedirs(OUTROOT, exist_ok=True)
    cell_type = args.cell_type

    cell_type_str = args.cell_type
    tag = f"{cell_type_str}_Spectra"

    print("Loading SF_2018 training data...")
 
    # ---------------- Load SF2018 ----------------
    train_data = sc.read(f"{TRAIN_ROOT}data/{cell_type}_tpm.tsv.gz").transpose()
    gene_names = pd.read_csv(f"{TRAIN_ROOT}data/{cell_type}_gene_info.tsv", sep="\t")
    meta_cell  = pd.read_csv(f"{TRAIN_ROOT}data/{cell_type}_cell_info.tsv", sep="\t")

    train_data.var_names = gene_names["gene"].values
    train_data.obs["cell_type"] = pd.Categorical(meta_cell["response"])
    train_data.obs["sample"]   = meta_cell["sample"].values

    track_counts(train_data.X, "SF_2018 raw")
    median_first20(train_data.X)

    # check_rawness(train_data, "SF_2018")
    train_data.X = np.expm1(
        train_data.X.A if sp.issparse(train_data.X) else train_data.X
    )
    track_counts(train_data.X, "SF_2018 raw by using expm1")
    median_first20(train_data.X)

    # ---------------- Load Yost2019 ----------------
    test_data = sc.read(f"{TEST_ROOT}data/yost_cd8_counts.tsv.gz").transpose()
    meta_cell_test = pd.read_csv(f"{TEST_ROOT}data/yost_cd8_meta.tsv", sep="\t")
    meta_ind_test  = pd.read_excel(
        f"{TEST_ROOT}data/41591_2019_522_MOESM2_ESM.xlsx",
        sheet_name="SuppTable1",
        skiprows=3,
        nrows=15,
    )

    # ---------------- Build labels on Yost 2019 ----------------
    meta_ind_test["Response"].replace({"Yes (CR)": "Yes"}, inplace=True)

    ind_Y = meta_ind_test.loc[meta_ind_test["Response"] == "Yes", "Patient"]
    ind_N = meta_ind_test.loc[meta_ind_test["Response"] == "No",  "Patient"]

    test_data.obs["cell_type"] = np.select(
        [meta_cell_test.patient.isin(ind_Y), meta_cell_test.patient.isin(ind_N)],
        ["Yes", "No"], default="Unknown",
    )
    test_data.obs["cell_type"] = test_data.obs.cell_type.astype("category")
    test_data.obs["individual"] = meta_cell_test["patient"].values

    track_counts(test_data.X, "Yost_2019 raw")
    median_first20(test_data.X)

   # ---------------- Gene Filtering ----------------
    gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")
    train_data = train_data[:, ~train_data.var_names.isin(gene_mito["hgnc_symbol"])]
    prop_nonzero = (train_data.X != 0).sum(axis=0) / train_data.shape[0]
    train_data = train_data[:, prop_nonzero > 0.02]

    gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")
    test_data = test_data[:, ~test_data.var_names.isin(gene_mito["hgnc_symbol"])]
    prop_nonzero = (test_data.X != 0).sum(axis=0) / test_data.shape[0]
    test_data = test_data[:, prop_nonzero > 0.02]

    track_counts(train_data.X, "SF_2018 filtered")
    median_first20(train_data.X)
    track_counts(test_data.X,  "Yost_2019 filtered")
    median_first20(test_data.X)

    # ---------------- Intersection ----------------
    shared_genes = prepare_shared_gene_list_from_filtered(train_data, test_data)
    train_data = train_data[:, shared_genes].copy()
    test_data  = test_data[:, shared_genes].copy()

    assert train_data.var_names.equals(test_data.var_names)

    # ---------------- Normalize AFTER alignment ----------------
    sc.pp.normalize_total(train_data, target_sum=1e4)
    sc.pp.log1p(train_data)
    sc.pp.normalize_total(test_data, target_sum=1e4)
    sc.pp.log1p(test_data)

    track_counts(train_data.X, "SF_2018 normalized")
    track_counts(test_data.X,  "Yost_2019 normalized")

    # ---------------- Train Spectra ----------------
    print(f"[INFO] Shared genes after QC: {len(shared_genes)}")

    # Gene selection (DE + HVG)
    gene_list = get_de_gene_list(
        train_data,
        train_data.obs["cell_type"].cat.categories.values,
        tag=tag,
        out_dir=OUTROOT,
    )

    train_data = train_data[:, gene_list].copy()
    test_data  = test_data[:, gene_list].copy()

    # Train Spectra
    Xtr = train_data.X.toarray() if sp.issparse(train_data.X) else train_data.X

    edge_index = construct_gene_graph(train_data.var_names)
    sp_adj = to_scipy_sparse_matrix(edge_index, num_nodes=train_data.n_vars)
    sp_adj = ((sp_adj + sp_adj.T) > 0).astype(float)
    sp_adj.setdiag(0)
    sp_adj.eliminate_zeros()
    A = sp_adj.toarray()

    model = Spectra.SPECTRA_Model(
        X=Xtr,
        labels=None,
        L=int(args.spectra_L),
        adj_matrix=A,
        gs_dict=None,
        lam=args.spectra_lam,
        delta=args.spectra_delta,
        kappa=None,
        rho=args.spectra_rho,
        use_cell_types=False,
        vocab=list(train_data.var_names),
    )
    model.train(X=Xtr, num_epochs=int(args.spectra_epochs))

    # Loadings
    W = model.return_factors()
    S = W.T  # genes × components

    # ---------------- SAVE ----------------
    np.save(f"{OUTROOT}train_s_{tag}.npy", S)
    pd.Series(train_data.var_names).to_csv(
        f"{OUTROOT}gene_list_{tag}.txt", index=False, header=False
    )

    # Projection
    Xte = test_data.X.toarray() if sp.issparse(test_data.X) else test_data.X

    Ztr = model.return_cell_scores()
    Zte = Xte @ S

    prog_names = [f"Spectra_{i}" for i in range(S.shape[1])]
    train_red = AnnData(Ztr, obs=train_data.obs.copy(),
                        var=pd.DataFrame(index=prog_names))
    test_red  = AnnData(Zte, obs=test_data.obs.copy(),
                        var=pd.DataFrame(index=prog_names))

    train_red.write(f"{OUTROOT}train_reduced_{tag}.h5ad")
    test_red.write(f"{OUTROOT}test_reduced_{tag}.h5ad")

    # Training data: SF_2018 → keep NR / R
    train_mask = train_red.obs["cell_type"].isin(["NR", "R"])

    # Test data: Yost_2019 → keep Yes / No
    test_mask  = test_red.obs["cell_type"].isin(["Yes", "No"])

    train_red = train_red[train_mask].copy()
    test_red  = test_red[test_mask].copy()

    # harmonize train labels 
    train_red.obs["cell_type"] = train_red.obs["cell_type"].replace(
        {"NR": "No", "R": "Yes"}
    ).astype(str)
    set_eval_labels(pos_label="Yes", neg_label="No")
    _eval_and_save(
        train_red,
        test_red,
        "Spectra",
        f"{OUTROOT}results_classifiers_{tag}.csv",
        f"{OUTROOT}logreg_feature_importance_{tag}.csv",
        ["No", "Yes"],
        ind_N,
        ind_Y,
    )

    print("[INFO][Spectra] done.")
    print(f"[INFO] train_s shape: {S.shape}")
    print(f"[INFO] shared genes: {len(shared_genes)} genes")
    print(f"[INFO] test_reduced shape: {test_red.shape}")
    print(f"[SAVED] Spectra results → {OUTROOT}")
    print("[INFO][Spectra] done.")

# ------------------ Run sciRED ------------------
def run_scired():
    """
    sciRED backend (RAW COUNTS for modeling).
    Gene selection uses DE + HVG on a SANITIZED COPY
    via get_de_gene_list().
    """

    print("[INFO] Running sciRED backend...")

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    from anndata import AnnData

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SkPipe

    from sciRED import glm as sc_glm
    from sciRED import rotations as rot

    from SDAN.evaluation import set_eval_labels, eval_and_save as _eval_and_save

    # ---------------- Paths / tags ----------------
    TRAIN_ROOT = "./SF_2018/"
    TEST_ROOT  = "./Yost_2019/"
    OUTROOT    = "./Yost_2019/output_v2/"
    os.makedirs(OUTROOT, exist_ok=True)

    cell_type = args.cell_type
    tag = f"{cell_type}_sciRED"

    # -------------------- Load SF-2018 (NON-reversed) --------------------
    # 1. Load SF2018 (train) — RAW COUNTS
    train_data = sc.read(f"{TRAIN_ROOT}data/{cell_type}_tpm.tsv.gz").transpose()
    gene_names = pd.read_csv(f"{TRAIN_ROOT}data/{cell_type}_gene_info.tsv", sep="\t")
    meta_cell  = pd.read_csv(f"{TRAIN_ROOT}data/{cell_type}_cell_info.tsv", sep="\t")

    train_data.var_names = gene_names["gene"].values
    train_data.obs["cell_type"] = pd.Categorical(meta_cell["response"])
    train_data.obs["individual"] = meta_cell["sample"].values

    # Check rawness and then reverse-engineer
    track_counts(train_data.X, "SF_2018 raw")
    median_first20(train_data.X)
    check_rawness(train_data, "SF_2018")

    # -------------------- Load Yost-2019 (RAW counts) --------------------
    # 2. Load Yost2019 (test) — RAW COUNTS
    test_data = sc.read(f"{TEST_ROOT}data/yost_cd8_counts.tsv.gz").transpose()
    meta_cell_test = pd.read_csv(f"{TEST_ROOT}data/yost_cd8_meta.tsv", sep="\t")
    meta_ind_test  = pd.read_excel(
        f"{TEST_ROOT}data/41591_2019_522_MOESM2_ESM.xlsx",
        sheet_name="SuppTable1",
        skiprows=3,
        nrows=15,
    )

    meta_ind_test["Response"].replace({"Yes (CR)": "Yes"}, inplace=True)
    ind_Y = meta_ind_test.loc[meta_ind_test["Response"] == "Yes", "Patient"]
    ind_N = meta_ind_test.loc[meta_ind_test["Response"] == "No",  "Patient"]

    test_data.obs["cell_type"] = np.select(
        [meta_cell_test.patient.isin(ind_Y), meta_cell_test.patient.isin(ind_N)],
        ["Yes", "No"], default="Unknown",
    )
    test_data.obs["cell_type"] = test_data.obs.cell_type.astype("category")
    test_data.obs["individual"] = meta_cell_test["patient"].values

    # -------------------- Gene filtering --------------------
    # 3. Gene filtering (mito + low expression)
    gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")

    train_data = train_data[:, ~train_data.var_names.isin(gene_mito["hgnc_symbol"])]
    train_data = train_data[:, (train_data.X != 0).sum(axis=0) / train_data.shape[0] > 0.02]

    test_data  = test_data[:,  ~test_data.var_names.isin(gene_mito["hgnc_symbol"])]
    test_data  = test_data[:,  (test_data.X != 0).sum(axis=0) / test_data.shape[0] > 0.02]

    # -------------------- Shared genes --------------------
    # 4. Intersection
    shared_genes = prepare_shared_gene_list_from_filtered(train_data, test_data)
    train_data = train_data[:, shared_genes].copy()
    test_data  = test_data[:, shared_genes].copy()

    assert train_data.var_names.equals(test_data.var_names)

    track_counts(train_data.X, "SF_2018 ")
    track_counts(test_data.X,  "Yost_2019 ")

    # -------------------- DE gene list --------------------
    # 5. Gene selection (DE + HVG) 
    gene_list = get_de_gene_list(
        train_data,
        train_data.obs["cell_type"].cat.categories.values,
        tag=tag,
        out_dir=OUTROOT,
    )

    # # reverse-engineer 
    # train_data.X = np.expm1(
    #     train_data.X.A if sp.issparse(train_data.X) else train_data.X
    # )

    train_data = train_data[:, gene_list].copy()
    test_data  = test_data[:, gene_list].copy()

    genes = np.asarray(train_data.var_names)
    G = train_data.n_vars
    print(f"[INFO] Number of selected genes: {G}")

    track_counts(train_data.X, "SF_2018 ")
    track_counts(test_data.X,  "Yost_2019 ")

    # ---------------- Train Spectra ----------------
    print(f"[INFO] Shared genes after QC: {len(shared_genes)}")

    # ---------------- sciRED Poisson GLM -----------
    # 6. sciRED Poisson GLM (RAW COUNTS)
    def _dense64(adata):
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        return X.astype(np.float64, copy=False)

    Xtr = _dense64(train_data)
    Xte = _dense64(test_data)

    lib_tr = Xtr.sum(axis=1).reshape(-1, 1)
    lib_te = Xte.sum(axis=1).reshape(-1, 1)

    Dtr = np.column_stack([np.ones((Xtr.shape[0], 1)), lib_tr])
    Dte = np.column_stack([np.ones((Xte.shape[0], 1)), lib_te])

    print("[sciRED] Fitting Poisson GLM (train)...")
    rtr = sc_glm.poissonGLM(y=Xtr, x=Dtr)
    Ytr = rtr["resid_pearson"]

    print("[sciRED] Fitting Poisson GLM (test)...")
    rte = sc_glm.poissonGLM(y=Xte, x=Dte)
    Yte = rte["resid_pearson"]

    if Ytr.shape[1] != G:
        Ytr = Ytr.T
    if Yte.shape[1] != G:
        Yte = Yte.T

    # 7. PCA + Varimax
    k = int(getattr(args, "n_comp", 40))
    pipe = SkPipe([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(k, G), random_state=888)),
    ])

    Ztr_pca = pipe.fit_transform(Ytr)
    Zte_pca = pipe.transform(Yte)

    L_pca = pipe.named_steps["pca"].components_.T
    vr = rot.varimax(L_pca)
    L = vr["rotloading"]

    Ztr = rot.get_rotated_scores(Ztr_pca, vr["rotmat"])
    Zte = rot.get_rotated_scores(Zte_pca, vr["rotmat"])

    # 8. Save
    np.save(f"{OUTROOT}train_s_{tag}.npy", L)
    pd.Series(genes).to_csv(f"{OUTROOT}gene_list_{tag}.txt", index=False, header=False)

    prog_names = [f"sciRED_{i}" for i in range(L.shape[1])]
    train_red = AnnData(Ztr, obs=train_data.obs.copy(),
                        var=pd.DataFrame(index=prog_names))
    test_red  = AnnData(Zte, obs=test_data.obs.copy(),
                        var=pd.DataFrame(index=prog_names))

    train_red.write(f"{OUTROOT}train_reduced_{tag}.h5ad")
    test_red.write(f"{OUTROOT}test_reduced_{tag}.h5ad")

    # -------------------- Labels + evaluation --------------------
    if "cell_type" not in train_red.obs.columns:
        train_red.obs["cell_type"] = train_data.obs["cell_type"].astype(str)

    if "cell_type" not in test_red.obs.columns:
        test_red.obs["cell_type"] = test_data.obs["cell_type"].astype(str)

    # CORRECT LABEL FILTERING
    # Training data: SF_2018 → keep NR / R
    train_mask = train_red.obs["cell_type"].isin(["NR", "R"])

    # Test data: Yost_2019 → keep Yes / No
    test_mask  = test_red.obs["cell_type"].isin(["Yes", "No"])

    train_red = train_red[train_mask].copy()
    test_red  = test_red[test_mask].copy()

    print("[DEBUG] train labels:")
    print(train_red.obs["cell_type"].value_counts(dropna=False))
    print("[DEBUG] test labels:")
    print(test_red.obs["cell_type"].value_counts(dropna=False))

    # harmonize train labels 
    train_red.obs["cell_type"] = train_red.obs["cell_type"].replace(
        {"NR": "No", "R": "Yes"}
    ).astype(str)

    # 9. Evaluation
    set_eval_labels(pos_label="Yes", neg_label="No")

    _eval_and_save(
        train_red,
        test_red,
        "sciRED",
        f"{OUTROOT}results_classifiers_{tag}.csv",
        f"{OUTROOT}logreg_feature_importance_{tag}.csv",
        ["No", "Yes"],
        ind_N,
        ind_Y,
    )

    print("[INFO][sciRED] finished successfully.")
    print(f"[INFO] train_s shape: {L.shape}")
    print(f"[INFO] train_reduced shape: {train_red.shape}")
    print(f"[INFO] test_reduced shape: {test_red.shape}")


# -------------- Backend Dispatcher --------------
BACKENDS = {"GNN": run_gnn, "Spectra": run_spectra, "sciRED": run_scired}

if __name__ == "__main__":
    backend = args.backend
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Choose from {list(BACKENDS.keys())}")
    try:
        BACKENDS[backend]()
    except Exception as e:
        print("\n[DEBUG] Exception caught:", type(e).__name__)
        print("[DEBUG] Message:", e)
        print("[DEBUG] Full traceback:")
        traceback.print_exc(file=sys.stdout)





