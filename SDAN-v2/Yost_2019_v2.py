"""
This file unified pipeline for SDAN, Spectra, sciRED, and scNET backends.
Compatible with args.py and preprocess.py.
ALL outputs saved into: ./Yost_2019/output_v2/
The backend output folders are:
    SDAN: output_v2/SDAN
    Spectra: output_v2/Spectra
    sciRED: output_v2/sciRED
    scNET: output_v2/scNET
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
from tqdm import tqdm
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
def get_de_gene_list(train_data, cell_type_list, tag, out_dir, n_top_genes=None):
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
        n_top_genes=args.n_top_genes if n_top_genes is None else n_top_genes,
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

def run_sdan():
    args = parse_args()
    cell_type = args.cell_type
    tag = f"{args.cell_type}_SDAN_{args.graph_weight}"
    out_dir = os.path.join(OUTROOT, "SDAN")
    os.makedirs(out_dir, exist_ok=True)

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

    # ---------------- Train SDAN using SF 2018 ----------------
    args.mc_weight = args.graph_weight
    args.o_weight  = args.graph_weight

    (train_SDAN, val_SDAN), (train_labels, val_labels), cell_type_list, gene_list = \
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
    pd.Series(gene_list).to_csv(
        os.path.join(out_dir, f"gene_list_{tag}.txt"), index=False, header=False
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
    prog_names = [f"SDAN_{i}" for i in range(train_s.shape[1])] 

    train_red = AnnData(
        Ztr,
        obs=train_data.obs.copy(),
        var=pd.DataFrame(index=prog_names),
    )
    train_red.write(os.path.join(out_dir, f"train_reduced_{tag}.h5ad"))

    test_red = AnnData(
        Zte,
        obs=test_data.obs.copy(),
        var=pd.DataFrame(index=prog_names),
    )
    test_red.write(os.path.join(out_dir, f"test_reduced_{tag}.h5ad"))

    # Training data: SF_2018 to keep NR / R
    train_mask = train_red.obs["cell_type"].isin(["NR", "R"])

    # Test data: Yost_2019 to keep Yes / No
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
        "SDAN",
        os.path.join(out_dir, f"results_classifiers_{tag}.csv"),
        os.path.join(out_dir, f"logreg_feature_importance_{tag}.csv"),
        ["No", "Yes"],
        ind_N,
        ind_Y,
    )
    print("\n[INFO] SDAN pipeline finished successfully.")
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
    out_dir = os.path.join(OUTROOT, "Spectra")
    os.makedirs(out_dir, exist_ok=True)
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
        out_dir=out_dir,
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
    np.save(os.path.join(out_dir, f"train_s_{tag}.npy"), S)
    pd.Series(train_data.var_names).to_csv(
        os.path.join(out_dir, f"gene_list_{tag}.txt"), index=False, header=False
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

    train_red.write(os.path.join(out_dir, f"train_reduced_{tag}.h5ad"))
    test_red.write(os.path.join(out_dir, f"test_reduced_{tag}.h5ad"))

    # Training data: SF_2018 to keep NR / R
    train_mask = train_red.obs["cell_type"].isin(["NR", "R"])

    # Test data: Yost_2019 to keep Yes / No
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
        os.path.join(out_dir, f"results_classifiers_{tag}.csv"),
        os.path.join(out_dir, f"logreg_feature_importance_{tag}.csv"),
        ["No", "Yes"],
        ind_N,
        ind_Y,
    )

    print("[INFO][Spectra] done.")
    print(f"[INFO] train_s shape: {S.shape}")
    print(f"[INFO] shared genes: {len(shared_genes)} genes")
    print(f"[INFO] test_reduced shape: {test_red.shape}")
    print(f"[SAVED] Spectra results to {out_dir}")
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
    out_dir = os.path.join(OUTROOT, "sciRED")
    os.makedirs(out_dir, exist_ok=True)

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
        out_dir=out_dir,
    )
    
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
    np.save(os.path.join(out_dir, f"train_s_{tag}.npy"), L)
    pd.Series(genes).to_csv(os.path.join(out_dir, f"gene_list_{tag}.txt"), index=False, header=False)

    prog_names = [f"sciRED_{i}" for i in range(L.shape[1])]
    train_red = AnnData(Ztr, obs=train_data.obs.copy(),
                        var=pd.DataFrame(index=prog_names))
    test_red  = AnnData(Zte, obs=test_data.obs.copy(),
                        var=pd.DataFrame(index=prog_names))

    train_red.write(os.path.join(out_dir, f"train_reduced_{tag}.h5ad"))
    test_red.write(os.path.join(out_dir, f"test_reduced_{tag}.h5ad"))

    # -------------------- Labels + evaluation --------------------
    if "cell_type" not in train_red.obs.columns:
        train_red.obs["cell_type"] = train_data.obs["cell_type"].astype(str)

    if "cell_type" not in test_red.obs.columns:
        test_red.obs["cell_type"] = test_data.obs["cell_type"].astype(str)

    # CORRECT LABEL FILTERING
    # Training data: SF_2018 to keep NR / R
    train_mask = train_red.obs["cell_type"].isin(["NR", "R"])

    # Test data: Yost_2019 to keep Yes / No
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
        os.path.join(out_dir, f"results_classifiers_{tag}.csv"),
        os.path.join(out_dir, f"logreg_feature_importance_{tag}.csv"),
        ["No", "Yes"],
        ind_N,
        ind_Y,
    )

    print("[INFO][sciRED] finished successfully.")
    print(f"[INFO] train_s shape: {L.shape}")
    print(f"[INFO] train_reduced shape: {train_red.shape}")
    print(f"[INFO] test_reduced shape: {test_red.shape}")

# -------------- scNET --------------
def run_scnet():
    print("[INFO] Running scNET backend (train on SF_2018 to test on Yost_2019)...")

    import scNET
    from scNET.MultyGraphModel import scNET as scNET_model
    from scNET.Utils import save_obj
    from torch_geometric.data import Data
    from torch_geometric.utils import train_test_split_edges
    from SDAN.preprocess import construct_gene_graph
    # ------------------ Paths ------------------
    out_dir = os.path.join(OUTROOT, "scNET")
    os.makedirs(out_dir, exist_ok=True)

    cell_type = args.cell_type
    tag = f"{cell_type}_scNET75"
    model_name = tag
    device = scNET.main.device

    # LOAD DATA 
    train_data = sc.read(f"{TRAIN_ROOT}data/{cell_type}_tpm.tsv.gz").transpose()
    gene_names = pd.read_csv(f"{TRAIN_ROOT}data/{cell_type}_gene_info.tsv", sep="\t")
    meta_cell  = pd.read_csv(f"{TRAIN_ROOT}data/{cell_type}_cell_info.tsv", sep="\t")

    train_data.var_names = gene_names["gene"].values
    train_data.obs["cell_type"] = pd.Categorical(meta_cell["response"])
    train_data.obs["individual"] = meta_cell["sample"].values

    # reverse TPM to counts
    train_data.X = np.expm1(
        train_data.X.A if sp.issparse(train_data.X) else train_data.X
    )

    # ---------------- Yost ----------------
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

    # FILTER + ALIGN
    gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")

    train_data = train_data[:, ~train_data.var_names.isin(gene_mito["hgnc_symbol"])]
    train_data = train_data[:, (train_data.X != 0).sum(axis=0) / train_data.shape[0] > 0.02]

    test_data  = test_data[:,  ~test_data.var_names.isin(gene_mito["hgnc_symbol"])]
    test_data  = test_data[:,  (test_data.X != 0).sum(axis=0) / test_data.shape[0] > 0.02]

    shared_genes = prepare_shared_gene_list_from_filtered(train_data, test_data)
    train_data = train_data[:, shared_genes].copy()
    test_data  = test_data[:, shared_genes].copy()

    # NORMALIZE
    sc.pp.normalize_total(train_data, target_sum=1e4)
    sc.pp.log1p(train_data)

    sc.pp.normalize_total(test_data, target_sum=1e4)
    sc.pp.log1p(test_data)

    # DE genes are selected on train only, then intersected with test genes.
    gene_list = get_de_gene_list(
        train_data,
        train_data.obs["cell_type"].cat.categories.values,
        tag=tag,
        out_dir=OUTROOT,
    )

    gene_list = pd.Index(gene_list.astype(str))
    overlap_gene_list = gene_list[gene_list.isin(test_data.var_names)]
    print(f"[INFO] DE genes selected on train: {len(gene_list)}")
    print(f"[INFO] Overlap genes used for scNET: {len(overlap_gene_list)}")
    overlap_gene_path = os.path.join(out_dir, f"gene_list_{tag}.npy")
    np.save(overlap_gene_path, np.array(overlap_gene_list))
    print(f"[SAVED] {overlap_gene_path}")

    # Subset to the overlapping gene set for both datasets.
    train_data = train_data[:, overlap_gene_list].copy()
    test_data  = test_data[:, overlap_gene_list].copy()

    print(f"[INFO] Number of DE genes: {len(overlap_gene_list)}")

    # 5. BUILD GRAPH
    edge_index, _ = construct_gene_graph(overlap_gene_list.tolist())

    if isinstance(edge_index, list):
        edge_index = np.array(edge_index)

    if edge_index.ndim == 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    elif edge_index.ndim == 1:
        edge_index = edge_index.reshape(-1, 2).T

    assert edge_index.shape[0] == 2
    print(f"[INFO] Fixed edge_index shape: {edge_index.shape}")

    # Save original cell names before concatenation.
    train_cells = train_data.obs_names.tolist()
    test_cells = test_data.obs_names.tolist()

    # Transductive concatenate, keeping all train cells and subsampling test only.
    train_data.obs["split"] = "train"
    test_data.obs["split"] = "test"
    obj = train_data.concatenate(test_data, index_unique=None)

    train_sub = obj[obj.obs["split"] == "train"].copy()
    test_sub  = obj[obj.obs["split"] == "test"].copy()
    sc.pp.subsample(test_sub, n_obs=min(7500, test_sub.n_obs), random_state=888)

    obj = train_sub.concatenate(test_sub, index_unique=None)
    print(f"[INFO] After subsample: {obj.shape}")
    print(
        f"[INFO] scNET retained cells: train={train_sub.n_obs}, "
        f"test={test_sub.n_obs}, total={obj.n_obs}"
    )

    if sp.issparse(obj.X):
        obj.X = obj.X.toarray()
    obj.X = np.asarray(obj.X, dtype=np.float32)

    print(
        "Check data's scale:",
        "min =", obj.X.min(),
        "max =", obj.X.max(),
        "mean =", obj.X.mean(),
    )
    sc.pp.neighbors(obj, n_neighbors=10, n_pcs=15)

    # Align graph indices to the concatenated object.
    gene_to_idx = {g: i for i, g in enumerate(obj.var_names)}
    genes = list(overlap_gene_list)
    edges = []
    for i in range(edge_index.shape[1]):
        g1 = genes[edge_index[0, i]]
        g2 = genes[edge_index[1, i]]
        if g1 in gene_to_idx and g2 in gene_to_idx:
            edges.append([gene_to_idx[g1], gene_to_idx[g2]])

    if len(edges) == 0:
        raise ValueError("No valid edges after alignment")

    ppi_edge_index = torch.tensor(edges, dtype=torch.long).T.to(device)
    print(f"[INFO] gene graph: {ppi_edge_index.shape}")

    node_feature = obj.X.T  # genes x cells
    print(f"[INFO] expression: {node_feature.shape}")

    knn_edge_index, highly_variable_index = scNET.main.build_knn_graph(obj)
    print(f"[INFO] knn: {knn_edge_index.shape}")

    x = torch.tensor(node_feature, dtype=torch.float32)
    x = ((x.T - x.mean(dim=1)) / (x.std(dim=1) + 1e-5)).T

    data = Data(x=x, edge_index=ppi_edge_index.cpu())
    data = train_test_split_edges(data)
    data = data.to(device)
    x = x.to(device)
    ppi_edge_index = ppi_edge_index.to(device)

    loader = scNET.main.mini_batch_knn(
        knn_edge_index,
        max(1, knn_edge_index.shape[1] // max(1, args.scnet_batches))
    )

    embedding_dim = args.n_comp
    model = scNET_model(
        x.shape[0],
        x.shape[1],
        250,
        embedding_dim,
        250,
        embedding_dim,
        lambda_rows=1,
        lambda_cols=1,
        num_layers=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    print("[INFO] Training scNET...")
    epoch_bar = tqdm(range(args.scnet_epochs), desc="scNET Training", total=args.scnet_epochs)
    for epoch in epoch_bar:
        model.train()
        for batch in loader:
            knn_edge_index_batch = batch.T.to(device)
            loss, _, _ = model.calculate_loss(
                x,
                knn_edge_index_batch,
                data.train_pos_edge_index,
                highly_variable_index,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

    # ---------------- SAVE MODEL ----------------
    model_path = os.path.join(out_dir, f"models_{tag}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "gene_list": list(overlap_gene_list),
        "args": vars(args),
    }, model_path)
    print(f"[SAVED MODEL] {model_path}")

    print("[INFO] Saving embeddings...")
    full_knn_edge_index = torch.cat([batch.T.to(device) for batch in loader], dim=1)
    model.eval()
    with torch.no_grad():
        row_embed, col_embed, out_features = model(
            x,
            full_knn_edge_index,
            data.train_pos_edge_index,
        )

    row_embed_np = row_embed.detach().cpu().numpy()
    col_embed_np = col_embed.detach().cpu().numpy()
    out_features_np = out_features.detach().cpu().numpy()
    print("[DEBUG] row_embed shape:", row_embed_np.shape)
    print("[DEBUG] embedded_cells shape:", col_embed_np.shape)

    node_features = pd.DataFrame(
        row_embed_np,
        index=obj.var_names,
        columns=[f"dim_{i}" for i in range(row_embed_np.shape[1])]
    )
    embedded_cells_df = pd.DataFrame(
        col_embed_np,
        index=obj.obs_names,
        columns=[f"dim_{i}" for i in range(col_embed_np.shape[1])]
    )

    import pkg_resources
    embed_dir = os.path.join(out_dir)
    os.makedirs(embed_dir, exist_ok=True)

    node_path = os.path.join(embed_dir, f"node_features_{model_name}.pkl")
    cell_path = os.path.join(embed_dir, f"embedded_cells_{model_name}.pkl")
    out_features_path = os.path.join(embed_dir, f"out_features_{model_name}.pkl")

    node_features.to_pickle(node_path)
    embedded_cells_df.to_pickle(cell_path)
    save_obj(out_features_np, os.path.join(embed_dir, f"out_features_{model_name}"))

    print(f"[SAVED] {node_path} shape={node_features.shape}")
    print(f"[SAVED] {cell_path} shape={embedded_cells_df.shape}")
    print(f"[SAVED] {out_features_path} shape={out_features_np.shape}")

    pkg_embed_dir = pkg_resources.resource_filename(scNET.__name__, "./Embedding/")
    os.makedirs(pkg_embed_dir, exist_ok=True)
    node_features.to_pickle(os.path.join(pkg_embed_dir, f"node_features_{model_name}"))
    save_obj(row_embed_np, os.path.join(pkg_embed_dir, f"row_embedding_{model_name}"))
    save_obj(col_embed_np, os.path.join(pkg_embed_dir, f"col_embedding_{model_name}"))
    save_obj(out_features_np, os.path.join(pkg_embed_dir, f"out_features_{model_name}"))

    _, embedded_cells, _, _ = scNET.load_embeddings(model_name)
    embedded_cells = embedded_cells.values if hasattr(embedded_cells, "values") else np.asarray(embedded_cells)

    # Split embedding
    cell_to_idx = {cell: i for i, cell in enumerate(obj.obs_names)}

    train_cells_sub = [c for c in train_cells if c in cell_to_idx]
    test_cells_sub  = [c for c in test_cells if c in cell_to_idx]

    Ztr = embedded_cells[[cell_to_idx[c] for c in train_cells_sub]]
    Zte = embedded_cells[[cell_to_idx[c] for c in test_cells_sub]]

    print("Train cells after subsample:", len(Ztr))
    print("Test cells after subsample:", len(Zte))
    print(f"[INFO] train: {Ztr.shape}, test: {Zte.shape}")

    train_red = AnnData(X=Ztr, obs=train_data.obs.loc[train_cells_sub].copy())
    test_red = AnnData(X=Zte, obs=test_data.obs.loc[test_cells_sub].copy())

    train_red.write(os.path.join(out_dir, f"train_reduced_{tag}.h5ad"))
    test_red.write(os.path.join(out_dir, f"test_reduced_{tag}.h5ad"))

    # 10. LABEL FIX
    train_red = train_red[train_red.obs["cell_type"].isin(["NR", "R"])].copy()
    test_red  = test_red[test_red.obs["cell_type"].isin(["Yes", "No"])].copy()

    train_red.obs["cell_type"] = train_red.obs["cell_type"].replace(
        {"NR": "No", "R": "Yes"}
    )

    # 11. EVALUATION
    set_eval_labels(pos_label="Yes", neg_label="No")

    _eval_and_save(
        train_red,
        test_red,
        "scNET",
        os.path.join(out_dir, f"results_classifiers_{tag}.csv"),
        os.path.join(out_dir, f"logreg_feature_importance_{tag}.csv"),
        ["No", "Yes"],
        ind_N,
        ind_Y,
    )

    print("[INFO][scNET] done.")

# -------------- Backend Dispatcher --------------
BACKENDS = {"SDAN": run_sdan, "Spectra": run_spectra, "sciRED": run_scired, "scNET": run_scnet}

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
