"""
This file unified pipeline for SDAN, sciRED, and scNET backends.
Compatible with args.py and preprocess.py.
ALL outputs saved into: ./SEA_AD/output_comparison/
The backend output folders are:
    SDAN: output_comparison/SDAN
    sciRED: output_comparison/sciRED
    scNET: output_comparison/scNET
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
from tqdm import tqdm
from anndata import AnnData
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
os.makedirs(f"{d}output_comparison/", exist_ok=True)
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

# ------------------------- Split -------------------------
def split_by_donor(data, T, C):
    print("[INFO] Splitting by donor...")
    test_ind = pd.concat([
        pd.Series(T).sample(n=math.floor(0.5 * len(T))),
        pd.Series(C).sample(n=math.floor(0.5 * len(C)))
    ])
    train_ind = pd.concat([pd.Series(T), pd.Series(C)]).drop(test_ind.index)

    train_data = data[data.obs["donor_id"].isin(train_ind)].copy()
    test_data = data[data.obs["donor_id"].isin(test_ind)].copy()
    val_data_obs = train_data.obs.sample(n=math.floor(0.1 * len(train_data)))
    val_data = train_data[val_data_obs.index].copy()
    train_data_obs = train_data.obs.drop(val_data_obs.index)
    train_data = train_data[train_data_obs.index].copy()

    print(f"[INFO] Split complete: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
    return train_data, val_data, test_data
    
# ------------------------- DE-gene selection -------------------------
def get_de_gene_list(train_data, cell_type_list, tag, out_dir=None):
    if out_dir is None:
        out_dir = f"{d}output_comparison/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gene_list_{tag}.npy")
    print("[INFO] Constructing DE gene list...")
    X = train_data.X.toarray() if sp.issparse(train_data.X) else np.asarray(train_data.X)
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = 0.0

    train_data = train_data.copy()
    train_data.X = X

    de_input = train_data.copy()
    if float(np.nanmax(X)) > 50:
        sc.pp.normalize_total(de_input, target_sum=1e4)
        sc.pp.log1p(de_input)

    gene_list = construct_gene_list(
        de_input,
        cell_type_list,
        n_top_genes=args.n_top_genes,
        method="fdr_bh",
        alpha=0.05,
    )
    gene_list = pd.Index(gene_list.astype(str))
    np.save(out_path, np.array(gene_list))
    print(f"[SAVED] {out_path}")
    return gene_list


# ------------------------- SDAN -------------------------
def run_sdan():

    print("[INFO] Running SDAN backend ...")
    data, T, C = load_and_label_data()
    qc(data)
    train_data, val_data, test_data = split_by_donor(data, T, C)
    out_dir = os.path.join(d, "output_comparison", "SDAN")
    os.makedirs(out_dir, exist_ok=True)
    # Keep SDAN feature selection inside `pipeline()` so the comparison run
    # uses the same DE/HVG logic as the standalone SDAN path.
    # gene_list = get_de_gene_list(
    #     train_data,
    #     train_data.obs["cell_type"].cat.categories.values,
    #     tag=args.cell_type,
    #     out_dir=out_dir,
    # )
    # print(f"[INFO] The number of DE genes: {len(gene_list)}")
    #
    # for ad in [train_data, val_data, test_data]:
    #     ad._inplace_subset_var([g for g in gene_list if g in ad.var_names])
    #     ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")
    for ad in [train_data, val_data, test_data]:
        ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")
        
    args.mc_weight = args.graph_weight
    args.o_weight = args.graph_weight
    # tag = f"{args.cell_type}_{args.graph_weight:.1f}"
    tag = f"{args.cell_type}_SDAN_{args.graph_weight}"
    print("[INFO] Launching SDAN pipeline...")
    for ad in [train_data, val_data, test_data]:
        if sp.issparse(ad.X):
            ad.X = ad.X.toarray().astype(np.float32)
        else:
            ad.X = ad.X.astype(np.float32)
    torch.set_default_dtype(torch.float32)

    (train_SDAN, val_SDAN, test_SDAN), _, cell_type_list, gene_list = pipeline(
        [train_data, val_data, test_data], args, d, tag
    )

    train_s = torch.tensor(np.load(f"{d}output/train_s_{tag}.npy"))
    np.save(os.path.join(out_dir, f"train_s_{tag}.npy"), train_s.detach().cpu().numpy())

    Ztr = train_SDAN.x.t().detach().cpu().numpy() @ train_s.detach().cpu().numpy()
    Zte = test_SDAN.x.t().detach().cpu().numpy() @ train_s.detach().cpu().numpy()

    train_reduced_adata = AnnData(X=Ztr, obs=train_data.obs.copy())
    test_reduced_adata = AnnData(X=Zte, obs=test_data.obs.copy())
    train_reduced_adata.write(os.path.join(out_dir, f"train_reduced_{tag}.h5ad"))
    test_reduced_adata.write(os.path.join(out_dir, f"test_reduced_{tag}.h5ad"))

    ensure_individual_col(train_reduced_adata)
    ensure_individual_col(test_reduced_adata)

    # Define consistent label order
    cell_type_list = sorted(pd.unique(train_data.obs["cell_type"]))
    print(f"[INFO] Cell type list: {cell_type_list}")

    set_eval_labels(pos_label="Dementia", neg_label="No dementia")
    _eval_and_save(
        train_reduced_adata,
        test_reduced_adata,
        "SDAN",
        os.path.join(out_dir, f"results_classifiers_{tag}.csv"),
        os.path.join(out_dir, f"logreg_feature_importance_{tag}.csv"),
        sorted(pd.unique(train_data.obs["cell_type"])),
        C, T
    )

    print(f"[INFO] train_s shape : {train_s.shape}")
    print(f"[INFO] train_reduced : {train_reduced_adata.X.shape}")
    print(f"[INFO] test_reduced  : {test_reduced_adata.X.shape}")
    print(f"[SAVED] {os.path.join(out_dir, f'train_s_{tag}.npy')}")
    print(f"[SAVED] {os.path.join(out_dir, f'train_reduced_{tag}.h5ad')}")
    print(f"[SAVED] {os.path.join(out_dir, f'test_reduced_{tag}.h5ad')}")
    print("[INFO][SDAN] done.")

# ------------------------- sciRED -------------------------
def run_scired():
    print("[INFO] Running sciRED backend (raw counts, no normalization)...")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SkPipe
    from sciRED import glm as sc_glm
    from sciRED import rotations as rot

    data, T, C = load_and_label_data()
    train_data, _, test_data = split_by_donor(data, T, C)
    out_dir = os.path.join(d, "output_comparison", "sciRED")
    os.makedirs(out_dir, exist_ok=True)
    gene_list = get_de_gene_list(
        train_data,
        train_data.obs["cell_type"].cat.categories.values,
        tag=args.cell_type,
        out_dir=out_dir,
    )
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

    tag = f"{args.cell_type}_sciRED"
    np.save(os.path.join(out_dir, f"train_s_{tag}.npy"), L_varimax.astype(np.float64))
    pd.DataFrame(L_varimax, index=genes,
                 columns=[f"sciRED_{i}" for i in range(L_varimax.shape[1])]) \
        .to_csv(os.path.join(out_dir, f"varimax_loading_{tag}.csv"))

    prog_names = [f"sciRED_{i}" for i in range(L_varimax.shape[1])]
    train_reduced_adata = AnnData(X=Ztr_full, obs=train_data.obs.copy(),
                                  var=pd.DataFrame(index=prog_names))
    test_reduced_adata = AnnData(X=Zte_full, obs=test_data.obs.copy(),
                                 var=pd.DataFrame(index=prog_names))
    train_reduced_adata.write(os.path.join(out_dir, f"train_reduced_{tag}.h5ad"))
    test_reduced_adata.write(os.path.join(out_dir, f"test_reduced_{tag}.h5ad"))

    ensure_individual_col(train_reduced_adata)
    ensure_individual_col(test_reduced_adata)

    # Define consistent label order
    cell_type_list = sorted(pd.unique(train_data.obs["cell_type"]))
    print(f"[INFO] Cell type list: {cell_type_list}")
    
    set_eval_labels(pos_label="Dementia", neg_label="No dementia")
    _eval_and_save(
        train_reduced_adata, test_reduced_adata,
        "sciRED",
        os.path.join(out_dir, f"results_classifiers_{tag}.csv"),
        os.path.join(out_dir, f"logreg_feature_importance_{tag}.csv"),
        sorted(pd.unique(train_data.obs["cell_type"])),
        C, T,
    )

    print(f"[INFO] train_s shape : {L_varimax.shape}")
    print(f"[INFO] train_reduced : {train_reduced_adata.X.shape}")
    print(f"[INFO] test_reduced  : {test_reduced_adata.X.shape}")
    print(f"[SAVED] {os.path.join(out_dir, f'train_s_{tag}.npy')}")
    print(f"[SAVED] {os.path.join(out_dir, f'train_reduced_{tag}.h5ad')}")
    print(f"[SAVED] {os.path.join(out_dir, f'test_reduced_{tag}.h5ad')}")
    print(f"[INFO][sciRED] done ({tag}).")

# ------------------------- scNET -------------------------
def run_scnet():
    import scNET
    from scNET.MultyGraphModel import scNET as scNET_model
    from scNET.Utils import save_obj
    from torch_geometric.data import Data
    from torch_geometric.utils import train_test_split_edges

    print("[INFO] scNET backend (SEA_AD custom graph)")

    # Load + split
    data, T, C = load_and_label_data()
    qc(data)

    train_data, val_data, test_data = split_by_donor(data, T, C)
    tag = f"{args.cell_type}_scNET75"
    out_dir = os.path.join(d, "output_comparison", "scNET")
    os.makedirs(out_dir, exist_ok=True)

    # DE genes
    gene_list = get_de_gene_list(
        train_data,
        train_data.obs["cell_type"].cat.categories.values,
        tag=args.cell_type,
        out_dir=out_dir,
    )
    print(f"[INFO] The number of DE genes: {len(gene_list)}")

    gene_list = pd.Index(gene_list.astype(str))

    for ad in [train_data, val_data, test_data]:
        ad._inplace_subset_var([g for g in gene_list if g in ad.var_names])
        ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")

    print(f"[INFO] DE genes used: {len(gene_list)}")

    # Build gene graph
    edge_index, gene_names = construct_gene_graph(gene_list.tolist())

    if isinstance(edge_index, list):
        edge_index = np.array(edge_index)

    if edge_index.ndim == 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    elif edge_index.ndim == 1:
        edge_index = edge_index.reshape(-1, 2).T

    assert edge_index.shape[0] == 2
    print(f"[INFO] Fixed edge_index shape: {edge_index.shape}")

    # Save original cells
    train_cells = train_data.obs_names.tolist()
    test_cells = test_data.obs_names.tolist()

    # Concatenate (transductive)
    train_data.obs["split"] = "train"
    test_data.obs["split"] = "test"

    obj = train_data.concatenate(test_data, index_unique=None)

    # Balanced subsampling
    train_sub = obj[obj.obs["split"] == "train"].copy()
    test_sub  = obj[obj.obs["split"] == "test"].copy()

    sc.pp.subsample(train_sub, n_obs=7500, random_state=888)
    sc.pp.subsample(test_sub,  n_obs=7500, random_state=888)

    obj = train_sub.concatenate(test_sub, index_unique=None)

    print(f"[INFO] After subsample: {obj.shape}")

    # Preprocess
    if sp.issparse(obj.X):
        obj.X = obj.X.toarray()
    obj.X = np.asarray(obj.X, dtype=np.float32)

    print("Check data's scale", obj.X.min(), obj.X.max())
    sc.pp.neighbors(obj, n_neighbors=10, n_pcs=15)

    # Align gene graph
    gene_to_idx = {g: i for i, g in enumerate(obj.var_names)}

    genes = list(gene_list)
    edges = []
    for i in range(edge_index.shape[1]):
        g1 = genes[edge_index[0, i]]
        g2 = genes[edge_index[1, i]]

        if g1 in gene_to_idx and g2 in gene_to_idx:
            edges.append([gene_to_idx[g1], gene_to_idx[g2]])

    device = scNET.main.device
    ppi_edge_index = torch.tensor(edges, dtype=torch.long).T.to(device)

    print(f"[INFO] gene graph: {ppi_edge_index.shape}")

    # Expression matrix
    node_feature = obj.X.T  # genes x cells
    print(f"[INFO] expression: {node_feature.shape}")

    # KNN graph
    knn_edge_index, highly_variable_index = scNET.main.build_knn_graph(obj)
    print(f"[INFO] knn: {knn_edge_index.shape}")

    x = torch.tensor(node_feature, dtype=torch.float32)
    x = ((x.T - x.mean(dim=1)) / (x.std(dim=1) + 1e-5)).T

    data = Data(x=x, edge_index=ppi_edge_index.cpu())
    data = train_test_split_edges(data)
    data = data.to(device)
    x = x.to(device)
    ppi_edge_index = ppi_edge_index.to(device)

    batch_size = max(1, knn_edge_index.shape[1] // max(1, args.scnet_batches))
    loader = scNET.main.mini_batch_knn(knn_edge_index, batch_size)
    model_name = tag
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
    for epoch in tqdm(range(args.scnet_epochs), desc="scNET Training", total=args.scnet_epochs):
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

    model_path = os.path.join(out_dir, f"models_{tag}.pt")
    # save BOTH model + useful info
    torch.save({
        "model_state_dict": model.state_dict(),
        "gene_list": list(gene_list),
        "args": vars(args),
    }, model_path)
    print(f"[SAVED MODEL] {model_path}")

    # SAVE embeddings 
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

    train_ad = AnnData(
        X=Ztr,
        obs=train_data.obs.loc[train_cells_sub].copy()
    )

    test_ad = AnnData(
        X=Zte,
        obs=test_data.obs.loc[test_cells_sub].copy()
    )

    ensure_individual_col(train_ad)
    ensure_individual_col(test_ad)

    train_ad.write(os.path.join(out_dir, f"train_reduced_{tag}.h5ad"))
    test_ad.write(os.path.join(out_dir, f"test_reduced_{tag}.h5ad"))

    # evaluation
    set_eval_labels(pos_label="Dementia", neg_label="No dementia")

    _eval_and_save(
        train_ad,
        test_ad,
        "scNET",
        os.path.join(out_dir, f"results_classifiers_{tag}.csv"),
        os.path.join(out_dir, f"logreg_feature_importance_{tag}.csv"),
        sorted(pd.unique(train_data.obs["cell_type"])),
        C,
        T,
    )


    print("[INFO][scNET] done.")

# ------------------------- Dispatch -------------------------
BACKENDS = {"SDAN": run_sdan,  
            "sciRED": run_scired,
            "scNET": run_scnet}

if __name__ == "__main__":
    backend = args.backend
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Choose from {list(BACKENDS.keys())}")
    BACKENDS[backend]()
