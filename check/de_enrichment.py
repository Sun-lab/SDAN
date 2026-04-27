import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import hypergeom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from SDAN.preprocess import construct_gene_list, qc

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classical DE -> enrichment -> module score -> logistic baseline "
            "for Su_2020 or SEA_AD without visualization."
        )
    )
    parser.add_argument("--root", type=str, default=".", help="Project root path.")
    parser.add_argument(
        "--cohort",
        type=str,
        default="su2020",
        choices=["su2020", "sea_ad", "yost_2019"],
        help="Cohort to run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cd4_BL",
        help=(
            "Cell subset name. Examples: cd4_BL/cd8_BL (su2020), "
            "Astro/Micro-PVM (sea_ad), CD8T (yost_2019)."
        ),
    )
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--de-padj-cutoff", type=float, default=0.05)
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=1000,
        help="n_top_genes passed to construct_gene_list.",
    )
    return parser.parse_args()


def parse_gmt(path: Path) -> dict[str, set[str]]:
    terms: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            term = parts[0]
            genes = set(g for g in parts[2:] if g)
            if genes:
                terms[term] = genes
    return terms


def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def median_pairwise_jaccard(term_genes: list[set[str]]) -> float:
    vals: list[float] = []
    n = len(term_genes)
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(jaccard(term_genes[i], term_genes[j]))
    return float(np.median(vals)) if vals else 0.0


def load_and_split_su2020(root: Path, dataset: str, seed: int):
    np.random.seed(seed)
    data_dir = root / "Su_2020"
    anno_dir = root / "Annotation"

    mtx_path = data_dir / f"gex_{dataset}.mtx.gz"
    gene_path = data_dir / f"gex_{dataset}_genes.txt"
    cell_path = data_dir / f"cell_info_{dataset}.csv"
    meta_path = data_dir / "Table_S1.xlsx"

    adata = sc.read(mtx_path)
    meta_cell = pd.read_csv(cell_path)
    gene_names = pd.read_csv(gene_path, header=None).squeeze().astype(str)
    meta_ind = pd.read_excel(meta_path, sheet_name="S1.1 Patient Clinical Data")

    adata.var_names = gene_names.values
    adata.obs_names = meta_cell["V1"].values
    adata.obs["barcode"] = meta_cell["V1"].values
    adata.obs["individual"] = meta_cell["individual"].values

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    mito = pd.read_csv(anno_dir / "mito_genes.tsv", sep="\t")
    adata = adata[:, ~adata.var_names.isin(mito["hgnc_symbol"])].copy()

    nonzero_prop = np.asarray((adata.X != 0).sum(axis=0)).ravel() / adata.shape[0]
    adata = adata[:, nonzero_prop > 0.02].copy()

    meta_ind = meta_ind.copy()
    meta_ind["Who Ordinal Scale"] = meta_ind["Who Ordinal Scale"].replace("1 or 2", 2)
    meta_wos = meta_ind.groupby("Study Subject ID")["Who Ordinal Scale"].max()

    mild_ind = meta_wos[meta_wos <= 2].index.to_series()
    severe_ind = meta_wos[meta_wos >= 5].index.to_series()

    adata.obs["cell_type"] = np.select(
        [
            adata.obs["individual"].isin(mild_ind),
            adata.obs["individual"].isin(severe_ind),
        ],
        ["mild", "severe"],
        default="moderate",
    )
    adata = adata[adata.obs["cell_type"].isin(["mild", "severe"])].copy()

    test_ind = pd.concat(
        [
            mild_ind.sample(n=math.floor(0.5 * len(mild_ind))),
            severe_ind.sample(n=math.floor(0.5 * len(severe_ind))),
        ]
    )
    train_ind = pd.concat([mild_ind, severe_ind]).drop(test_ind.index)

    train_adata = adata[adata.obs["individual"].isin(train_ind)].copy()
    test_adata = adata[adata.obs["individual"].isin(test_ind)].copy()

    return {
        "base_dir": data_dir,
        "train_adata": train_adata,
        "test_adata": test_adata,
        "cell_type_list": ["mild", "severe"],
        "pos_label_train": "severe",
        "pos_label_test": "severe",
        "donor_col": "individual",
    }


def load_and_split_sea_ad(root: Path, dataset: str, seed: int):
    np.random.seed(seed)
    data_dir = root / "SEA_AD"
    anno_dir = root / "Annotation"

    h5ad_path = data_dir / "data" / f"{dataset}.h5ad"
    donor_path = data_dir / "data" / "sea-ad_cohort_donor_metadata_082222.xlsx"
    if not donor_path.exists():
        # fallback to available file in this repo snapshot
        donor_path = data_dir / "data" / "sea-ad_cohort_donor_metadata_020624.xlsx"

    data = sc.read(h5ad_path)
    meta_ind = pd.read_excel(donor_path)

    # same preprocess call used in SEA_AD.py
    qc(data)

    ind_t = meta_ind[meta_ind["Cognitive Status"] == "Dementia"]["Donor ID"]
    ind_c = meta_ind[meta_ind["Cognitive Status"] == "No dementia"]["Donor ID"]

    data.obs["cell_type"] = data.obs["Cognitive status"]
    data.var_names = data.var.feature_name.astype(str)

    mito = pd.read_csv(anno_dir / "mito_genes.tsv", sep="\t")
    data = data[:, ~data.var_names.isin(mito["hgnc_symbol"])].copy()

    nonzero_prop = np.asarray((data.X != 0).sum(axis=0)).ravel() / data.shape[0]
    data = data[:, nonzero_prop > 0.02].copy()

    data = data[data.obs["donor_id"].isin(pd.concat([ind_t, ind_c]))].copy()

    test_ind = pd.concat(
        [
            ind_t.sample(n=math.floor(0.5 * len(ind_t))),
            ind_c.sample(n=math.floor(0.5 * len(ind_c))),
        ]
    )
    train_ind = pd.concat([ind_t, ind_c]).drop(test_ind.index)

    train_adata = data[data.obs["donor_id"].isin(train_ind)].copy()
    test_adata = data[data.obs["donor_id"].isin(test_ind)].copy()

    # keep only binary labels used in SEA_AD.py
    keep = ["No dementia", "Dementia"]
    train_adata = train_adata[train_adata.obs["cell_type"].isin(keep)].copy()
    test_adata = test_adata[test_adata.obs["cell_type"].isin(keep)].copy()

    return {
        "base_dir": data_dir,
        "train_adata": train_adata,
        "test_adata": test_adata,
        "cell_type_list": ["No dementia", "Dementia"],
        "pos_label_train": "Dementia",
        "pos_label_test": "Dementia",
        "donor_col": "donor_id",
    }


def load_and_split_yost_2019(root: Path, dataset: str, seed: int):
    np.random.seed(seed)
    train_dir = root / "SF_2018" / "data"
    test_dir = root / "Yost_2019" / "data"
    anno_dir = root / "Annotation"

    # Train data from SF_2018 (as in Yost_2019.py)
    train_expr_path = train_dir / f"{dataset}_tpm.tsv.gz"
    train_meta_cell_path = train_dir / f"{dataset}_cell_info.tsv"
    train_meta_gene_path = train_dir / f"{dataset}_gene_info.tsv"

    train_data = sc.read(train_expr_path).transpose()
    train_meta_cell = pd.read_csv(train_meta_cell_path, sep="\t")
    train_meta_gene = pd.read_csv(train_meta_gene_path, sep="\t")

    train_data.obs["cell_type"] = train_meta_cell["response"].values
    train_data.obs["sample"] = train_meta_cell["sample"].values
    train_data.obs["donor_id"] = train_meta_cell["patient"].values
    train_data.var_names = train_meta_gene["gene"].astype(str).values

    train_data = train_data[train_data.obs["cell_type"].isin(["NR", "R"])].copy()

    gene_mito = pd.read_csv(anno_dir / "mito_genes.tsv", sep="\t")
    train_data = train_data[:, ~train_data.var_names.isin(gene_mito["hgnc_symbol"])].copy()

    train_nonzero_prop = np.asarray((train_data.X != 0).sum(axis=0)).ravel() / train_data.shape[0]
    train_data = train_data[:, train_nonzero_prop > 0.02].copy()

    # Test data from Yost_2019 (as in Yost_2019.py)
    test_expr_path = test_dir / "yost_cd8_counts.tsv.gz"
    test_meta_cell_path = test_dir / "yost_cd8_meta.tsv"
    test_meta_ind_path = test_dir / "41591_2019_522_MOESM2_ESM.xlsx"

    test_data = sc.read(test_expr_path).transpose()
    test_meta_cell = pd.read_csv(test_meta_cell_path, sep="\t")
    test_meta_ind = pd.read_excel(test_meta_ind_path, sheet_name="SuppTable1", skiprows=3, nrows=15)

    sc.pp.normalize_total(test_data, target_sum=1e4)
    sc.pp.log1p(test_data)

    test_meta_ind = test_meta_ind.copy()
    test_meta_ind["Response"] = test_meta_ind["Response"].replace({"Yes (CR)": "Yes"})
    ind_yes = test_meta_ind[test_meta_ind["Response"] == "Yes"]["Patient"]
    ind_no = test_meta_ind[test_meta_ind["Response"] == "No"]["Patient"]

    test_data.obs["cell_type"] = np.select(
        [
            test_meta_cell["patient"].isin(ind_yes),
            test_meta_cell["patient"].isin(ind_no),
        ],
        ["Yes", "No"],
        default="Unknown",
    )
    test_data.obs["donor_id"] = test_meta_cell["patient"].values
    test_data = test_data[test_data.obs["cell_type"].isin(["No", "Yes"])].copy()

    # Keep only genes present in both cohorts for scoring downstream.
    common_genes = train_data.var_names.intersection(test_data.var_names)
    train_data = train_data[:, common_genes].copy()
    test_data = test_data[:, common_genes].copy()

    return {
        "base_dir": root / "Yost_2019",
        "train_adata": train_data,
        "test_adata": test_data,
        "cell_type_list": ["NR", "R"],
        "pos_label_train": "R",
        "pos_label_test": "Yes",
        "donor_col": "donor_id",
    }


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    sc.settings.verbosity = 1

    root = Path(args.root).resolve()
    anno_dir = root / "Annotation"

    if args.cohort == "su2020":
        ctx = load_and_split_su2020(root, args.dataset, args.seed)
    elif args.cohort == "sea_ad":
        ctx = load_and_split_sea_ad(root, args.dataset, args.seed)
    else:
        ctx = load_and_split_yost_2019(root, args.dataset, args.seed)

    base_dir: Path = ctx["base_dir"]
    train_adata = ctx["train_adata"]
    test_adata = ctx["test_adata"]
    cell_type_list = ctx["cell_type_list"]
    pos_label_train = ctx["pos_label_train"]
    pos_label_test = ctx["pos_label_test"]
    donor_col = ctx["donor_col"]

    out_dir = base_dir / "enrichment"
    out_dir.mkdir(exist_ok=True)

    print("ROOT:", root)
    print("COHORT:", args.cohort)
    print("DATASET:", args.dataset)
    print("Train cells:", train_adata.n_obs, "Test cells:", test_adata.n_obs)
    print("Train donors:", train_adata.obs[donor_col].nunique(), "Test donors:", test_adata.obs[donor_col].nunique())

    train_adata.obs["cell_type"] = train_adata.obs["cell_type"].astype("category")
    train_adata.obs["cell_type"] = train_adata.obs["cell_type"].cat.set_categories(cell_type_list)

    de_genes_idx = construct_gene_list(
        data=train_adata,
        cell_type_list=cell_type_list,
        n_top_genes=args.n_top_genes,
        method="fdr_bh",
        alpha=args.de_padj_cutoff,
    )
    de_genes = set(de_genes_idx.astype(str))
    de_df = pd.DataFrame({"gene": sorted(de_genes)})

    print("DE gene list source: SDAN.preprocess.construct_gene_list")
    print("n_top_genes per class (post-FDR):", args.n_top_genes)
    print("Significant+selected DE genes (union):", len(de_genes))

    gmt_files = {
        "GO_BP": anno_dir / "c5.go.bp.v2023.2.Hs.symbols.gmt",
        "Reactome": anno_dir / "c2.cp.reactome.v2023.2.Hs.symbols.gmt",
        "Immune": anno_dir / "c7.all.v2023.2.Hs.symbols.gmt",
    }

    terms: dict[str, set[str]] = {}
    for source, path in gmt_files.items():
        for term, genes in parse_gmt(path).items():
            terms[f"{source}::{term}"] = genes

    universe = set(train_adata.var_names.astype(str))
    N = len(de_genes & universe)
    M = len(universe)

    records: list[tuple[str, int, int, float, set[str]]] = []
    for term_name, genes in terms.items():
        gs = genes & universe
        n = len(gs)
        if n < args.min_overlap:
            continue
        k = len(gs & de_genes)
        if k < args.min_overlap:
            continue
        pval = hypergeom.sf(k - 1, M, n, N)
        records.append((term_name, n, k, pval, gs))

    enrich = pd.DataFrame(records, columns=["term", "set_size", "overlap", "pval", "genes"])
    if enrich.empty:
        raise RuntimeError("No enriched terms found. Try lowering --min-overlap or --de-padj-cutoff.")

    enrich["padj"] = multipletests(enrich["pval"], method="fdr_bh")[1]
    enrich = enrich.sort_values(["padj", "pval", "overlap"], ascending=[True, True, False]).reset_index(drop=True)
    sig_enrich = enrich[enrich["padj"] < 0.05].copy()

    selected_idx: list[int] = []
    for idx, row in sig_enrich.iterrows():
        g = row["genes"]
        keep = True
        for j in selected_idx:
            if jaccard(g, sig_enrich.loc[j, "genes"]) >= args.jaccard_threshold:
                keep = False
                break
        if keep:
            selected_idx.append(idx)
        if len(selected_idx) >= args.top_k:
            break

    selected = sig_enrich.loc[selected_idx].copy().reset_index(drop=True)
    selected["n_genes_used"] = selected["genes"].apply(len)
    median_jacc = median_pairwise_jaccard(selected["genes"].tolist())

    print("Total enriched terms:", sig_enrich.shape[0])
    print("Selected non-redundant terms:", selected.shape[0])
    print("Median pairwise Jaccard among selected terms:", round(median_jacc, 4))

    selected["score_col"] = [f"term_score_{i:02d}" for i in range(selected.shape[0])]
    for _, row in selected.iterrows():
        genes = sorted(list(row["genes"]))
        score_col = row["score_col"]
        sc.tl.score_genes(train_adata, gene_list=genes, score_name=score_col, random_state=0, use_raw=False)
        sc.tl.score_genes(test_adata, gene_list=genes, score_name=score_col, random_state=0, use_raw=False)

    feature_cols = selected["score_col"].tolist()
    X_train = train_adata.obs[feature_cols].to_numpy()
    X_test = test_adata.obs[feature_cols].to_numpy()
    y_train = (train_adata.obs["cell_type"].values == pos_label_train).astype(int)
    y_test = (test_adata.obs["cell_type"].values == pos_label_test).astype(int)

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)

    test_cell_prob = clf.predict_proba(X_test)[:, 1]
    cell_auc = roc_auc_score(y_test, test_cell_prob)

    test_pred_df = pd.DataFrame(
        {
            donor_col: test_adata.obs[donor_col].values,
            "label": y_test,
            "pred_prob": test_cell_prob,
        }
    )
    donor_df = test_pred_df.groupby(donor_col, as_index=False).agg(
        donor_pred_prob=("pred_prob", "mean"),
        donor_label=("label", "max"),
        n_cells=("label", "size"),
    )
    donor_auc = roc_auc_score(donor_df["donor_label"], donor_df["donor_pred_prob"])

    print(f"Cell-level AUC (classical):  {cell_auc:.4f}")
    print(f"Donor-level AUC (classical): {donor_auc:.4f}")

    selected["term_genes"] = selected["genes"].apply(lambda gs: ";".join(sorted(gs)))
    selected["term_de_genes"] = selected["genes"].apply(lambda gs: ";".join(sorted(gs & de_genes)))
    selected["n_term_de_genes"] = selected["genes"].apply(lambda gs: len(gs & de_genes))

    selected_export = selected[
        [
            "term",
            "set_size",
            "overlap",
            "pval",
            "padj",
            "n_genes_used",
            "n_term_de_genes",
            "score_col",
            "term_genes",
            "term_de_genes",
        ]
    ].copy()
    selected_export.to_csv(out_dir / f"selected_terms_{args.dataset}.tsv", sep="\t", index=False)

    sig_export = sig_enrich[["term", "set_size", "overlap", "pval", "padj"]].copy()
    sig_export.to_csv(out_dir / f"all_enriched_terms_{args.dataset}.tsv", sep="\t", index=False)

    donor_df.to_csv(out_dir / f"donor_predictions_{args.dataset}.tsv", sep="\t", index=False)

    summary = pd.Series(
        {
            "cohort": args.cohort,
            "dataset": args.dataset,
            "cell_auc_classical": cell_auc,
            "donor_auc_classical": donor_auc,
            "n_total_enriched_terms": int(sig_enrich.shape[0]),
            "n_selected_nonredundant_terms": int(selected.shape[0]),
            "median_pairwise_jaccard_selected": float(median_jacc),
        }
    )
    summary.to_csv(out_dir / f"summary_{args.dataset}.tsv", sep="\t", header=False)

    de_df.to_csv(out_dir / f"de_genes_{args.dataset}.tsv", sep="\t", index=False)

    print("Saved:")
    print(out_dir / f"de_genes_{args.dataset}.tsv")
    print(out_dir / f"selected_terms_{args.dataset}.tsv")
    print(out_dir / f"all_enriched_terms_{args.dataset}.tsv")
    print(out_dir / f"donor_predictions_{args.dataset}.tsv")
    print(out_dir / f"summary_{args.dataset}.tsv")


if __name__ == "__main__":
    run(parse_args())
