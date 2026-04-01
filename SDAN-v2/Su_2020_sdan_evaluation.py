# Su_2020_sdan_evaluation.py
import math
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

from SDAN.evaluation import eval_and_save, set_eval_labels
from SDAN.preprocess import qc


ROOT = Path("./Su_2020")
INPUT = ROOT / "output"
OUTDIR = ROOT / "output_v2" / "SDAN"

np.random.seed(888)


def ensure_individual_col(ad_obj):
    if "individual" not in ad_obj.obs.columns:
        for col in ["individual", "donor_id", "Donor ID", "participant", "subject_id", "sample_id"]:
            if col in ad_obj.obs.columns:
                ad_obj.obs["individual"] = ad_obj.obs[col].astype(str)
                return
        ad_obj.obs["individual"] = ad_obj.obs_names.astype(str)


def load_and_label_data(cell_type: str):
    data = sc.read(ROOT / f"gex_{cell_type}.mtx.gz", cache=True)
    gene_names = pd.read_csv(ROOT / f"gex_{cell_type}_genes.txt", header=None).iloc[:, 0].astype(str).to_numpy()
    meta_cell = pd.read_csv(ROOT / f"cell_info_{cell_type}.csv")
    meta_ind = pd.read_excel(ROOT / "Table_S1.xlsx", sheet_name="S1.1 Patient Clinical Data")

    data.var["gene_symbols"] = gene_names
    data.var_names = pd.Index(gene_names)
    data.obs["barcode"] = meta_cell["V1"].astype(str).to_numpy()
    data.obs_names = pd.Index(data.obs["barcode"])

    gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")
    mito_col = "hgnc_symbol" if "hgnc_symbol" in gene_mito.columns else gene_mito.columns[0]
    data = data[:, ~data.var_names.isin(gene_mito[mito_col])]

    data_nonzero_prop = (data.X != 0).sum(axis=0) / data.shape[0]
    data = data[:, data_nonzero_prop > 0.02]

    wos = meta_ind["Who Ordinal Scale"].astype(str).str.replace("1 or 2", "2", regex=False)
    wos = pd.to_numeric(wos, errors="coerce")
    meta_ind = meta_ind.assign(WOS=wos)
    meta_ind_wos = meta_ind.groupby("Study Subject ID")["WOS"].max().dropna()
    mild_ind = meta_ind_wos[meta_ind_wos <= 2].index.to_series()
    severe_ind = meta_ind_wos[meta_ind_wos >= 5].index.to_series()

    data.obs["cell_type"] = np.select(
        [meta_cell["individual"].isin(mild_ind), meta_cell["individual"].isin(severe_ind)],
        ["mild", "severe"],
        default="moderate",
    )
    data.obs["individual"] = meta_cell["individual"].values
    data.obs["cell_type"] = data.obs["cell_type"].astype("category")
    return data, meta_cell, mild_ind, severe_ind


def split_by_individual(data, meta_cell, mild_ind, severe_ind):
    np.random.seed(888)
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
    return train_data, test_data, mild_ind, severe_ind


def dense_float32(X):
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return X.astype(np.float32, copy=False)


def source_tags():
    return sorted(path.stem.removeprefix("train_s_") for path in INPUT.glob("train_s_*.npy"))


def build_new_tag(src_tag: str):
    cell_type, weight = src_tag.rsplit("_", 1)
    return cell_type, weight, f"{cell_type}_SDAN_{weight}"


def process_one(src_tag: str):
    cell_type, weight, new_tag = build_new_tag(src_tag)
    print(f"[INFO] Processing {src_tag} -> {new_tag}")

    data, meta_cell, mild_ind, severe_ind = load_and_label_data(cell_type)
    qc(data)
    train_data, _test_data, mild_ind, severe_ind = split_by_individual(data, meta_cell, mild_ind, severe_ind)

    gene_list = pd.read_csv(INPUT / f"gene_list_{src_tag}.txt", header=None).iloc[:, 0].astype(str).tolist()
    train_data = train_data[:, gene_list].copy()

    train_s = np.load(INPUT / f"train_s_{src_tag}.npy")
    Ztr = dense_float32(train_data.X) @ train_s
    train_red = AnnData(X=Ztr, obs=train_data.obs.copy())
    ensure_individual_col(train_red)

    train_s_dst = OUTDIR / f"train_s_{new_tag}.npy"
    train_red_dst = OUTDIR / f"train_reduced_{new_tag}.h5ad"
    test_red_dst = OUTDIR / f"test_reduced_{new_tag}.h5ad"
    results_dst = OUTDIR / f"results_classifiers_{new_tag}.csv"
    coef_dst = OUTDIR / f"logreg_feature_importance_{new_tag}.csv"
    gene_list_dst = OUTDIR / f"gene_list_{cell_type}.npy"

    np.save(train_s_dst, train_s)
    np.save(gene_list_dst, np.array(gene_list, dtype=str))
    train_red.write(train_red_dst)
    shutil.copy2(INPUT / f"test_reduced_{src_tag}.h5ad", test_red_dst)

    test_red = ad.read_h5ad(test_red_dst)
    if "individual" not in test_red.obs.columns:
        _, test_data, _, _ = split_by_individual(data, meta_cell, mild_ind, severe_ind)
        test_individual = test_data.obs["individual"].astype(str).to_numpy()
        if np.array_equal(test_red.obs_names.astype(str), test_data.obs_names.astype(str)):
            test_red.obs["individual"] = test_data.obs.loc[test_red.obs_names, "individual"].astype(str).to_numpy()
        elif test_red.n_obs == test_data.n_obs:
            test_red.obs["individual"] = test_individual
        else:
            raise ValueError(
                f"Cannot restore test individuals for {src_tag}: "
                f"obs_names do not match and n_obs differs "
                f"({test_red.n_obs} vs {test_data.n_obs})."
            )
    ensure_individual_col(test_red)

    set_eval_labels(pos_label="severe", neg_label="mild")
    results = eval_and_save(
        train_red,
        test_red,
        "SDAN",
        str(results_dst),
        str(coef_dst),
        sorted(pd.unique(train_red.obs["cell_type"])),
        mild_ind,
        severe_ind,
    )

    return {
        "cell_type": cell_type,
        "weight": float(weight),
        "tag": new_tag,
        "auc_cell": float(results[0]["auc_cell"]),
        "auc_ind": float(results[0]["auc_ind"]),
        "train_reduced": train_red_dst.name,
        "test_reduced": test_red_dst.name,
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = [process_one(tag) for tag in source_tags()]
    summary = pd.DataFrame(rows).sort_values(["cell_type", "weight"]).reset_index(drop=True)
    for cell_type, df_cell in summary.groupby("cell_type", sort=False):
        out_path = OUTDIR / f"auc_summary_{cell_type}.csv"
        df_cell.reset_index(drop=True).to_csv(out_path, index=False)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
