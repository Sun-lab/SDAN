# SEA_AD_sdan_evaluation.py
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


ROOT = Path("./SEA_AD")
INPUT = ROOT / "output"
OUTDIR = ROOT / "output_v2" / "SDAN"

np.random.seed(888)


def ensure_individual_col(ad_obj):
    if "individual" not in ad_obj.obs.columns:
        for col in ["donor_id", "Donor ID", "participant", "subject_id", "sample_id"]:
            if col in ad_obj.obs.columns:
                ad_obj.obs["individual"] = ad_obj.obs[col].astype(str)
                return
        ad_obj.obs["individual"] = ad_obj.obs_names.astype(str)


def load_and_label_data(cell_type: str):
    data = sc.read(ROOT / "data" / f"{cell_type}.h5ad", cache=True)
    data.var_names = data.var["feature_name"].astype(str)

    mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t").iloc[:, 0].astype(str)
    data = data[:, ~data.var_names.isin(mito)].copy()
    sc.pp.filter_genes(data, min_cells=math.ceil(0.02 * data.n_obs))

    meta = pd.read_excel(ROOT / "data" / "sea-ad_cohort_donor_metadata_020624.xlsx")
    dementia = meta.loc[meta["Cognitive Status"] == "Dementia", "Donor ID"].astype(str)
    control = meta.loc[meta["Cognitive Status"] == "No dementia", "Donor ID"].astype(str)

    donors = data.obs["donor_id"].astype(str)
    dementia = pd.Index(dementia).intersection(donors.unique())
    control = pd.Index(control).intersection(donors.unique())

    data.obs["cell_type"] = pd.Categorical(
        np.where(donors.isin(dementia), "Dementia", "No dementia"),
        categories=["Dementia", "No dementia"],
    )
    return data, dementia, control


def split_by_donor(data, dementia, control):
    np.random.seed(888)
    test_donors = pd.Index(pd.concat([
        pd.Series(dementia).sample(n=math.floor(0.5 * len(dementia))),
        pd.Series(control).sample(n=math.floor(0.5 * len(control))),
    ]).unique())
    all_donors = dementia.union(control)
    train_pool = all_donors.difference(test_donors)

    val_d_n = min(len(dementia), math.floor(0.1 * len(dementia)))
    val_c_n = min(len(control), math.floor(0.1 * len(control)))
    val_donors = pd.Index(pd.concat([
        pd.Series(list(dementia)).sample(n=val_d_n),
        pd.Series(list(control)).sample(n=val_c_n),
    ]).unique())
    train_donors = train_pool.difference(val_donors)

    donors = data.obs["donor_id"].astype(str)
    train_data = data[donors.isin(train_donors)].copy()
    test_data = data[donors.isin(test_donors)].copy()
    return train_data, test_data


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

    data, dementia, control = load_and_label_data(cell_type)
    qc(data)
    train_data, _test_data = split_by_donor(data, dementia, control)

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
        _, test_data = split_by_donor(data, dementia, control)
        test_individual = test_data.obs["donor_id"].astype(str).to_numpy()
        if np.array_equal(test_red.obs_names.astype(str), test_data.obs_names.astype(str)):
            test_red.obs["individual"] = test_data.obs.loc[test_red.obs_names, "donor_id"].astype(str).to_numpy()
        elif test_red.n_obs == test_data.n_obs:
            test_red.obs["individual"] = test_individual
        else:
            raise ValueError(
                f"Cannot restore test individuals for {src_tag}: "
                f"obs_names do not match and n_obs differs "
                f"({test_red.n_obs} vs {test_data.n_obs})."
            )
    ensure_individual_col(test_red)

    set_eval_labels(pos_label="Dementia", neg_label="No dementia")
    results = eval_and_save(
        train_red,
        test_red,
        "SDAN",
        str(results_dst),
        str(coef_dst),
        sorted(pd.unique(train_red.obs["cell_type"])),
        control,
        dementia,
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
