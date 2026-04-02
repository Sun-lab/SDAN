# Yost_2019_sdan_evaluation.py
import shutil
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

from SDAN.evaluation import eval_and_save, set_eval_labels


TRAIN_ROOT = Path("./SF_2018")
TEST_ROOT = Path("./Yost_2019")
INPUT = TEST_ROOT / "output"
OUTDIR = TEST_ROOT / "output_comparison" / "SDAN"


def dense_float32(X):
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return X.astype(np.float32, copy=False)


def load_train_data(cell_type: str):
    train_data = sc.read(TRAIN_ROOT / "data" / f"{cell_type}_tpm.tsv.gz").transpose()
    gene_names = pd.read_csv(TRAIN_ROOT / "data" / f"{cell_type}_gene_info.tsv", sep="\t")
    meta_cell = pd.read_csv(TRAIN_ROOT / "data" / f"{cell_type}_cell_info.tsv", sep="\t")

    train_data.var_names = gene_names["gene"].values
    train_data.obs["cell_type"] = pd.Categorical(meta_cell["response"])
    train_data.obs["sample"] = meta_cell["sample"].values
    train_data.obs["individual"] = meta_cell["sample"].values

    train_data.X = np.expm1(train_data.X.A if sp.issparse(train_data.X) else train_data.X)

    gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep="\t")
    train_data = train_data[:, ~train_data.var_names.isin(gene_mito["hgnc_symbol"])]
    prop_nonzero = (train_data.X != 0).sum(axis=0) / train_data.shape[0]
    train_data = train_data[:, prop_nonzero > 0.02]

    sc.pp.normalize_total(train_data, target_sum=1e4)
    sc.pp.log1p(train_data)
    return train_data


def load_test_individual_sets():
    meta_ind_test = pd.read_excel(
        TEST_ROOT / "data" / "41591_2019_522_MOESM2_ESM.xlsx",
        sheet_name="SuppTable1",
        skiprows=3,
        nrows=15,
    )
    meta_ind_test["Response"].replace({"Yes (CR)": "Yes"}, inplace=True)
    ind_y = meta_ind_test.loc[meta_ind_test["Response"] == "Yes", "Patient"]
    ind_n = meta_ind_test.loc[meta_ind_test["Response"] == "No", "Patient"]
    return ind_n, ind_y


def load_test_data():
    test_data = sc.read(TEST_ROOT / "data" / "yost_cd8_counts.tsv.gz").transpose()
    meta_cell_test = pd.read_csv(TEST_ROOT / "data" / "yost_cd8_meta.tsv", sep="\t")
    meta_ind_test = pd.read_excel(
        TEST_ROOT / "data" / "41591_2019_522_MOESM2_ESM.xlsx",
        sheet_name="SuppTable1",
        skiprows=3,
        nrows=15,
    )
    meta_ind_test["Response"] = meta_ind_test["Response"].replace({"Yes (CR)": "Yes"})
    ind_y = meta_ind_test.loc[meta_ind_test["Response"] == "Yes", "Patient"]
    ind_n = meta_ind_test.loc[meta_ind_test["Response"] == "No", "Patient"]

    test_data.obs["cell_type"] = np.select(
        [meta_cell_test.patient.isin(ind_y), meta_cell_test.patient.isin(ind_n)],
        ["Yes", "No"],
        default="Unknown",
    )
    test_data.obs["cell_type"] = test_data.obs["cell_type"].astype("category")
    test_data.obs["individual"] = meta_cell_test["patient"].astype(str).to_numpy()
    return test_data


def source_tags():
    return sorted(path.stem.removeprefix("train_s_") for path in INPUT.glob("train_s_*.npy"))


def build_new_tag(src_tag: str):
    cell_type, weight = src_tag.rsplit("_", 1)
    return cell_type, weight, f"{cell_type}_SDAN_{weight}"


def process_one(src_tag: str):
    cell_type, weight, new_tag = build_new_tag(src_tag)
    print(f"[INFO] Processing {src_tag} -> {new_tag}")

    train_data = load_train_data(cell_type)
    gene_list = pd.read_csv(INPUT / f"gene_list_{src_tag}.txt", header=None).iloc[:, 0].astype(str).tolist()
    train_data = train_data[:, gene_list].copy()

    train_s = np.load(INPUT / f"train_s_{src_tag}.npy")
    Ztr = dense_float32(train_data.X) @ train_s

    prog_names = [f"SDAN_{i}" for i in range(train_s.shape[1])]
    train_red = AnnData(Ztr, obs=train_data.obs.copy(), var=pd.DataFrame(index=prog_names))
    train_red = train_red[train_red.obs["cell_type"].isin(["NR", "R"])].copy()
    train_red.obs["cell_type"] = train_red.obs["cell_type"].replace({"NR": "No", "R": "Yes"}).astype(str)

    train_s_dst = OUTDIR / f"train_s_{new_tag}.npy"
    train_red_dst = OUTDIR / f"train_reduced_{new_tag}.h5ad"
    test_red_dst = OUTDIR / f"test_reduced_{new_tag}.h5ad"
    results_dst = OUTDIR / f"results_classifiers_{new_tag}.csv"
    coef_dst = OUTDIR / f"logreg_feature_importance_{new_tag}.csv"
    gene_list_dst = OUTDIR / f"gene_list_{new_tag}.txt"

    np.save(train_s_dst, train_s)
    pd.Series(gene_list).to_csv(gene_list_dst, index=False, header=False)
    train_red.write(train_red_dst)
    shutil.copy2(INPUT / f"test_reduced_{src_tag}.h5ad", test_red_dst)

    test_red = ad.read_h5ad(test_red_dst)
    if ("individual" not in test_red.obs.columns) or ("cell_type" not in test_red.obs.columns):
        test_data = load_test_data()
        test_individual = test_data.obs["individual"].astype(str).to_numpy()
        test_cell_type = test_data.obs["cell_type"].astype(str).to_numpy()
        if np.array_equal(test_red.obs_names.astype(str), test_data.obs_names.astype(str)):
            aligned = test_data.obs.loc[test_red.obs_names]
            test_red.obs["individual"] = aligned["individual"].astype(str).to_numpy()
            test_red.obs["cell_type"] = aligned["cell_type"].astype(str).to_numpy()
        elif test_red.n_obs == test_data.n_obs:
            test_red.obs["individual"] = test_individual
            test_red.obs["cell_type"] = test_cell_type
        else:
            raise ValueError(
                f"Cannot restore test metadata for {src_tag}: "
                f"obs_names do not match and n_obs differs "
                f"({test_red.n_obs} vs {test_data.n_obs})."
            )
    test_red = test_red[test_red.obs["cell_type"].isin(["Yes", "No"])].copy()

    ind_n, ind_y = load_test_individual_sets()
    set_eval_labels(pos_label="Yes", neg_label="No")
    results = eval_and_save(
        train_red,
        test_red,
        "SDAN",
        str(results_dst),
        str(coef_dst),
        ["No", "Yes"],
        ind_n,
        ind_y,
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
