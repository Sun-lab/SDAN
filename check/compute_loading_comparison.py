#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DATASET_CONFIG = {
    "Su_2020": {
        "study_dir": "Su_2020",
        "cell_types": ["cd4_BL", "cd8_BL"],
        "spectra_gene_list": "gene_list_cd4_cd8_union.txt",
        "spectra_loading": "train_s_cd4_cd8_BL_Spectra_union.npy",
        "pos_label": "severe",
        "neg_label": "mild",
    },
    "SEA_AD": {
        "study_dir": "SEA_AD",
        "cell_types": ["Astro", "Micro-PVM"],
        "spectra_gene_list": "gene_list_Astro_Micro-PVM_union_filtered.txt",
        "spectra_loading": "train_s_Astro_Micro-PVM_union.npy",
        "pos_label": "Dementia",
        "neg_label": "No dementia",
    },
    "Yost_2019": {
        "study_dir": "Yost_2019",
        "cell_types": ["CD8T"],
        "spectra_gene_list": "gene_list_CD8T_Spectra.txt",
        "spectra_loading": "train_s_CD8T_Spectra.npy",
        "pos_label": "Yes",
        "neg_label": "No",
    },
}


def find_repo_root(start: Optional[Path] = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "SDAN_Comparison").exists():
            return path
    raise FileNotFoundError("Could not locate repo root containing SDAN_Comparison/.")


def read_program_gene_sets(path: Path) -> dict[str, set[str]]:
    programs = {}
    for idx, line in enumerate(path.read_text().strip().splitlines()):
        genes = [gene.strip() for gene in line.split(",") if gene.strip()]
        programs[f"SDAN_{idx}"] = set(genes)
    return programs


def load_gene_list_text_or_npy(path_txt: Path, path_npy: Path) -> list[str]:
    if path_txt.exists():
        return pd.read_csv(path_txt, header=None).iloc[:, 0].astype(str).tolist()
    if path_npy.exists():
        arr = np.load(path_npy, allow_pickle=True)
        return pd.Index(arr).astype(str).tolist()
    raise FileNotFoundError(f"Missing gene list. Checked: {path_txt} and {path_npy}")


def load_scired_loading_matrix(scired_output: Path, cell_type: str) -> tuple[pd.DataFrame, list[str]]:
    varimax_path = scired_output / f"varimax_loading_{cell_type}_sciRED.csv"
    if varimax_path.exists():
        scired = pd.read_csv(varimax_path)
        if "Unnamed: 0" in scired.columns:
            scired = scired.rename(columns={"Unnamed: 0": "gene"})
        elif "gene" not in scired.columns:
            scired = scired.rename(columns={scired.columns[0]: "gene"})
        scired["gene"] = scired["gene"].astype(str)
        factor_cols = [col for col in scired.columns if col != "gene"]
        return scired[["gene", *factor_cols]], factor_cols

    train_s_path = scired_output / f"train_s_{cell_type}_sciRED.npy"
    if not train_s_path.exists():
        raise FileNotFoundError(
            f"Missing sciRED loading matrix for {cell_type}. Checked {varimax_path} and {train_s_path}"
        )
    loading = np.load(train_s_path)
    genes = load_gene_list_text_or_npy(
        scired_output / f"gene_list_{cell_type}_sciRED.txt",
        scired_output / f"gene_list_{cell_type}_sciRED.npy",
    )
    factor_cols = [f"sciRED_{idx}" for idx in range(loading.shape[1])]
    scired = pd.DataFrame(loading, columns=factor_cols)
    scired.insert(0, "gene", pd.Index(genes).astype(str))
    return scired, factor_cols


def load_sdan_scired_inputs(
    sdan_output: Path,
    scired_output: Path,
    cell_type: str,
    sdan_weight: str,
):
    gene_list_path = sdan_output / f"gene_list_{cell_type}_{sdan_weight}.txt"
    program_path = sdan_output / f"name_s_{cell_type}_{sdan_weight}.txt"
    gene_universe = pd.read_csv(gene_list_path, header=None).iloc[:, 0].astype(str).tolist()
    program_gene_sets = read_program_gene_sets(program_path)
    scired, factor_cols = load_scired_loading_matrix(scired_output, cell_type)
    return gene_universe, program_gene_sets, scired, factor_cols


def compute_program_factor_metrics(
    sdan_output: Path,
    scired_output: Path,
    cell_type: str,
    sdan_weight: str,
) -> pd.DataFrame:
    gene_universe, program_gene_sets, scired, factor_cols = load_sdan_scired_inputs(
        sdan_output, scired_output, cell_type, sdan_weight
    )
    scired = scired.drop_duplicates(subset="gene").set_index("gene")
    common_genes = [gene for gene in gene_universe if gene in scired.index]
    score_matrix = scired.loc[common_genes, factor_cols].abs()

    rows = []
    for program_name, genes_in_program in program_gene_sets.items():
        y_true = np.array([gene in genes_in_program for gene in common_genes], dtype=int)
        positive_genes = int(y_true.sum())
        negative_genes = int((1 - y_true).sum())
        if positive_genes == 0 or negative_genes == 0:
            continue

        for factor_name in factor_cols:
            y_score = score_matrix[factor_name].to_numpy()
            rows.append(
                {
                    "cell_type": cell_type,
                    "sdan_weight": sdan_weight,
                    "program": program_name,
                    "factor": factor_name,
                    "n_genes_overlap": len(common_genes),
                    "n_program_genes_total": len(genes_in_program),
                    "n_program_genes_overlap": positive_genes,
                    "auroc": roc_auc_score(y_true, y_score),
                    "average_precision": average_precision_score(y_true, y_score),
                }
            )
    return pd.DataFrame(rows)


def load_sdan_spectra_inputs(
    sdan_output: Path,
    spectra_output: Path,
    dataset_info: dict,
    cell_type: str,
    sdan_weight: str,
):
    sdan_gene_list_path = sdan_output / f"gene_list_{cell_type}_{sdan_weight}.txt"
    sdan_program_path = sdan_output / f"name_s_{cell_type}_{sdan_weight}.txt"
    spectra_gene_list_path = spectra_output / dataset_info["spectra_gene_list"]
    spectra_loading_path = spectra_output / dataset_info["spectra_loading"]

    sdan_gene_universe = pd.read_csv(sdan_gene_list_path, header=None).iloc[:, 0].astype(str).tolist()
    sdan_program_gene_sets = read_program_gene_sets(sdan_program_path)

    spectra_genes = pd.read_csv(spectra_gene_list_path, header=None).iloc[:, 0].astype(str).tolist()
    spectra_loading = np.load(spectra_loading_path)
    spectra_factor_cols = [f"Spectra_{idx}" for idx in range(spectra_loading.shape[1])]
    spectra = pd.DataFrame(spectra_loading, index=spectra_genes, columns=spectra_factor_cols).reset_index()
    spectra = spectra.rename(columns={"index": "gene"})
    return sdan_gene_universe, sdan_program_gene_sets, spectra, spectra_factor_cols, spectra_genes


def compute_program_spectra_metrics(
    sdan_output: Path,
    spectra_output: Path,
    dataset_info: dict,
    cell_type: str,
    sdan_weight: str,
) -> pd.DataFrame:
    (
        sdan_gene_universe,
        program_gene_sets,
        spectra,
        factor_cols,
        spectra_genes,
    ) = load_sdan_spectra_inputs(sdan_output, spectra_output, dataset_info, cell_type, sdan_weight)
    spectra = spectra.drop_duplicates(subset="gene").set_index("gene")

    common_genes = [gene for gene in sdan_gene_universe if gene in spectra.index]
    score_matrix = spectra.loc[common_genes, factor_cols].abs()

    rows = []
    for program_name, genes_in_program in program_gene_sets.items():
        y_true = np.array([gene in genes_in_program for gene in common_genes], dtype=int)
        positive_genes = int(y_true.sum())
        negative_genes = int((1 - y_true).sum())
        if positive_genes == 0 or negative_genes == 0:
            continue

        for factor_name in factor_cols:
            y_score = score_matrix[factor_name].to_numpy()
            rows.append(
                {
                    "cell_type": cell_type,
                    "sdan_weight": sdan_weight,
                    "program": program_name,
                    "factor": factor_name,
                    "n_sdan_genes": len(sdan_gene_universe),
                    "n_spectra_genes": len(spectra_genes),
                    "n_genes_overlap": len(common_genes),
                    "n_program_genes_total": len(genes_in_program),
                    "n_program_genes_overlap": positive_genes,
                    "auroc": roc_auc_score(y_true, y_score),
                    "average_precision": average_precision_score(y_true, y_score),
                }
            )
    return pd.DataFrame(rows)


def summarize_best_matches(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    best = metrics.loc[metrics.groupby("program")[metric].idxmax()].copy()
    return best.sort_values(metric, ascending=False).reset_index(drop=True)


def construct_gene_graph_array(repo_root: Path, gene_list: list[str]) -> np.ndarray:
    mapping = pd.Series(range(len(gene_list)), index=pd.Index(gene_list))
    edge_df = pd.read_csv(
        repo_root / "Annotation" / "BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt.gz",
        compression="gzip",
        sep="\t",
        low_memory=False,
        usecols=["Official Symbol Interactor A", "Official Symbol Interactor B"],
    )
    edge_df = edge_df[
        edge_df["Official Symbol Interactor A"].isin(gene_list)
        & edge_df["Official Symbol Interactor B"].isin(gene_list)
    ].drop_duplicates()

    edge_array = np.vstack(
        [
            edge_df.iloc[:, 0].map(mapping).to_numpy(dtype=np.int64),
            edge_df.iloc[:, 1].map(mapping).to_numpy(dtype=np.int64),
        ]
    ).T
    reverse_edges = edge_array[:, ::-1]
    edge_array = np.vstack([edge_array, reverse_edges])
    edge_array = edge_array[edge_array[:, 0] != edge_array[:, 1]]
    edge_array = np.unique(edge_array, axis=0)
    return edge_array


def compute_sdan_connectivity_quantiles(
    repo_root: Path,
    sdan_output: Path,
    cell_type: str,
    sdan_weight: str,
    threshold: float,
    n_random: int,
    seed: int,
) -> pd.DataFrame:
    train_s_path = sdan_output / f"train_s_{cell_type}_{sdan_weight}.npy"
    gene_list_path = sdan_output / f"gene_list_{cell_type}_{sdan_weight}.txt"
    train_s = np.load(train_s_path)
    gene_list = pd.read_csv(gene_list_path, header=None).iloc[:, 0].astype(str).tolist()
    edge_array = construct_gene_graph_array(repo_root, gene_list)
    rng = np.random.default_rng(seed)

    rows = []
    for idx in range(train_s.shape[1]):
        gene_index = np.where(train_s[:, idx] > threshold)[0]
        if len(gene_index) == 0:
            continue

        gene_index_set = set(gene_index.tolist())
        edge_count = sum(1 for a, b in edge_array if int(a) in gene_index_set and int(b) in gene_index_set)
        avg_degree = 2 * edge_count / len(gene_index)

        random_degrees = np.zeros(n_random, dtype=float)
        for j in range(n_random):
            random_index = rng.choice(len(gene_list), size=len(gene_index), replace=False)
            random_index_set = set(random_index.tolist())
            random_edge_count = sum(
                1 for a, b in edge_array if int(a) in random_index_set and int(b) in random_index_set
            )
            random_degrees[j] = 2 * random_edge_count / len(gene_index)

        rows.append(
            {
                "cell_type": cell_type,
                "sdan_weight": sdan_weight,
                "program": f"SDAN_{idx}",
                "n_active_genes": int(len(gene_index)),
                "n_internal_edges": int(edge_count),
                "avg_degree": float(avg_degree),
                "connectivity_quantile": float((random_degrees < avg_degree).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("program").reset_index(drop=True)


def read_h5ad_obs_column(obs_group, column: str) -> np.ndarray:
    obj = obs_group[column]
    if isinstance(obj, h5py.Group):
        categories = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in obj["categories"][:]]
        codes = obj["codes"][:].astype(int)
        return np.array([categories[code] if code >= 0 else None for code in codes], dtype=object)

    values = obj[:]
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def infer_individual_ids(
    repo_root: Path,
    dataset: str,
    dataset_info: dict,
    obs_group,
    cell_type: str,
) -> np.ndarray:
    for candidate in ["individual", "donor_id", "Donor ID", "participant", "subject_id", "sample_id"]:
        if candidate in obs_group:
            return read_h5ad_obs_column(obs_group, candidate)

    if dataset == "Su_2020":
        barcode_col = "barcode" if "barcode" in obs_group else ("_index" if "_index" in obs_group else None)
        if barcode_col is not None:
            barcodes = read_h5ad_obs_column(obs_group, barcode_col)
            meta_cell_path = repo_root / dataset_info["study_dir"] / f"cell_info_{cell_type}.csv"
            meta_cell = pd.read_csv(meta_cell_path, usecols=["V1", "individual"])
            barcode_to_individual = (
                meta_cell.drop_duplicates(subset="V1")
                .assign(V1=lambda x: x["V1"].astype(str), individual=lambda x: x["individual"].astype(str))
                .set_index("V1")["individual"]
            )
            mapped = pd.Series(barcodes.astype(str)).map(barcode_to_individual)
            if mapped.isna().any():
                n_missing = int(mapped.isna().sum())
                raise KeyError(
                    f"Could not map {n_missing} cells to individuals from {meta_cell_path}. "
                    f"Checked barcode source '{barcode_col}'."
                )
            return mapped.to_numpy(dtype=object)

    if dataset == "Yost_2019" and "_index" in obs_group:
        idx_values = read_h5ad_obs_column(obs_group, "_index")
        # Expected format like: bcc.su001.post.tcell_<barcode>
        parsed = []
        for value in idx_values.astype(str):
            tokens = value.split(".")
            if len(tokens) > 2:
                parsed.append(f"{tokens[1]}.{tokens[2]}")
            elif len(tokens) > 1:
                parsed.append(tokens[1])
            else:
                parsed.append(value)
        return np.array(parsed, dtype=object)

    if dataset == "SEA_AD" and "barcode" in obs_group:
        barcodes = read_h5ad_obs_column(obs_group, "barcode")
        # Example: <cell_barcode>-<donor_or_sample_id>
        parsed = []
        for value in barcodes.astype(str):
            parsed.append(value.rsplit("-", 1)[-1] if "-" in value else value)
        return np.array(parsed, dtype=object)

    raise KeyError(
        "Missing individual-level identifier in test_reduced obs. "
        "Expected one of: individual, donor_id, Donor ID, participant, subject_id, sample_id."
    )


def normalize_labels(labels: np.ndarray, pos_label: str, neg_label: str) -> np.ndarray:
    labels_str = pd.Series(labels).astype(str).to_numpy()
    mask = np.isin(labels_str, [pos_label, neg_label])
    if mask.any():
        return labels_str, mask
    uniques = sorted(pd.unique(labels_str))
    if len(uniques) != 2:
        raise ValueError(f"Expected binary labels but found {uniques}")
    # Fallback: keep both classes, larger lexical label treated as positive.
    auto_neg, auto_pos = uniques[0], uniques[1]
    auto_mask = np.isin(labels_str, [auto_neg, auto_pos])
    return labels_str, auto_mask


def compute_sdan_test_separation_scores(
    repo_root: Path,
    sdan_output: Path,
    dataset: str,
    dataset_info: dict,
    cell_type: str,
    sdan_weight: str,
) -> pd.DataFrame:
    test_reduced_path = sdan_output / f"test_reduced_{cell_type}_{sdan_weight}.h5ad"
    with h5py.File(test_reduced_path, "r") as f:
        x = f["X"][:]
        obs_group = f["obs"]
        labels = read_h5ad_obs_column(obs_group, "cell_type")
        individuals = infer_individual_ids(repo_root, dataset, dataset_info, obs_group, cell_type)

    labels_str, mask = normalize_labels(labels, dataset_info["pos_label"], dataset_info["neg_label"])
    labels_str = labels_str[mask]
    individuals = individuals[mask]
    x = x[mask]

    y_true = (labels_str == dataset_info["pos_label"]).astype(int)
    if y_true.min() == y_true.max():
        # fallback for auto-detected binary labels
        unique_labels = sorted(pd.unique(labels_str))
        y_true = (labels_str == unique_labels[1]).astype(int)

    label_df = pd.DataFrame({"individual": individuals, "label": labels_str})
    cell_label_bin = (label_df["label"] == dataset_info["pos_label"]).astype(int)
    # Robust to mixed per-cell labels: define individual label by majority vote.
    label_by_individual = (
        pd.DataFrame({"individual": label_df["individual"], "label_bin": cell_label_bin})
        .groupby("individual", sort=True)["label_bin"]
        .mean()
    )
    label_by_individual = (label_by_individual >= 0.5).astype(int)
    score_df = pd.DataFrame(x, columns=[f"SDAN_{idx}" for idx in range(x.shape[1])]).assign(individual=individuals)
    individual_scores = score_df.groupby("individual", sort=True).mean()
    individual_labels = label_by_individual.loc[individual_scores.index].to_numpy(dtype=int)

    rows = []
    for idx in range(x.shape[1]):
        program = f"SDAN_{idx}"
        rows.append(
            {
                "cell_type": cell_type,
                "sdan_weight": sdan_weight,
                "program": program,
                "test_group_auroc": float(roc_auc_score(y_true, x[:, idx])),
                "individual_level_auroc": float(
                    roc_auc_score(individual_labels, individual_scores[program].to_numpy())
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("program").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute loading-vector comparison CSVs (SDAN vs sciRED/Spectra).")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIG))
    parser.add_argument("--weight", default="2.0", help="SDAN weight string, e.g. 2.0")
    parser.add_argument("--connectivity-threshold", type=float, default=0.8)
    parser.add_argument("--n-random", type=int, default=1000, help="Random samples per program for connectivity quantile.")
    parser.add_argument("--seed", type=int, default=888)
    args = parser.parse_args()

    repo_root = find_repo_root()
    dataset_info = DATASET_CONFIG[args.dataset]
    sdan_output = repo_root / dataset_info["study_dir"] / "output"
    scired_output = repo_root / "SDAN_Comparison" / dataset_info["study_dir"] / "output_comparison" / "sciRED"
    spectra_output = repo_root / "SDAN_Comparison" / dataset_info["study_dir"] / "output_comparison" / "Spectra"
    scired_results_dir = (
        repo_root / "SDAN_Comparison" / dataset_info["study_dir"] / "output_comparison" / "SDAN_vs_sciRED_loading"
    )
    spectra_results_dir = (
        repo_root / "SDAN_Comparison" / dataset_info["study_dir"] / "output_comparison" / "SDAN_vs_Spectra_loading"
    )
    scired_results_dir.mkdir(parents=True, exist_ok=True)
    spectra_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset={args.dataset}, weight={args.weight}")

    combined_scired_best_ap = []
    combined_scired_best_auroc = []
    combined_spectra_best_ap = []
    combined_spectra_best_auroc = []

    for cell_type in dataset_info["cell_types"]:
        print(f"[INFO] Processing cell_type={cell_type}")
        connectivity = compute_sdan_connectivity_quantiles(
            repo_root,
            sdan_output,
            cell_type,
            args.weight,
            threshold=args.connectivity_threshold,
            n_random=args.n_random,
            seed=args.seed,
        )
        test_sep = compute_sdan_test_separation_scores(
            repo_root, sdan_output, args.dataset, dataset_info, cell_type, args.weight
        )

        scired_metrics = compute_program_factor_metrics(
            sdan_output, scired_output, cell_type, args.weight
        )
        scired_metrics.to_csv(scired_results_dir / f"{cell_type}_vs_sciRED_program_factor_metrics.csv", index=False)

        scired_best_ap = summarize_best_matches(scired_metrics, "average_precision")
        scired_best_ap = scired_best_ap.merge(connectivity, on=["cell_type", "sdan_weight", "program"], how="left")
        scired_best_ap = scired_best_ap.merge(test_sep, on=["cell_type", "sdan_weight", "program"], how="left")
        combined_scired_best_ap.append(scired_best_ap)

        scired_best_auroc = summarize_best_matches(scired_metrics, "auroc")
        scired_best_auroc = scired_best_auroc.merge(
            connectivity, on=["cell_type", "sdan_weight", "program"], how="left"
        )
        scired_best_auroc = scired_best_auroc.merge(test_sep, on=["cell_type", "sdan_weight", "program"], how="left")
        combined_scired_best_auroc.append(scired_best_auroc)

        spectra_metrics = compute_program_spectra_metrics(
            sdan_output, spectra_output, dataset_info, cell_type, args.weight
        )
        spectra_metrics.to_csv(spectra_results_dir / f"{cell_type}_vs_Spectra_program_factor_metrics.csv", index=False)

        spectra_best_ap = summarize_best_matches(spectra_metrics, "average_precision")
        spectra_best_ap = spectra_best_ap.merge(connectivity, on=["cell_type", "sdan_weight", "program"], how="left")
        spectra_best_ap = spectra_best_ap.merge(test_sep, on=["cell_type", "sdan_weight", "program"], how="left")
        combined_spectra_best_ap.append(spectra_best_ap)

        spectra_best_auroc = summarize_best_matches(spectra_metrics, "auroc")
        spectra_best_auroc = spectra_best_auroc.merge(
            connectivity, on=["cell_type", "sdan_weight", "program"], how="left"
        )
        spectra_best_auroc = spectra_best_auroc.merge(test_sep, on=["cell_type", "sdan_weight", "program"], how="left")
        combined_spectra_best_auroc.append(spectra_best_auroc)

    out_scired_ap = pd.concat(combined_scired_best_ap, ignore_index=True).sort_values(
        ["cell_type", "average_precision"], ascending=[True, False]
    )
    out_scired_auroc = pd.concat(combined_scired_best_auroc, ignore_index=True).sort_values(
        ["cell_type", "auroc"], ascending=[True, False]
    )
    out_spectra_ap = pd.concat(combined_spectra_best_ap, ignore_index=True).sort_values(
        ["cell_type", "average_precision"], ascending=[True, False]
    )
    out_spectra_auroc = pd.concat(combined_spectra_best_auroc, ignore_index=True).sort_values(
        ["cell_type", "auroc"], ascending=[True, False]
    )

    out_scired_ap.to_csv(scired_results_dir / f"{args.dataset}_best_matches_by_average_precision.csv", index=False)
    out_scired_auroc.to_csv(scired_results_dir / f"{args.dataset}_best_matches_by_auroc.csv", index=False)
    out_spectra_ap.to_csv(
        spectra_results_dir / f"{args.dataset}_best_Spectra_matches_by_average_precision.csv", index=False
    )
    out_spectra_auroc.to_csv(
        spectra_results_dir / f"{args.dataset}_best_Spectra_matches_by_auroc.csv", index=False
    )

    print(f"[DONE] sciRED CSVs -> {scired_results_dir}")
    print(f"[DONE] Spectra CSVs -> {spectra_results_dir}")


if __name__ == "__main__":
    main()
