#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from compute_loading_comparison import (
    DATASET_CONFIG,
    compute_sdan_test_separation_scores,
    find_repo_root,
)


def discover_sdan_runs(sdan_output: Path) -> list[tuple[str, str]]:
    runs = []
    for train_s_path in sdan_output.glob("train_s_*.npy"):
        stem = train_s_path.stem.removeprefix("train_s_")
        if "_" not in stem:
            continue
        cell_type, weight = stem.rsplit("_", 1)
        test_reduced_path = sdan_output / f"test_reduced_{cell_type}_{weight}.h5ad"
        if test_reduced_path.exists():
            runs.append((cell_type, weight))
    return sorted(set(runs), key=lambda item: (item[0], float(item[1]), item[1]))


def build_auc_summary(repo_root: Path, dataset: str) -> pd.DataFrame:
    dataset_info = DATASET_CONFIG[dataset]
    sdan_output = repo_root / dataset_info["study_dir"] / "output"

    frames = []
    for cell_type, weight in discover_sdan_runs(sdan_output):
        frame = compute_sdan_test_separation_scores(
            repo_root=repo_root,
            sdan_output=sdan_output,
            dataset=dataset,
            dataset_info=dataset_info,
            cell_type=cell_type,
            sdan_weight=weight,
        ).rename(
            columns={
                "test_group_auroc": "cell_level_auroc",
                "individual_level_auroc": "individual_level_auroc",
            }
        )
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No SDAN runs found under {sdan_output}")

    summary = pd.concat(frames, ignore_index=True)
    summary["weight_numeric"] = pd.to_numeric(summary["sdan_weight"], errors="coerce")
    summary["program_index"] = summary["program"].str.removeprefix("SDAN_").astype(int)
    summary = summary.sort_values(
        ["cell_type", "weight_numeric", "program_index"],
        kind="stable",
    ).drop(columns=["weight_numeric", "program_index"])
    return summary.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-program SDAN AUROC summaries for all saved weights in a dataset/output folder."
    )
    parser.add_argument("dataset", choices=sorted(DATASET_CONFIG), help="Dataset name, e.g. Su_2020")
    args = parser.parse_args()

    repo_root = find_repo_root()
    dataset_output = repo_root / DATASET_CONFIG[args.dataset]["study_dir"] / "output"
    summary = build_auc_summary(repo_root, args.dataset)

    out_path = dataset_output / "sdan_program_auc_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
