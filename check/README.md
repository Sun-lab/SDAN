# check Workflow Guideline

The `check/` directory contains diagnostic, robustness, and post hoc evaluation utilities for SDAN. These scripts are designed to inspect how SDAN behaves under different gene-selection and graph settings, compare learned programs against external factor models, and benchmark a classical differential-expression plus enrichment baseline. In practice, this folder complements the main training pipelines by helping validate whether learned SDAN programs are stable, interpretable, and predictive.

## Scope

The scripts in `check/` are mainly used for three purposes:

- Sensitivity analysis of SDAN preprocessing and graph design.
- Comparison of SDAN programs against `sciRED` and `Spectra` loadings.
- Classical enrichment-based baseline evaluation for the same disease-label prediction tasks.

Some scripts operate on the PBMC benchmark under `Zheng_2017/`, while others consume saved outputs from `Su_2020/`, `SEA_AD/`, `Yost_2019/`, and `SDAN_Comparison/`.

## Code Structure Overview

<pre>
SDAN/
│
├── check/
│   ├── <a href="check_de.py">check_de.py</a>                    # Sensitivity analysis for DE-based gene pre-selection vs. using all genes
│   ├── <a href="check_graph.py">check_graph.py</a>                 # Robustness test for perturbing the annotation / interaction graph
│   ├── <a href="check_weight.py">check_weight.py</a>                # Summarize how SDAN graph weights change scores, program sizes, and connectivity
│   ├── <a href="compute_loading_comparison.py">compute_loading_comparison.py</a>  # Compare SDAN programs with sciRED and Spectra factors
│   ├── <a href="compute_sdan_program_auc.py">compute_sdan_program_auc.py</a>   # Per-program SDAN AUROC summary across saved weights
│   ├── <a href="de_enrichment.py">de_enrichment.py</a>              # Classical DE -> enrichment -> module-score logistic baseline
│   ├── <a href="check_de.ipynb">check_de.ipynb</a>                # Notebook companion for DE sensitivity analysis
│   ├── <a href="check_graph.ipynb">check_graph.ipynb</a>             # Notebook companion for graph robustness analysis
│   ├── <a href="check_genes.ipynb">check_genes.ipynb</a>             # Exploratory notebook for selected genes / programs
│   └── <a href="loading_comparison.ipynb">loading_comparison.ipynb</a>      # Notebook for inspecting SDAN vs. sciRED / Spectra matching results
│
└── <a href="README.md">README.md</a>
</pre>

## Inputs and Dependencies

These scripts reuse the main SDAN codebase and expect the project root layout used elsewhere in this repository.

Common dependencies include:

- `SDAN.preprocess`, `SDAN.train`, and `SDAN.utils`
- `scanpy`, `torch`, `torch_geometric`, `numpy`, `pandas`
- `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`

Several scripts also require annotation resources under `Annotation/`, especially:

- `BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt.gz`
- `mito_genes.tsv`
- MSigDB `.gmt` files used by `de_enrichment.py`

## Conda Environment Notes

There is no separate environment file inside `check/`. Use the existing repository environments depending on the task:

```bash
conda activate sdan
```

This environment is the main one for `check_de.py`, `check_graph.py`, and most downstream analyses. If you are comparing SDAN outputs against results produced by other methods, make sure those comparison outputs already exist under the expected folders before running the relevant scripts.

## How to Run the Checks

Run commands from the project root `SDAN/`.

### 1. Gene Pre-selection Sensitivity

`check_de.py` runs SDAN once for a chosen setting:

- If `--fdr` is provided, genes are selected by uncapped DE filtering at that FDR threshold.
- If `--fdr` is omitted, all genes after shared preprocessing are used.

Example:

```bash
# Use all genes
python check/check_de.py \
  --data_dir ./Zheng_2017/ \
  --cell_types cd4_t_helper naive_t

# Use DE genes with FDR <= 0.05
python check/check_de.py \
  --data_dir ./Zheng_2017/ \
  --cell_types cd4_t_helper naive_t \
  --fdr 0.05
```

Outputs are written to:

```text
Zheng_2017/output/comparison_preselection_<cell_types>.csv
Zheng_2017/output/model_<cell_types>_<setting>.pth
Zheng_2017/output/train_s_<cell_types>_<setting>.npy
Zheng_2017/figures/comparison_auc_<cell_types>.pdf
```

### 2. Annotation Graph Robustness

`check_graph.py` tests how sensitive SDAN is to perturbations of the annotation graph while keeping DE-HVG gene selection fixed.

Example:

```bash
# Original graph
python check/check_graph.py \
  --data_dir ./Zheng_2017/ \
  --cell_types cd4_t_helper naive_t \
  --perturb_frac 0.0

# Rewire 25% of graph edges
python check/check_graph.py \
  --data_dir ./Zheng_2017/ \
  --cell_types cd4_t_helper naive_t \
  --perturb_frac 0.25
```

Outputs are written to:

```text
Zheng_2017/output/annotation_robustness_<cell_types>_dehvg_top<n>_annotation_perturb*.csv
Zheng_2017/figures/annotation_robustness_<cell_types>_annotation_perturb*.pdf
```

### 3. Graph Weight Diagnostics

`check_weight.py` summarizes a set of already-saved SDAN runs across graph weights. It expects files such as `score_ind_*`, `train_s_*`, `gene_list_*`, and `name_s_*` to already exist in `<study_name>/output/`.

Example:

```bash
python check/check_weight.py Su_2020 cd4_BL
```

This generates plots in:

```text
Su_2020/check_weight/
```

including prediction-score boxplots, component counts, edge and gene summaries, connectivity distributions, clustering agreement heatmaps, and a confusion matrix between selected weights.

### 4. SDAN vs. sciRED / Spectra Loading Comparison

`compute_loading_comparison.py` compares SDAN program membership against `sciRED` and `Spectra` factor loadings using AUROC and average precision. It also appends SDAN connectivity quantiles and per-program test separation scores.

Example:

```bash
python check/compute_loading_comparison.py --dataset Su_2020 --weight 2.0
python check/compute_loading_comparison.py --dataset SEA_AD --weight 2.0
python check/compute_loading_comparison.py --dataset Yost_2019 --weight 2.0
```

Outputs are written to:

```text
SDAN_Comparison/<dataset>/output_comparison/SDAN_vs_sciRED_loading/
SDAN_Comparison/<dataset>/output_comparison/SDAN_vs_Spectra_loading/
```

with per-cell-type metrics files and dataset-level best-match summaries.

### 5. Per-program SDAN AUC Summary

`compute_sdan_program_auc.py` scans all saved SDAN weights for a dataset and summarizes, for each learned program:

- cell-level AUROC
- individual-level AUROC
- connectivity quantile

Example:

```bash
python check/compute_sdan_program_auc.py Su_2020
python check/compute_sdan_program_auc.py SEA_AD
python check/compute_sdan_program_auc.py Yost_2019
```

Output:

```text
<dataset>/output/sdan_program_auc_summary.csv
```

### 6. Classical DE + Enrichment Baseline

`de_enrichment.py` builds a non-neural baseline:

1. Select DE genes with the same SDAN gene-selection helper.
2. Perform enrichment against GO BP, Reactome, and immune signatures.
3. Remove redundant enriched terms by Jaccard overlap.
4. Score cells using the selected terms.
5. Train a logistic-regression classifier and report cell- and donor-level AUROC.

Examples:

```bash
python check/de_enrichment.py --cohort su2020 --dataset cd4_BL
python check/de_enrichment.py --cohort sea_ad --dataset Astro
python check/de_enrichment.py --cohort yost_2019 --dataset CD8T
```

Outputs are written to:

```text
Su_2020/enrichment/
SEA_AD/enrichment/
Yost_2019/enrichment/
```

and include DE gene lists, enriched terms, donor predictions, and summary statistics.

## Notebook Files

The notebooks in this folder are intended for interactive inspection of the same analyses:

- `check_de.ipynb` and `check_graph.ipynb` mirror the script-based robustness checks.
- `check_genes.ipynb` supports manual exploration of selected genes and programs.
- `loading_comparison.ipynb` is useful for inspecting SDAN-to-factor matching tables after the CSV outputs have been generated.

## Practical Notes

- Most scripts assume you are running from the repository root.
- `check_de.py` and `check_graph.py` are designed for the `Zheng_2017/` PBMC benchmark and expect `sc9_train.h5ad` and `sc9_test.h5ad`.
- `check_weight.py` does not train models; it analyzes outputs from completed SDAN runs.
- `compute_loading_comparison.py` and `compute_sdan_program_auc.py` require saved SDAN outputs as well as the relevant comparison outputs already present on disk.
- `de_enrichment.py` uses cohort-specific loading logic for `Su_2020`, `SEA_AD`, and `Yost_2019`.
