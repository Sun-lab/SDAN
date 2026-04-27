# SDAN (Supervised Deep Learning with Gene Annotation for Cell Classification)

## Overview

SDAN is a supervised deep learning framework for cell classification that incorporates gene-annotation structure into representation learning. This repository contains the core SDAN implementation, dataset-specific training pipelines, result-generation scripts, and companion workflows for method comparison and diagnostic analysis.

The repository supports:

- Main SDAN training and evaluation on multiple scRNA-seq datasets.
- Reproduction of the paper’s dataset-specific results and figures.
- Fair comparison pipelines against `Spectra`, `sciRED`, and `scNET`.
- Diagnostic and robustness analyses for gene selection, graph structure, and learned programs.

## Installation

Clone the repository:

```bash
git clone https://github.com/Sun-lab/SDAN
cd SDAN
```

## Environment Setup

This project uses a local Python virtual environment together with the dependencies in `requirements.txt`.

Create and activate the environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To activate the environment in later sessions:

```bash
source venv/bin/activate
```

On a new machine, install Python 3.10 first, then run the same steps above.

## Getting Started

The quickest way to understand the workflow is the tutorial notebook:

```text
tutorial.ipynb
```

It demonstrates SDAN on the `Zheng_2017` PBMC benchmark for cell classification.

## Repository Structure

<pre>
SDAN/
│
├── SDAN/                    # Core SDAN package
│   ├── model.py             # Main SDAN model
│   ├── preprocess.py        # Data preprocessing and gene / graph construction
│   ├── train.py             # Training and evaluation routines
│   ├── layers.py            # Custom neural network layers
│   ├── utils.py             # Utility functions for plotting and analysis
│   └── args.py              # Hyperparameter definitions
│
├── Su_2020.py               # SDAN workflow for Su et al. 2020
├── SEA_AD.py                # SDAN workflow for SEA-AD
├── Yost_2019.py             # SDAN workflow for SF_2018 / Yost_2019 transfer setting
│
├── Su_2020_plot.py          # Plotting utilities for Su_2020 outputs
├── SEA_AD_plot.py           # Plotting utilities for SEA_AD outputs
├── Yost_2019_plot.py        # Plotting utilities for Yost_2019 outputs
│
├── SDAN_Comparison/         # Comparison workflows for SDAN vs. Spectra / sciRED / scNET
├── check/                   # Diagnostic, robustness, and post hoc evaluation utilities
│
├── Annotation/              # Gene annotation, PPI, and enrichment resources
├── Zheng_2017/              # PBMC benchmark data used in tutorial and checks
├── Su_2020/                 # COVID-19 dataset files and outputs
├── SEA_AD/                  # SEA-AD dataset files and outputs
├── SF_2018/                 # Training cohort used for Yost_2019 workflow
└── Yost_2019/               # Melanoma response dataset files and outputs
</pre>

## Main SDAN Workflows

To reproduce the main SDAN results, run the dataset-specific scripts from the project root:

```bash
python Su_2020.py --cell_type cd4_BL
python Su_2020.py --cell_type cd8_BL
python SEA_AD.py --cell_type Astro
python SEA_AD.py --cell_type Micro-PVM
python Yost_2019.py --cell_type CD8T
```

These runs generate the SDAN outputs needed for downstream plotting and analysis.

To create the corresponding figures, run:

```bash
python Su_2020_plot.py
python SEA_AD_plot.py
python Yost_2019_plot.py
```

## Outputs and Figures

Each dataset stores its SDAN outputs inside its own `output/` folder.

Common output files include:

- `gene_list`: genes selected for model training.
- `model`: trained SDAN model checkpoints.
- `name_s`: gene membership for each learned component or program.
- `score` and `score_ind`: prediction scores at the cell and individual levels.
- `test_reduced`: reduced-dimensional test data.
- `train_s`: learned assignment matrix.

Figures for each dataset are stored in the corresponding `figures/` folder.

Typical figure types include:

- `auc` and `loss`: training curves.
- `boxplot_score`: individual-level prediction-score distributions.
- `confusion`, `contingency`, and `tsne`: clustering and prediction diagnostics.
- `heatmap_s`: heatmap of the learned assignment matrix.
- `score_cross`: cross-cell-type consistency plots.
- `score_test`: cell-level score histograms.

## Weight Comparison Example

To compare saved SDAN runs across graph weights, use the diagnostic script in `check/`:

```bash
python check/check_weight.py Su_2020 cd4_BL
python check/check_weight.py Su_2020 cd8_BL
python check/check_weight.py SEA_AD Astro
python check/check_weight.py SEA_AD Micro-PVM
python check/check_weight.py Yost_2019 CD8T
```

These summaries are saved to the dataset-specific `check_weight/` folder and include:

- prediction-score boxplots
- component, edge, and gene summaries
- connectivity diagnostics
- Jaccard index and adjusted Rand index heatmaps

## SDAN_Comparison

`SDAN_Comparison/` is an extension workspace for the paper’s comparison experiments. It contains the pipelines used to compare SDAN against `Spectra`, `sciRED`, and `scNET` across multiple scRNA-seq datasets with consistent preprocessing and evaluation.

Use this folder when you want to:

- reproduce the cross-method comparison experiments
- run the `Spectra`-specific preprocessing and training workflows
- regenerate comparison figures and benchmarking outputs

See [SDAN_Comparison/README.md](/Users/zxlin/Documents/GitHub/SDAN/SDAN_Comparison/README.md) for the full workflow guide.

## check

`check/` contains lightweight analysis utilities that complement the main SDAN pipelines. These scripts are mainly used for robustness checks, interpretability summaries, and post hoc comparisons of SDAN programs with external factor models and classical baselines.

Typical use cases include:

- testing sensitivity to DE-based gene pre-selection
- perturbing the annotation graph to study robustness
- comparing SDAN programs with `sciRED` and `Spectra` loadings
- summarizing per-program predictive performance
- running a classical DE plus enrichment baseline

See [check/README.md](/Users/zxlin/Documents/GitHub/SDAN/check/README.md) for command examples and expected outputs.

## Dataset Folders

The repository includes dataset-specific folders that contain raw or processed inputs, helper scripts, result files, and figure outputs:

- `Zheng_2017/`: PBMC benchmark used in the tutorial and several diagnostic checks.
- `Su_2020/`: COVID-19 severity dataset and SDAN outputs.
- `SEA_AD/`: Seattle Alzheimer’s Disease Cell Atlas subset and SDAN outputs.
- `SF_2018/`: training cohort used in the `Yost_2019` transfer setting.
- `Yost_2019/`: independent melanoma response evaluation dataset and SDAN outputs.

## Notes

- Run commands from the repository root unless a script explicitly says otherwise.
- Some downstream scripts assume that SDAN outputs already exist in the expected dataset folders.
- Comparison and diagnostic workflows rely on annotation resources under `Annotation/`, especially BIOGRID and enrichment gene-set files.
