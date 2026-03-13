# SDAN-v2 Workflow Guideline

SDAN-v2 is an extension of SDAN that implements the experimental pipelines used in this paper to compare SDAN (GNN), Spectra, and sciRED across multiple scRNA-seq datasets with consistent preprocessing and evaluation. The code is used to generate fair and reproducible comparisons across Su_2020, SEA-AD, SF_2018, and Yost_2019.

## Datasets

Su_2020 corresponds to COVID-19 severity data from Su et al. (2020). SEA-AD corresponds to dementia data from the Seattle Alzheimer’s Disease Cell Atlas (SEA-AD) consortium (Gabitto et al., 2023). For immune checkpoint inhibitor response analysis, training data were obtained from Sade-Feldman et al. (2018), and model performance was evaluated on the independent dataset from Yost et al. (2019).

## Code Structure Overview

**Tip**: To reproduce our results, please follow the Code Structure Overview below to set up the codebase.

<pre>
SDAN/
│
├── <a href="Su_2020_v2.py">Su_2020_v2.py</a>                  # Full workflow for Su 2020 dataset (GNN / sciRED)
├── <a href="SEA_AD_v2.py">SEA_AD_v2.py</a>                   # Full workflow for SEA-AD dataset (GNN / sciRED)
├── <a href="Yost_2019_v2.py">Yost_2019_v2.py</a>                # Full workflow for SF_2018 and Yost_2019 datasets (GNN / sciRED)
│ 
│   # Spectra pipeline (Su 2020: CD4 + CD8 T cells)
├── <a href="combine_cd4_cd8.py">combine_cd4_cd8.py</a>             # Step 0: combine CD4 and CD8 datasets
├── <a href="Su_2020_spectra_preprocess.py">Su_2020_spectra_preprocess.py</a>  # Step 1: preprocess CD4 and CD8 data for Spectra model
├── <a href="Su_2020_spectra_training.py">Su_2020_spectra_training.py</a>    # Step 2: train Spectra model and save latent representations
├── <a href="Su_2020_spectra_evaluation.py">Su_2020_spectra_evaluation.py</a>  # Step 3: evaluate Spectra results using classifiers
│
│   # Spectra pipeline (SEA-AD: Astro + Micro-PVM)
├── <a href="combine_Astro_Micro-PVM.py">combine_Astro_Micro-PVM.py</a>     # Step 0: combine Astro and Micro-PVM datasets
├── <a href="SEA_AD_spectra_preprocess.py">SEA_AD_spectra_preprocess.py</a>   # Step 1: preprocess Astro and Micro-PVM data for Spectra model
├── <a href="SEA_AD_spectra_training.py">SEA_AD_spectra_training.py</a>     # Step 2: train Spectra model and save latent representations
├── <a href="SEA_AD_spectra_evaluation.py">SEA_AD_spectra_evaluation.py</a>   # Step 3: evaluate Spectra results using classifiers
│
├── SDAN/
│   └── <a href="evaluation.py">evaluation.py</a>              # Shared evaluation utilities
│   └── <a href="args.py">args.py</a>                    # Defines and parses command-line arguments 
│   └── <a href="plots.ipynb">plots.ipynb</a>                # Plots
│
├── Su_2020/
│   └── <a href="Su_2020/output_v2/">output_v2/</a>                 # All output files, figures, and results
│   └── <a href="Su_2020/output_v2_4k/">output_v2_4k/</a>              # All output files, figures, and results
│
├── SEA_AD/
│   └── <a href="SEA_AD/output_v2/">output_v2/</a>                 # All output files, figures, and results
│   └── <a href="SEA_AD/output_v2_4k/">output_v2_4k/</a>              # All output files, figures, and results
│
├── <a href="sdan-spectra.yml">sdan-spectra.yml</a>               # Environment for Spectra model
├── <a href="sdan-scired.yml">sdan-scired.yml</a>                # Environment for sciRED model
└── <a href="sdan-gnn.yml">sdan-gnn.yml</a>                   # Environment for Graph Neural Network (GNN)
│
└── <a href="README.md">README.md</a>                  
</pre>

## Conda Environment Setup
Create and activate the three environments corresponding to each model backend:
```
# 1. Spectra environment
conda env create -f sdan-spectra.yml

# 2. sciRED environment
conda env create -f sdan-scired.yml

# 3. GNN environment
conda env create -f sdan-gnn.yml
```
Activate as needed:
```
conda activate sdan-spectra   # for Spectra pipeline
conda activate sdan-scired    # for sciRED pipeline
conda activate sdan-gnn       # for GNN pipeline
```

(**Tip:** Make sure to activate the correct Conda environment for each model backend, and deactivate environments when running a different backend to avoid dependency conflicts.)

## How to Run the Pipelines

Each dataset is controlled by a single driver script. The backend (GNN, Spectra, or sciRED) is specified as the first command-line argument.

### Su_2020 Dataset

CD4
```
# Run Spectra on CD4+ T cells
python Su_2020_v2.py Spectra --cell_type cd4_BL --spectra_L 40

# Run sciRED on CD4+ T cells
python Su_2020_v2.py sciRED --cell_type cd4_BL --n_comp 40

# Run GNN on CD4+ T cells
python Su_2020_v2.py GNN --cell_type cd4_BL --n_comp 40 --graph_weight 2.0

```
CD8
```
# Run Spectra on CD8+ T cells
python Su_2020_v2.py Spectra --cell_type cd8_BL --spectra_L 40

# Run sciRED on CD8+ T cells
python Su_2020_v2.py sciRED --cell_type cd8_BL --n_comp 40

# Run GNN on CD8+ T cells
python Su_2020_v2.py GNN --cell_type cd8_BL --n_comp 40 --graph_weight 2.0
```
### SEA_AD Dataset
Astro
```
# Run Spectra on Astrocytes
python SEA_AD_v2.py Spectra --cell_type Astro --spectra_L 40

# Run sciRED on Astrocytes
python SEA_AD_v2.py sciRED --cell_type Astro --n_comp 40

# Run GNN on Astrocytes
python SEA_AD_v2.py GNN --cell_type Astro --n_comp 40 --graph_weight 2.0
```
Micro-PVM
```
# Run Spectra on Microglia (Micro-PVM)
python SEA_AD_v2.py Spectra --cell_type Micro-PVM --spectra_L 40

# Run sciRED on Microglia
python SEA_AD_v2.py sciRED --cell_type Micro-PVM --n_comp 40

# Run GNN on Microglia
python SEA_AD_v2.py GNN --cell_type Micro-PVM --n_comp 40 --graph_weight 2.0
```
### SF_2018 and Yost_2019 Datasets
CD8T
```
# Run Spectra on CD8+ T
python Yost_2019_v2.py Spectra --cell_type CD8T  --spectra_L 40

# Run sciRED on CD8+ T
python Yost_2019_v2.py sciRED --cell_type CD8T  --n_comp 40

# Run GNN on CD8+ T
python Yost_2019_v2.py GNN --cell_type CD8T  --n_comp 40 --graph_weight 2.0
```

## Outputs
Each run automatically generates results in the corresponding output folder:
```
Su_2020/output_v2/
Su_2020/output_v2_4k/
SEA_AD/output_v2/
SEA_AD/output_v2_4k/
Yost_2019/output_v2/
```
