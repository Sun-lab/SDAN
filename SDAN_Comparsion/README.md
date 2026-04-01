# SDAN_Comparison Workflow Guideline

SDAN_Comparison is an extension of SDAN that implements the experimental pipelines used in this paper to compare SDAN, Spectra, sciRED, and scNET across multiple scRNA-seq datasets with consistent preprocessing and evaluation. The code is used to generate fair and reproducible comparisons across Su_2020, SEA-AD, SF_2018, and Yost_2019.

## Datasets

Su_2020 corresponds to COVID-19 severity data from Su et al. (2020). SEA-AD corresponds to dementia data from the Seattle Alzheimer’s Disease Cell Atlas (SEA-AD) consortium (Gabitto et al., 2023). For immune checkpoint inhibitor response analysis, training data were obtained from Sade-Feldman et al. (2018), and model performance was evaluated on the independent dataset from Yost et al. (2019).

## Code Structure Overview

**Important**: In this repository, all comparison scripts associated with the paper are organized under the `SDAN_Comparison/` directory. This design choice ensures that the original `SDAN/` codebase remains clean and unmodified.

**Reproducibility**: To reproduce the reported results, users should place the scripts and related files into the appropriate locations as specified in the directory structure below prior to execution. The `SDAN_Comparison/` directory functions as a self-contained workspace for organizing experimental code; however, the provided scripts assume the directory layout illustrated below.

**Recommended usage**
- If you are developing or organizing the comparison experiments, keep everything inside `SDAN_Comparsion/`.
- If you want to reproduce the results exactly as the scripts expect, copy or link the files from `SDAN_Comparsion/` into the corresponding paths shown below.

<pre>
SDAN/
│
├── <a href="Su_2020_comparsion.py">Su_2020_comparsion.py</a>          # Full workflow for Su 2020 dataset (SDAN / sciRED / scNET)
├── <a href="SEA_AD_comparison.py">SEA_AD_comparison.py</a>           # Full workflow for SEA-AD dataset (SDAN / sciRED / scNET)
├── <a href="Yost_2019_comparsion.py">Yost_2019_comparsion.py</a>        # Full workflow for SF_2018 and Yost_2019 datasets (SDAN / Spectra / sciRED / scNET)
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
│   └── <a href="Su_2020/output_comparison/">output_comparison/</a>         # All output files, figures, and results
│
├── SEA_AD/
│   └── <a href="SEA_AD/output_comparison/">output_comparison/</a>         # All output files, figures, and results
│ 
├── Yost_2019/
│   └── <a href="Yost_2019/output_comparison/">output_comparison/</a>         # All output files, figures, and results
│
├── <a href="sdan-spectra.yml">sdan-spectra.yml</a>               # Environment for Spectra model
├── <a href="sdan-scired.yml">sdan-scired.yml</a>                # Environment for sciRED model
├── <a href="sdan-scnet.yml">sdan-scnet.yml</a>                 # Environment for scNET model
└── <a href="sdan.yml">sdan.yml</a>                       # Environment for Graph Neural Network (SDAN)
│
└── <a href="README.md">README.md</a>                  
</pre>

## Conda Environment Setup
Create and activate the following environments corresponding to each model backend:
```
# 1. Spectra environment
conda env create -f sdan-spectra.yml

# 2. sciRED environment
conda env create -f sdan-scired.yml

# 3. SDAN environment
conda env create -f sdan.yml

# 4. scNET environment
conda env create -f sdan-scnet.yml
```
Activate as needed:
```
conda activate sdan-spectra   # for Spectra pipeline
conda activate sdan-scired    # for sciRED pipeline
conda activate sdan           # for SDAN pipeline
conda activate sdan-scnet     # for sciNET pipeline
```

(**Tip:** Make sure to activate the correct Conda environment for each model backend, and deactivate environments when running a different backend to avoid dependency conflicts.)

## How to Run the Pipelines

Before running any pipeline, make sure that:

- You have activated the correct Conda environment for the selected backend.
- The files have been placed in the expected reproduction layout shown in the Code Structure Overview above.
- You are running the commands from the project root `SDAN/`.

In all standard workflows, the backend (SDAN, Spectra, sciRED, or scNET) is given as the first positional argument to the dataset-specific driver script.

### Su_2020 Dataset

CD4+ T cells
```
# Run sciRED on CD4+ T cells
python Su_2020_comparison.py sciRED --cell_type cd4_BL --n_comp 40

# Run SDAN on CD4+ T cells
python Su_2020_comparison.py SDAN --cell_type cd4_BL --n_comp 40 --graph_weight 2.0

# Run scNET on CD4+ T cells
python Su_2020_comparison.py scNET --cell_type cd4_BL --scnet_epochs 250 --scnet_batches 40
```
CD8+ T cells
```
# Run sciRED on CD8+ T cells

python Su_2020_comparison.py sciRED --cell_type cd8_BL --n_comp 40

# Run SDAN on CD8+ T cells
python Su_2020_comparison.py SDAN --cell_type cd8_BL --n_comp 40 --graph_weight 2.0

# Run scNET on CD8+ T cells
python Su_2020_comparison.py scNET --cell_type cd8_BL --scnet_epochs 250 --scnet_batches 40
```

Spectra for Su_2020 CD4+ T cells and CD8+ T cells is a special multi-step workflow
```
# Step 1: combine CD4 and CD8

# Step 2: preprocess each subset separately
python Su_2020_spectra_preprocess.py --cell_type cd4_BL
python Su_2020_spectra_preprocess.py --cell_type cd8_BL

# Step 3: train Spectra on the combined CD4/CD8 union
python Su_2020_spectra_training.py Spectra --cell_type cd4_cd8_BL --spectra_L 40

# Step 4: evaluate on CD4 or CD8
python Su_2020_spectra_evaluation.py Spectra --cell_type cd4_BL
python Su_2020_spectra_evaluation.py Spectra --cell_type cd8_BL
```

### SEA_AD Dataset
Astrocytes
```
# Run sciRED on Astrocytes
python SEA_AD_comparison.py sciRED --cell_type Astro --n_comp 40

# Run SDAN on Astrocytes
python SEA_AD_comparison.py SDAN --cell_type Astro --n_comp 40 --graph_weight 2.0

# Run scNET on Astrocytes
python SEA_AD_comparison.py scNET --cell_type Astro --scnet_epochs 250 --scnet_batches 40
```
Micro-PVM
```
# Run sciRED on Microglia
python SEA_AD_comparison.py sciRED --cell_type Micro-PVM --n_comp 40

# Run SDAN on Microglia
python SEA_AD_comparison.py SDAN --cell_type Micro-PVM --n_comp 40 --graph_weight 2.0

# Run scNET on Microglia
python SEA_AD_comparison.py scNET --cell_type Micro-PVM --scnet_epochs 250 --scnet_batches 40
```
Spectra for SEA_AD Astrocytes and Micro-PVM is a special multi-step workflow
```
# Step 1: combine the Astro and Micro-PVM datasets
python combine_Astro_Micro-PVM.py

# Step 2: preprocess each subset separately
python SEA_AD_spectra_preprocess.py --cell_type Astro
python SEA_AD_spectra_preprocess.py --cell_type Micro-PVM

# Step 3: train Spectra on the combined Astro/Micro-PVM union dataset
python SEA_AD_spectra_training.py Spectra --cell_type Astro_Micro-PVM --spectra_L 40

# Step 4: evaluate the trained model on Astro or Micro-PVM cells
python SEA_AD_spectra_evaluation.py Spectra --cell_type Astro
python SEA_AD_spectra_evaluation.py Spectra --cell_type Micro-PVM
```


### SF_2018 and Yost_2019 Datasets
CD8+ T
```
# Run Spectra on CD8+ T
python Yost_2019_comparison.py Spectra --cell_type CD8T  --spectra_L 40

# Run sciRED on CD8+ T
python Yost_2019_comparison.py sciRED --cell_type CD8T  --n_comp 40

# Run SDAN on CD8+ T
python Yost_2019_comparison.py SDAN --cell_type CD8T  --n_comp 40 --graph_weight 2.0

# Run scNET on CD8+ T
python Yost_2019_comparison.py scNET --cell_type CD8T --scnet_epochs 250 --scnet_batches 40
```

## Outputs
Each run automatically generates results in the corresponding output folder:
```
Su_2020/output_comparison/
SEA_AD/output_comparison/
Yost_2019/output_comparison/
```
