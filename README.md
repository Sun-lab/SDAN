# SDAN (Supervised Deep Learning with Gene Annotation for Cell Classification)

## Overview

This repository contains pipelines for analyzing various datasets using SDAN, a deep learning approach designed for cell classification with gene annotation.

## Installation

To download and install SDAN, clone the repository using the following command:

```
git clone https://github.com/Sun-lab/SDAN
```

## Usage

To use SDAN, follow the tutorial provided in ``tutorial.ipynb``, where we demonstrate the application of SDAN on the ``Zheng_2017`` dataset for cell classification.


## Modules

All modules are located in the ``SDAN`` folder:
- ``model.py``: Main SDAN pipeline.
- ``preprocess.py``: Data preprocessing.
- ``train.py``: Neural network training and model evaluation.
- ``layers.py``: Custom neural network layers.
- ``utils.py``: Some useful functions.
- ``args.py``: Hyperparameter settings.

## Results

### Reproduction

To reproduce the results, execute the following Python scripts from the main folder:
```
python Su_2020.py --cell_type cd4_BL
python Su_2020.py --cell_type cd8_BL
python SEA_AD.py --cell_type Astro
python SEA_AD.py --cell_type Micro-PVM
python Yost_2019.py --cell_type CD8T
```
These scripts will generate the necessary outputs. To create the corresponding figures, run:
```
python Su_2020_plot.py
python SEA_AD_plot.py
python Yost_2019_plot.py
```
To generate figures comparing different weights, execute:
```
python check_weight.py Su_2020 cd4_BL
python check_weight.py Su_2020 cd8_BL
python check_weight.py SEA_AD Astro
python check_weight.py SEA_AD Micro-PVM
python check_weight.py Yost_2019 CD8T
```

Figures and outputs for each dataset are saved in their respective folders.

### Figures

Figures for different weights are stored in the ``figures`` folder for each dataset.

- The corresponding weight is appended to the end of the file name.
- ``auc`` and ``loss`` represent the AUC curve and loss curve during training, respectively.
- ``boxplot_score`` is a boxplot of prediction scores at the individual level.
- ``confusion``, ``contingency`` and ``tsne`` show the confusion matrix, contingency matrix, and t-SNE visualization based on unsupervised clustering at the cell level.
- ``heatmap_s`` is a heatmap of the trained assignment matrix.
- ``score_cross`` includes scatter plots to check consistency between two different cell types (e.g., CD4 and CD8, Astro and Micro-PVM). 
- ``score_test`` is a histogram of prediction scores at the cell level.

Figures comparing weights are saved in the ``check_weight`` folder.

- ``boxplot`` displays prediction scores at the individual level.
- ``num_comp``, ``num_degree``, ``num_edge``, and ``num_gene`` represent the number of components, average degree of each component, total number of edges, and total number of genes in all components, respectively.
- ``quantile_edge`` shows the quantile of the number of edges relative to its null distribution for each component.
- ``JI`` and ``RI`` represent the Jaccard index and adjusted Rand index for the components.

### Outputs

The outputs are saved in the ``output`` folder for each dataset.

- ``gene_list`` contains the list of genes selected for training based on differential expression tests.
- ``model`` is the trained model.
- ``name_s`` lists the names of genes in each component, with each row representing a component.
- ``score`` and ``score_ind`` are the prediction scores at the cell level and individual level, respectively, both saved as arrays. The first two columns represent the scores (which sum to 1), and the third column contains the true label (from the individual).
- ``test_reduced`` is the test data after dimension reduction.
- ``train_s`` is the trained assignment matrix, saved as an array.
      

