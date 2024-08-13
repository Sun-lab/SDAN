# SDAN (Supervised Deep Learning with Gene Annotation for Cell Classification)

## Overview

This GitHub repository contains the pipelines to analyze different datasets using SDAN.

## Installation

Download SDAN:

```
git clone https://github.com/Sun-lab/SDAN
```

## Usage

To use the SDAN, you can follow the tutorial ``tutorial.ipynb``, where we apply SDAN on Zheng_2017 data for cell classification.


## Modules

All the modules are saved in ``SDAN`` folder.
- model.py: Pipeline.
- preprocess.py: Data preprocessing.
- train.py: Training and test.
- layers.py: Layers.
- utils.py: Some useful functions.
- args.py: Hyperparameters.

## Results

### Replication

To reproduce the results, we run the python files in ``SDAN``. We can run
```
python Su_2020.py --cell_type cd4_BL
python Su_2020.py --cell_type cd8_BL
python SEA_AD.py --cell_type Astro
python SEA_AD.py --cell_type Micro-PVM
python Yost_2019.py --cell_type CD8T
```
for getting the outputs. Then we run
```
python Su_2020_plot.py
python SEA_AD_plot.py
python Yost_2019_plot.py
```
for getting all the plots. 

To get the figures for different weight, we run
```
python check_weight.py Su_2020 cd4_BL
python check_weight.py Su_2020 cd8_BL
python check_weight.py SEA_AD Astro
python check_weight.py SEA_AD Micro-PVM
python check_weight.py Yost_2019 CD8T
```

Figures and outputs for different datasets are saved in their own folders.

### Figures

Figures for different weights are saved in ``figures`` folder for each dataset. 

- The corresponding weight is added on the last.
- ``auc``, ``loss`` are the AUC curve, loss curve during training. 
- ``boxplot_score`` is the boxplot of the prediction scores on individual level. 
- ``confusion``, ``contingency``, ``tsne`` are the confusion matrix, contingency matrix, t-SNE based on unsupervised clustering on cell level. 
- ``heatmap_s`` is the heatmap for the trained assignment matrix.
- ``score_cross`` are the scatter plots for checking consistency over two different cell types. 
- ``score_test`` is the histogram of the prediction scores on cell level.

Figures across weights are saved in ``check_weight`` folder for each dataset.

- ``boxplot`` is the prediction scores on individual level.
- 

### Outputs

Outputs are saved in "output" folder for each dataset. 

- "_REP" refers to the replication, by changing the random seed for NN training. 
- "gene_list" is the list of genes selected for training by differential expression test. 
- "model" is the trained model. 
- "name_s" is the name of genes for each component, with each row is a component. 
- "score" and "score_ind" are the prediction scores on cell level and individual level, both are saved as the array. The first two columns are the scores (sum up to 1), and the third column is the true label (from the individual).
- "train_s" is the trained assignment matrix saved as the array.
      

