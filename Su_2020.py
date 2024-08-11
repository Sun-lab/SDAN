import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import math

from SDAN.model import pipeline
from SDAN.preprocess import qc
from SDAN.train import test_model
from SDAN.utils import compute_leiden, s2name
from SDAN.args import parse_args

import sklearn

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

np.random.seed(888)
torch.manual_seed(888)

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(figsize=(8, 6), dpi=80, facecolor='white')

args = parse_args()

# Set directions
d = "./Su_2020/"
cell_type_str = args.cell_type
data_dir = f'{d}gex_{cell_type_str}.mtx.gz'
genes_dir = f'{d}gex_{cell_type_str}_genes.txt'
meta_ind_dir = f'{d}Table_S1.xlsx'
meta_cell_dir = f'{d}cell_info_{cell_type_str}.csv'

# Creat folder
os.makedirs(f'{d}figures/', exist_ok=True)
os.makedirs(f'{d}output/', exist_ok=True)

# Load data
data = sc.read(data_dir)
gene_names = pd.read_csv(genes_dir, header=None)
meta_cell = pd.read_csv(meta_cell_dir)
cell_names = meta_cell['V1']
meta_ind = pd.read_excel(meta_ind_dir, sheet_name='S1.1 Patient Clinical Data')
data.var['gene_symbols'] = gene_names.values
data.var_names = gene_names.squeeze()
data.obs['barcode'] = cell_names.values
data.obs_names = cell_names.squeeze()

# Preprocess
qc(data)

# Filter Mito genes
gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep='\t')
data = data[:, ~data.var_names.isin(gene_mito['hgnc_symbol'])]

# Keep genes expressed >=2% cells
data_nonzero_prop = (data.X != 0).sum(axis=0)
data_nonzero_prop = data_nonzero_prop/data.shape[0]
data = data[:, data_nonzero_prop > 0.02]

# Construct response labels
meta_ind['Who Ordinal Scale'].value_counts()
meta_ind['Who Ordinal Scale'].replace('1 or 2', int(2), inplace=True)
meta_ind_WOS = meta_ind.groupby(['Study Subject ID'])['Who Ordinal Scale'].max()
mild_ind = meta_ind_WOS[meta_ind_WOS <= 2].index.to_series()
severe_ind = meta_ind_WOS[meta_ind_WOS >= 5].index.to_series()

data.obs['cell_type'] = np.select([(meta_cell['individual'].isin(mild_ind)), (meta_cell['individual'].isin(severe_ind))],
                                  ['mild', 'severe'], default='moderate')
data.obs['individual'] = meta_cell['individual'].values

data.obs.cell_type.value_counts()

# Split data
test_ind = pd.concat([mild_ind.sample(n=math.floor(0.5*len(mild_ind))),
                     severe_ind.sample(n=math.floor(0.5*len(severe_ind)))])
train_ind = pd.concat([mild_ind, severe_ind]).drop(test_ind.index)

train_cell_id = meta_cell[meta_cell['individual'].isin(train_ind)]['V1']
test_cell_id = meta_cell[meta_cell['individual'].isin(test_ind)]['V1']
val_cell_id = train_cell_id.sample(n=math.floor(0.1*len(train_cell_id)))
train_cell_id = train_cell_id.drop(val_cell_id.index)
train_data = data[train_cell_id]
test_data = data[test_cell_id]
val_data = data[val_cell_id]

train_data.obs['cell_type'] = train_data.obs.cell_type.astype('category')
test_data.obs['cell_type'] = test_data.obs.cell_type.astype('category')
val_data.obs['cell_type'] = val_data.obs.cell_type.astype('category')

args.mc_weight = args.graph_weight
args.o_weight = args.graph_weight
cell_type_str = args.cell_type + "_" + str(args.graph_weight)

[train_GNN, val_GNN, test_GNN], [train_labels, val_labels, test_labels], cell_type_list, gene_list = \
    pipeline([train_data, val_data, test_data], args, d, cell_type_str)

model_dir = f'{d}output/model_{cell_type_str}.pth'
model = torch.load(model_dir)
train_s_dir = f'{d}output/train_s_{cell_type_str}.npy'
train_s = torch.tensor(np.load(train_s_dir))

test_data_reduced = test_GNN.x.t() @ train_s
val_data_reduced = val_GNN.x.t() @ train_s
test_score, _, test_auc = test_model(model, test_data_reduced, test_labels)
val_score, _, val_auc = test_model(model, val_data_reduced, val_labels)

print(f'Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')

test_reduced = compute_leiden(test_data_reduced.detach().numpy(), test_data, n_neighbors=20, resolution=1)

test_reduced.obs.cell_type = test_reduced.obs.cell_type.cat.set_categories(['severe', 'mild'])
sc.tl.tsne(test_reduced, n_pcs=0)

s2name(train_s, gene_list, cell_type_str, d)

test_score_ind = pd.Series(test_score.detach().numpy()[:,1]).groupby(by=test_data.obs.individual.values).mean()
test_type_ind = np.select([(test_score_ind.index.isin(mild_ind)), (test_score_ind.index.isin(severe_ind))],
                           ['mild', 'severe'])
mapping_type = {type_ind: i for i, type_ind in enumerate(cell_type_list)}
test_labels_ind = [mapping_type[type_ind] for type_ind in test_type_ind]
print(f'Test AUC Ind: {sklearn.metrics.roc_auc_score(test_labels_ind, test_score_ind):.4f}')

np.save(f'{d}output/score_ind_{cell_type_str}.npy',
        np.column_stack([test_labels_ind, test_score_ind]))

np.save(f'{d}output/score_{cell_type_str}.npy', np.column_stack([test_score.detach().numpy(), test_labels.detach().numpy()]))

test_reduced.obs['barcode'] = test_data.obs['barcode'].values
test_reduced.write(f'{d}output/test_reduced_{cell_type_str}.h5ad')