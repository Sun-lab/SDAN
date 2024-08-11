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
d = "./SEA_AD/"
cell_type_str = args.cell_type
data_dir = f'{d}data/{cell_type_str}.h5ad'
donor_dir = f'{d}data/sea-ad_cohort_donor_metadata_082222.xlsx'

# Creat folder
os.makedirs(f'{d}figures/', exist_ok=True)
os.makedirs(f'{d}output/', exist_ok=True)

# Load data
data = sc.read(data_dir)
meta_ind = pd.read_excel(donor_dir)

# Preprocess
qc(data)

data.obs.donor_id.value_counts().sort_values(ascending=True)
data.obs.disease.value_counts()

ind_T = meta_ind[meta_ind['Cognitive Status'] == 'Dementia']['Donor ID']
ind_C = meta_ind[meta_ind['Cognitive Status'] == 'No dementia']['Donor ID']

data.obs['cell_type'] = data.obs['Cognitive status']
data.var_names = data.var.feature_name.astype(str)

# Filter Mito genes
gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep='\t')
data = data[:, ~data.var_names.isin(gene_mito['hgnc_symbol'])]

# Keep genes expressed >=2% cells
data_nonzero_prop = (data.X != 0).sum(axis=0)
data_nonzero_prop = data_nonzero_prop/data.shape[0]
data = data[:, data_nonzero_prop > 0.02]

test_ind = pd.concat([ind_T.sample(n=math.floor(0.5*len(ind_T))),
                     ind_C.sample(n=math.floor(0.5*len(ind_C)))])
train_ind = pd.concat([ind_T, ind_C]).drop(test_ind.index)

train_data = data[data.obs['donor_id'].isin(train_ind)]
test_data = data[data.obs['donor_id'].isin(test_ind)]
val_data_obs = train_data.obs.sample(n=math.floor(0.1*len(train_data)))
val_data = train_data[val_data_obs.index]
train_data_obs = train_data.obs.drop(val_data_obs.index)
train_data = train_data[train_data_obs.index]

train_data.obs['Cognitive status'].value_counts()
test_data.obs['Cognitive status'].value_counts()
val_data.obs['Cognitive status'].value_counts()

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

test_reduced.obs.cell_type = test_reduced.obs.cell_type.cat.set_categories(['Dementia', 'No dementia'])
sc.tl.tsne(test_reduced, n_pcs=0)

s2name(train_s, gene_list, cell_type_str, d)

test_score_ind = pd.Series(test_score.detach().numpy()[:,1]).groupby(by=test_data.obs.donor_id.values).mean()
test_type_ind = np.select([(test_score_ind.index.isin(ind_C)), (test_score_ind.index.isin(ind_T))],
                           ['No dementia', 'Dementia'])
mapping_type = {type_ind: i for i, type_ind in enumerate(cell_type_list)}
test_labels_ind = [mapping_type[type_ind] for type_ind in test_type_ind]
print(f'Test AUC Ind: {sklearn.metrics.roc_auc_score(test_labels_ind, test_score_ind):.4f}')

np.save(f'{d}output/score_ind_{cell_type_str}.npy',
        np.column_stack([test_labels_ind, test_score_ind]))

np.save(f'{d}output/score_{cell_type_str}.npy', np.column_stack([test_score.detach().numpy(), test_labels.detach().numpy()]))

test_reduced.obs['barcode'] = test_data.obs_names.values
test_reduced.write(f'{d}output/test_reduced_{cell_type_str}.h5ad')

for weight in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
    test_score = np.load(f'{d}output/score_{cell_type_str}_{weight}.npy')
    test_score_ind = np.load(f'{d}output/score_ind_{cell_type_str}_{weight}.npy')
    test_score_ind_names = pd.Series(test_score[:, 1]).groupby(by=test_data.obs.donor_id.values).mean().index
    np.save(f'{d}output/score_ind_{cell_type_str}_{weight}.npy',
        np.column_stack([test_score_ind, test_score_ind_names]))

