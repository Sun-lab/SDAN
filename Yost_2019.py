import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from SDAN.model import pipeline
from SDAN.preprocess import construct_labels
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
d = "./SF_2018/"
cell_type_str = args.cell_type
data_dir = f'{d}data/{cell_type_str}_tpm.tsv.gz'
meta_ind_dir = f'{d}data/patient_info.tsv'
meta_cell_dir = f'{d}data/{cell_type_str}_cell_info.tsv'
meta_gene_dir = f'{d}data/{cell_type_str}_gene_info.tsv'

# Creat folder
d = "./Yost_2019/"
os.makedirs(f'{d}figures/', exist_ok=True)
os.makedirs(f'{d}output/', exist_ok=True)

# Load data
data = sc.read(data_dir)
gene_names = pd.read_csv(meta_gene_dir, sep="\t")
meta_cell = pd.read_csv(meta_cell_dir, sep="\t")
meta_ind = pd.read_csv(meta_ind_dir, sep="\t")

data = data.transpose()
data.obs['cell_type'] = meta_cell['response'].values
data.obs['cell_type'] = data.obs.cell_type.astype('category')
data.obs['sample'] = meta_cell['sample'].values
data.var_names = gene_names["gene"].values

# Filter Mito genes
gene_mito = pd.read_csv("./Annotation/mito_genes.tsv", sep='\t')
data = data[:, ~data.var_names.isin(gene_mito['hgnc_symbol'])]

# Keep genes expressed >=2% cells
data_nonzero_prop = (data.X != 0).sum(axis=0)
data_nonzero_prop = data_nonzero_prop/data.shape[0]
data = data[:, data_nonzero_prop > 0.02]

meta_cell['sample'].value_counts()
meta_cell0 = meta_cell[meta_cell['sample'] != 'Post_P28']

ind_R = meta_cell0[meta_cell0['response'] == 'R'][['sample', 'patient']].drop_duplicates()
ind_NR = meta_cell0[meta_cell0['response'] == 'NR'][['sample', 'patient']].drop_duplicates()

args.mc_weight = args.graph_weight
args.o_weight = args.graph_weight
cell_type_str = args.cell_type + "_" + str(args.graph_weight)

[train_GNN, val_GNN], [train_labels, val_labels], cell_type_list, gene_list = \
    pipeline(data, args, d, cell_type_str, api=True)

model_dir = f'{d}output/model_{cell_type_str}.pth'
model = torch.load(model_dir)
train_s_dir = f'{d}output/train_s_{cell_type_str}.npy'
train_s = torch.tensor(np.load(train_s_dir))
gene_list_dir = f'{d}output/gene_list_{cell_type_str}.txt'
gene_list = pd.Index(pd.read_csv(gene_list_dir, sep=" ", header=None).squeeze())

s2name(train_s, gene_list, cell_type_str, d)


data_dir = f'{d}data/yost_cd8_counts.tsv.gz'
meta_ind_dir = f'{d}data/41591_2019_522_MOESM2_ESM.xlsx'
meta_cell_dir = f'{d}data/yost_cd8_meta.tsv'

data = sc.read(data_dir)
meta_cell = pd.read_csv(meta_cell_dir, sep="\t")
meta_ind = pd.read_excel(meta_ind_dir, sheet_name="SuppTable1", skiprows=3, nrows=15)

data = data.transpose()
sc.pp.normalize_total(data, target_sum=1e4)
sc.pp.log1p(data)

len(np.intersect1d(gene_list, data.var_names))
gene_list_mask = gene_list.isin(data.var_names)
gene_list = gene_list[gene_list_mask]

test_data_reduced = torch.tensor(data[:, gene_list].X) @ train_s[gene_list_mask, :]
ind_SCC = meta_ind[meta_ind['Tumor Type'] == "SCC"]['Patient']
ind_BCC = meta_ind[meta_ind['Tumor Type'] == "BCC"]['Patient']
meta_ind['Response'][meta_ind['Response'] == "Yes (CR)"] = "Yes"
ind_Y = meta_ind[(meta_ind['Response'] == "Yes")]['Patient']
ind_N = meta_ind[(meta_ind['Response'] == "No")]['Patient']
data.obs['cell_type'] = np.select([(meta_cell.patient.isin(ind_Y)), (meta_cell.patient.isin(ind_N))],
                           ['Yes', 'No'])
data.obs['cell_type'] = data.obs.cell_type.astype('category')
cell_type_list = ['No', 'Yes']
test_labels = construct_labels(data, cell_type_list)

test_score, _, test_auc = test_model(model, test_data_reduced, test_labels)

test_reduced = compute_leiden(test_data_reduced.detach().numpy(), data, n_neighbors=20, resolution=1)

test_reduced.obs.cell_type = test_reduced.obs.cell_type.cat.set_categories(['No', 'Yes'])
sc.tl.tsne(test_reduced, n_pcs=0)

test_score_ind = pd.Series(test_score.detach().numpy()[:,1]).groupby(by=meta_cell.patient.values).mean()
test_score_ind = test_score_ind[meta_ind["Patient"]]
test_type_ind = meta_ind["Response"].values
mapping_type = {type_ind: i for i, type_ind in enumerate(cell_type_list)}
test_labels_ind = [mapping_type[type_ind] for type_ind in test_type_ind]
print(f'Test AUC Ind: {sklearn.metrics.roc_auc_score(test_labels_ind, test_score_ind):.4f}')

np.save(f'{d}output/score_ind_{cell_type_str}.npy',
        np.column_stack([test_labels_ind, test_score_ind]))

np.save(f'{d}output/score_{cell_type_str}.npy', np.column_stack([test_score.detach().numpy(), test_labels.detach().numpy()]))

test_reduced.obs_names = data.obs_names.values
test_reduced.write(f'{d}output/test_reduced_{cell_type_str}.h5ad')