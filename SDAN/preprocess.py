import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import torch
import torch_geometric
from torch_geometric.data import Data
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# quality control
def qc(data):
    data.var['mt'] = data.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    data.X = data.X.toarray()
    return data


# load data
def load_data(train_dir, test_dir):
    train_data = ad.read_h5ad(train_dir)
    test_data = ad.read_h5ad(test_dir)
    train_data.X = train_data.X.toarray()
    test_data.X = test_data.X.toarray()
    return train_data, test_data


# construct p-values of differential expression (DE) genes
def construct_DE_gene(data, cell_type, cell_type_list):
    def test_pval(x):
        _, pval = mannwhitneyu(x[data.obs.cell_type == cell_type],
                               x[(data.obs.cell_type != cell_type)&(data.obs.cell_type.isin(cell_type_list))],
                               alternative='greater', method='asymptotic')
        return pval
    DE_pval = np.array(list(map(test_pval, data.X.transpose())))
    return DE_pval


# construct gene list from p-values of DE genes
def construct_gene_list(data, cell_type_list, n_top_genes, method="fdr_bh", alpha=0.05):
    gene_list = pd.Index([])
    for cell_type in cell_type_list:
        DE_pval = construct_DE_gene(data=data, cell_type=cell_type, cell_type_list=cell_type_list)
        DE_rej, _, _, _ = multipletests(DE_pval, alpha=alpha, method=method)
        DE_gene = data.var_names[DE_rej]
        print(f"The number of DE genes for {cell_type}: {len(DE_gene)}")
        data_DE = data[:, DE_gene]
        sc.pp.highly_variable_genes(data_DE, n_top_genes=n_top_genes)
        gene_list_DE = data_DE.var_names[data_DE.var.highly_variable]
        gene_list = gene_list.append(gene_list_DE)
    gene_list = gene_list.unique()
    print(f"The number of DE genes: {len(gene_list)}")
    return gene_list


# obtain undirected edge list without self loop
def construct_gene_graph(gene_list):
    mapping = {gene: i for i, gene in enumerate(gene_list)}
    edge_list = pd.read_csv("./Annotation/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt.gz", 
        compression="gzip", sep="\t", low_memory=False)
    edge_list = edge_list[["Official Symbol Interactor A", "Official Symbol Interactor B"]]
    edge_list = edge_list[
        (edge_list["Official Symbol Interactor A"].isin(gene_list)) &
        (edge_list["Official Symbol Interactor B"].isin(gene_list))]
    edge_list_index1 = [mapping[gene] for gene in edge_list.iloc[:, 0]]
    edge_list_index2 = [mapping[gene] for gene in edge_list.iloc[:, 1]]
    edge_list_index = torch.tensor([edge_list_index1, edge_list_index2], dtype=torch.long)
    edge_list_index = torch.unique(edge_list_index, dim=1)
    edge_list_index = torch_geometric.utils.to_undirected(edge_list_index)
    edge_list_index, _ = torch_geometric.utils.remove_self_loops(edge_list_index)
    return edge_list_index


# anndata to GNN
def construct_GNN(data, gene_list, edge_list_index, remove_isolated=False):
    # Degree of the nodes
    degree = torch_geometric.utils.degree(edge_list_index[0, :], num_nodes=len(gene_list))
    print(f"The proportion of non-isolated genes: {torch.count_nonzero(degree)/len(gene_list):.2f}")
    # Filter by gene list
    data_X = data[:, gene_list]
    data_X = data_X.X
    # Remove isolated nodes
    if remove_isolated:
        edge_list_index,_,mask = torch_geometric.utils.remove_isolated_nodes(edge_list_index, num_nodes=gene_list.size)
        data_X = data_X[:,mask]
    data_X = torch.from_numpy(data_X).t()
    data_GNN = Data(x=data_X, edge_index=edge_list_index)
    return data_GNN


# labels for supervised learning
def construct_labels(data, cell_type_list):
    labels = data.obs.cell_type
    mapping_cell = {cell: i for i, cell in enumerate(cell_type_list)}
    labels = [mapping_cell[cell] for cell in labels]
    labels = torch.tensor(labels)
    return labels
