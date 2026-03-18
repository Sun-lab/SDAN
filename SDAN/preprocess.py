import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import torch
import torch_geometric
from torch_geometric.data import Data
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from SDAN.utils import to_dense_normalized_adj, to_numpy_dense


# quality control
def qc(data):
    data.var['mt'] = data.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    data.X = to_numpy_dense(data.X)
    return data


# load data
def load_data(train_dir, test_dir):
    train_data = ad.read_h5ad(train_dir)
    test_data = ad.read_h5ad(test_dir)
    train_data.X = to_numpy_dense(train_data.X)
    test_data.X = to_numpy_dense(test_data.X)
    return train_data, test_data


# construct p-values of differential expression (DE) genes
def construct_DE_gene(data, cell_type, cell_type_list):
    def test_pval(x):
        # Support sparse and dense vectors from data.X.transpose()
        if hasattr(x, "toarray"):
            x = x.toarray().ravel()
        else:
            x = np.asarray(x).ravel()
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
    mapping = pd.Series(range(len(gene_list)), index=gene_list)
    edge_list = pd.read_csv("./Annotation/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt.gz", 
        compression="gzip", sep="\t", low_memory=False)
    edge_list = edge_list[["Official Symbol Interactor A", "Official Symbol Interactor B"]]
    edge_list = edge_list[
        (edge_list["Official Symbol Interactor A"].isin(gene_list)) &
        (edge_list["Official Symbol Interactor B"].isin(gene_list))]
    edge_list_index = torch.as_tensor(
        np.vstack([
            edge_list.iloc[:, 0].map(mapping).to_numpy(dtype=np.int64),
            edge_list.iloc[:, 1].map(mapping).to_numpy(dtype=np.int64),
        ]),
        dtype=torch.long
    )
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
    data_X = to_numpy_dense(data[:, gene_list].X)
    # Remove isolated nodes
    if remove_isolated:
        edge_list_index,_,mask = torch_geometric.utils.remove_isolated_nodes(edge_list_index, num_nodes=gene_list.size)
        data_X = data_X[:,mask]
    data_X = torch.as_tensor(data_X).t()
    data_GNN = Data(x=data_X, edge_index=edge_list_index)
    data_GNN.adj = to_dense_normalized_adj(edge_index=edge_list_index, max_num_nodes=data_GNN.num_nodes)
    return data_GNN


# labels for supervised learning
def construct_labels(data, cell_type_list):
    labels = pd.Categorical(data.obs.cell_type, categories=cell_type_list).codes
    return torch.as_tensor(labels, dtype=torch.long)
