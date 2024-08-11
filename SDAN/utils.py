import numpy as np
import scanpy as sc
import anndata as ad

import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import csv
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns


# edge index to normalized adjacency matrix
def to_dense_normalized_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    edge_index, _ = torch_geometric.utils.add_self_loops(edge_index=edge_index, edge_attr=edge_attr, num_nodes=max_num_nodes)
    adj = to_dense_adj(edge_index, batch, edge_attr, max_num_nodes)
    d = torch.sum(adj, 1)
    d = 1 / np.sqrt(d)
    D = torch.diagflat(d)
    adj = D @ adj @ D
    return adj


# compute leiden
def compute_leiden(data_reduced, data, n_neighbors, resolution=1):
    data_reduced = ad.AnnData(X=data_reduced)
    data_reduced.obs['cell_type'] = data.obs.cell_type.values
    data_A = sklearn.neighbors.kneighbors_graph(data_reduced.X, n_neighbors=n_neighbors, include_self=True)
    sc.tl.leiden(data_reduced, adjacency=data_A, resolution=resolution)
    return data_reduced


# plot contingency
def plot_contingency(reduced, cell_type_list, type_name, d):
    mapping_cell = {cell: i for i, cell in enumerate(cell_type_list)}
    test_contingency = sklearn.metrics.cluster.contingency_matrix(
        [mapping_cell[cell] for cell in reduced.obs.cell_type], reduced.obs.leiden)
    sns.heatmap(test_contingency, cmap="Blues", yticklabels=cell_type_list, cbar=False)
    plt.savefig(d + "figures/contingency_" + type_name + ".pdf")


# plot confusion
def plot_confusion(reduced, type_name, d):
    mapping_leiden = {cluster: reduced.obs.cell_type[reduced.obs.leiden == cluster].value_counts().idxmax()
                      for cluster in reduced.obs.leiden.cat.categories.values}
    pred_y = [mapping_leiden[cluster] for cluster in reduced.obs.leiden]
    test_y = reduced.obs.cell_type.values
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(test_y, pred_y, include_values=False, cmap="Blues")
    plt.savefig(d + "figures/confusion_" + type_name + ".pdf")


# plot assignment matrix
def plot_s(s, type_name, d):
    sns.clustermap(s.detach().numpy(), col_cluster=False, cmap="Blues")
    plt.savefig(d + "figures/heatmap_s_" + type_name + ".pdf")


def s2name(s, gene_list, type_name, d, thr=0.8):
    mapping_gene = {i: gene_list[i] for i in range(len(gene_list))}

    def s_index2name(x):
        gene_index = np.array(np.where(x > thr))
        gene_name = []
        if gene_index.shape[1] > 0:
            gene_name = np.vectorize(mapping_gene.get)(gene_index).flatten()
        return gene_name
    s_name = list(map(s_index2name, s.detach().numpy().transpose()))
    file = open(d + "output/name_s_" + type_name + ".txt", 'w+', newline='')
    write = csv.writer(file)
    write.writerows(s_name)


def compute_cluster_RI(s1, s2):
    s1_gene_list = np.array(s1.squeeze().str.cat(sep=',').split(','))
    s2_gene_list = np.array(s2.squeeze().str.cat(sep=',').split(','))
    s_gene_list = np.intersect1d(s1_gene_list, s2_gene_list)

    def find_cluster(str_find, s):
        for index in range(len(s)):
            str_all = s.squeeze()[index].split(",")
            if str_find in str_all:
                return index
        return -1
    find_cluster_s1 = partial(find_cluster, s=s1)
    s1_cluster = np.array(list(map(find_cluster_s1, s_gene_list)))
    find_cluster_s2 = partial(find_cluster, s=s2)
    s2_cluster = np.array(list(map(find_cluster_s2, s_gene_list)))
    return sklearn.metrics.adjusted_rand_score(s1_cluster, s2_cluster), len(s_gene_list), len(s_gene_list)/len(np.union1d(s1_gene_list, s2_gene_list))


def plot_loss(loss_list, type_name, d, api=False):
    if api:
        [train_loss_list, val_loss_list] = loss_list
    else:
        [train_loss_list, val_loss_list, test_loss_list] = loss_list
    plt.figure()
    plt.plot(train_loss_list, label='Train')
    plt.plot(val_loss_list, label='Val')
    if not api:
        plt.plot(test_loss_list, label='Test')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.savefig(f'{d}figures/loss_{type_name}.pdf')


def plot_auc(auc_list, type_name, d, api=False):
    if api:
        [train_auc_list, val_auc_list] = auc_list
    else:
        [train_auc_list, val_auc_list, test_auc_list] = auc_list
    plt.figure()
    plt.plot(train_auc_list, label='Train')
    plt.plot(val_auc_list, label='Val')
    if not api:
        plt.plot(test_auc_list, label='Test')
    plt.legend(loc='upper right')
    plt.title('AUC')
    plt.savefig(f'{d}figures/auc_{type_name}.pdf')


def plot_scores(scores, labels, cell_type_list, type_name, d):
    [val_score, test_score] = scores
    [val_labels, test_labels] = labels

    plt.figure()
    plt.figure(figsize=(3, 3), dpi=80)
    plt.hist([test_score[:, 1][test_labels == 1].detach().numpy(), test_score[:, 1][test_labels == 0].detach().numpy()],
             bins=10, label=[cell_type_list[1], cell_type_list[0]])
    plt.legend(loc='upper right')
    plt.title('Prediction scores on test data')
    plt.savefig(f'{d}figures/score_test_{type_name}.pdf')

    plt.figure()
    plt.figure(figsize=(3, 3), dpi=80)
    plt.hist([val_score[:, 1][val_labels == 1].detach().numpy(), val_score[:, 1][val_labels == 0].detach().numpy()],
             bins=10, label=[cell_type_list[1], cell_type_list[0]])
    plt.legend(loc='upper right')
    plt.title('Prediction scores on validation data')
    plt.savefig(f'{d}figures/score_val_{type_name}.pdf')

