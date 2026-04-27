import sys
import os
import random
import numpy as np
import pandas as pd
import torch
from functools import partial
from sklearn.metrics import confusion_matrix

from SDAN.preprocess import construct_gene_graph
from SDAN.utils import compute_cluster_RI

import matplotlib.pyplot as plt
import seaborn as sns


if len(sys.argv) > 2:
    study_name = sys.argv[1]
    cell_type = sys.argv[2]
    print(f"study_name: {study_name}, cell_type: {cell_type}")
else:
    raise ValueError("This code requires at least 2 parameters")

np.random.seed(888)
torch.manual_seed(888)

os.makedirs(f'{study_name}/check_weight/', exist_ok=True)

score_0 = dict()
score_1 = dict()
num_comp = dict()
num_edge = dict()
num_degree = dict()
num_gene = dict()
cluster = dict()
quantile_edge = dict()

weight_list = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
for weight in weight_list:
    score = np.load(f'{study_name}/output/score_ind_{cell_type}_{weight}.npy')
    score_0[weight] = score[:, 1][score[:, 0] == 0]
    score_1[weight] = score[:, 1][score[:, 0] == 1]
    train_s = np.load(f'{study_name}/output/train_s_{cell_type}_{weight}.npy')
    gene_list = pd.Index(pd.read_csv(f'{study_name}/output/gene_list_{cell_type}_{weight}.txt',
                                     sep=" ", header=None).squeeze())
    num_comp[weight] = 0
    num_edge[weight] = 0
    num_gene[weight] = 0
    num_degree[weight] = np.array([])
    train_s = train_s.transpose()
    edge_list_index = construct_gene_graph(gene_list)
    edge_list_index = edge_list_index.detach().numpy()
    size_list = np.array([])
    degree_list = np.array([])
    for i in range(len(train_s)):
        gene_index = np.where(train_s[i] > 0.8)[0]
        if len(gene_index) == 0:
            continue
        edge_list = np.array([index for index in edge_list_index.transpose() if index[0].item() in gene_index and index[1].item() in gene_index])
        size_list = np.append(size_list, len(gene_index))
        degree_list = np.append(degree_list, 2 * len(edge_list) / len(gene_index))
        num_degree[weight] = np.append(num_degree[weight], len(edge_list)/len(gene_index))
        num_edge[weight] += len(edge_list)
        num_comp[weight] += 1
        num_gene[weight] += len(gene_index)
    size_sim = np.array(list(set(size_list)))
    size_sim.sort()
    degree_sim = np.zeros((len(size_sim), 1000))
    for i in range(len(size_sim)):
        size = int(size_sim[i])
        for j in range(1000):
            gene_index = random.sample(range(len(gene_list)), size)
            edge_list = [index for index in edge_list_index.transpose() if
                         index[0].item() in gene_index and index[1].item() in gene_index]
            degree_sim[i][j] = 2 * len(edge_list) / size
    degree_quantile = np.array([])
    for i in range(len(size_list)):
        degree_quantile = np.append(degree_quantile, (
                    degree_sim[np.where(size_sim == size_list[i])[0]].flatten() < degree_list[i]).mean())
    quantile_edge[weight] = degree_quantile
    name_s = pd.read_csv(f'{study_name}/output/name_s_{cell_type}_{weight}.txt', sep='\t', header=None)
    cluster[weight] = name_s


RI = np.zeros((len(weight_list), len(weight_list)))
JI = np.zeros((len(weight_list), len(weight_list)))
for i in range(len(weight_list)):
    for j in range(len(weight_list)):
        RI[i][j], _, JI[i][j] = compute_cluster_RI(cluster[weight_list[i]], cluster[weight_list[j]])

labels = list(score_0.keys())


plt.figure(figsize=(6, 4), dpi=300)
legend_handles = []
for i, key in enumerate(labels):
    data0 = score_0[key]
    data1 = score_1[key]
    position0 = i * 2 - 0.25
    position1 = i * 2 + 0.25
    bp0 = plt.boxplot(data0, positions=[position0], patch_artist=True, boxprops=dict(facecolor=plt.cm.tab10(0)), widths=0.6)
    bp1 = plt.boxplot(data1, positions=[position1], patch_artist=True, boxprops=dict(facecolor=plt.cm.tab10(1)), widths=0.6)
    legend_handles.append(bp0['boxes'][0])
    legend_handles.append(bp1['boxes'][0])
plt.xticks(range(0, len(labels) * 2, 2), labels)
plt.xlabel('Weight')
plt.ylabel('Score')
plt.title('Prediction Scores, Ind')
plt.legend(legend_handles, ['label 0', 'label 1'], loc='upper right')
plt.savefig(f'{study_name}/check_weight/{cell_type}_boxplot.pdf')

values = list(num_comp.values())
labels = list(num_comp.keys())
plt.figure(figsize=(5, 4), dpi=300)
plt.scatter(labels, values)
plt.title('Number of components')
plt.xlabel('Weight')
plt.savefig(f'{study_name}/check_weight/{cell_type}_num_comp.pdf')

values = list(num_edge.values())
labels = list(num_edge.keys())
plt.figure(figsize=(5, 4), dpi=300)
plt.scatter(labels, values)
plt.title('Number of edges')
plt.xlabel('Weight')
plt.savefig(f'{study_name}/check_weight/{cell_type}_num_edge.pdf')

values = list(num_gene.values())
labels = list(num_gene.keys())
plt.figure(figsize=(5, 4), dpi=300)
plt.scatter(labels, values)
plt.title('Number of genes')
plt.xlabel('Weight')
plt.axhline(y=len(gene_list), color='red', linestyle='--')
plt.savefig(f'{study_name}/check_weight/{cell_type}_num_gene.pdf')

labels = list(num_degree.keys())
values = list(num_degree.values())
plt.figure(figsize=(8, 6), dpi=300)
plt.violinplot(values, showmeans=False, showmedians=True)
for i, label in enumerate(labels):
    x = [i] * len(num_degree[label])
    sns.stripplot(x=[i] * len(quantile_edge[label]), y=num_degree[label], color='black', size=3, alpha=0.7, jitter=True)
plt.title('Average degree for each component')
plt.xlabel('Weight')
plt.xticks([0, 1, 2, 3, 4, 5, 6], labels)
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(f'{study_name}/check_weight/{cell_type}_num_degree.pdf')

plt.figure(figsize=(5, 4), dpi=300)
plt.imshow(RI, cmap='viridis', interpolation='nearest')
plt.colorbar()
for i in range(len(RI)):
    for j in range(len(RI[i])):
        plt.text(j, i, f'{RI[i, j]:.2f}', ha='center', va='center', color='white', fontsize=12)
plt.xticks(range(len(weight_list)), weight_list)
plt.yticks(range(len(weight_list)), weight_list)
plt.xlabel('Weight')
plt.ylabel('Weight')
plt.title('Adjusted Rand index')
plt.savefig(f'{study_name}/check_weight/{cell_type}_RI.pdf')

plt.figure(figsize=(5, 4), dpi=300)
plt.imshow(JI, cmap='viridis', interpolation='nearest')
plt.colorbar()
for i in range(len(JI)):
    for j in range(len(JI[i])):
        plt.text(j, i, f'{JI[i, j]:.2f}', ha='center', va='center', color='white', fontsize=12)
plt.xticks(range(len(weight_list)), weight_list)
plt.yticks(range(len(weight_list)), weight_list)
plt.xlabel('Weight')
plt.ylabel('Weight')
plt.title('Jaccard Index')
plt.savefig(f'{study_name}/check_weight/{cell_type}_JI.pdf')


s1 = cluster[5.0]
s2 = cluster[10.0]
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
conf_matrix = confusion_matrix(s1_cluster, s2_cluster)
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('5.0')
plt.ylabel('10.0')
plt.title('Confusion Matrix')
plt.savefig(f'{study_name}/check_weight/{cell_type}_confusion.pdf')

labels = list(quantile_edge.keys())
values = list(quantile_edge.values())
plt.figure(figsize=(5, 4), dpi=300)
plt.violinplot(values, positions=range(len(labels)), showmeans=False, showmedians=True)
for i, label in enumerate(labels):
    sns.stripplot(x=[i] * len(quantile_edge[label]), y=quantile_edge[label], color='black', size=3, alpha=0.7, jitter=True)
plt.title('Quantiles of edges for each component')
plt.xlabel('Weight')
plt.xticks(range(len(labels)), labels, rotation=45)
plt.grid(True)
plt.savefig(f'{study_name}/check_weight/{cell_type}_quantile_edge.pdf')