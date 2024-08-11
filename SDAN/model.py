import numpy as np
import pandas as pd
import torch
import math
from SDAN.preprocess import construct_gene_list, construct_gene_graph, construct_GNN, construct_labels
from SDAN.train import train_with_args
from SDAN.utils import plot_loss, plot_auc


def pipeline(input_data, args, d, cell_type_str="", api=False):
    if api:
        train_data = input_data
        val_data_obs = train_data.obs.sample(n=math.floor(0.1 * len(train_data)))
        val_data = train_data[val_data_obs.index]
        train_data_obs = train_data.obs.drop(val_data_obs.index)
        train_data = train_data[train_data_obs.index]
    else:
        [train_data, val_data, test_data] = input_data

    # Highly variable genes
    cell_type_list = train_data.obs.cell_type.cat.categories.values
    gene_list = construct_gene_list(train_data, cell_type_list, n_top_genes=args.n_top_genes, alpha=0.05)

    # Construct gene graph
    edge_list = construct_gene_graph(gene_list)

    # Construct GNN input
    train_GNN = construct_GNN(train_data, gene_list, edge_list)
    val_GNN = construct_GNN(val_data, gene_list, edge_list)
    if not api:
        test_GNN = construct_GNN(test_data, gene_list, edge_list)

    # Construct labels
    train_labels = construct_labels(train_data, cell_type_list)
    val_labels = construct_labels(val_data, cell_type_list)
    if not api:
        test_labels = construct_labels(test_data, cell_type_list)

    in_channels = train_GNN.num_features
    out_channels = len(cell_type_list)

    if api:
        model, train_s, loss_list, auc_list = train_with_args([train_GNN, val_GNN],
                                                              [train_labels, val_labels], in_channels,
                                                              out_channels, args, d, cell_type_str, api=True)
    else:
        model, train_s, loss_list, auc_list = train_with_args([train_GNN, val_GNN, test_GNN],
                                                              [train_labels, val_labels, test_labels], in_channels,
                                                              out_channels, args, d, cell_type_str)

    plot_loss(loss_list, cell_type_str, d, api=api)
    plot_auc(auc_list, cell_type_str, d, api=api)

    train_s_dir = f'{d}output/train_s_{cell_type_str}.npy'
    gene_list_dir = f'{d}output/gene_list_{cell_type_str}.txt'
    model_dir = f'{d}output/model_{cell_type_str}.pth'

    np.save(train_s_dir, train_s.detach().numpy())
    pd.Series(gene_list).to_csv(gene_list_dir, header=False, index=False)
    torch.save(model, model_dir)

    if api:
        return [train_GNN, val_GNN], [train_labels, val_labels], cell_type_list, gene_list
    else:
        return [train_GNN, val_GNN, test_GNN], [train_labels, val_labels, test_labels], cell_type_list, gene_list

