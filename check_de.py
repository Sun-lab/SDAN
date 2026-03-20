"""
PBMC sensitivity analysis for pre-selection settings in SDAN.

Single-setting run:
- If --fdr is provided: DE genes with that FDR threshold and no per-class cap.
- If --fdr is omitted: all genes in raw input data (after shared preprocessing).
"""

import os
import math
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch_geometric
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from SDAN.preprocess import (
    qc,
    construct_DE_gene,
    construct_GNN,
    construct_labels,
)
from SDAN.train import train_with_args, test_model
from SDAN.utils import plot_loss, plot_auc

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

SEED = 888
np.random.seed(SEED)
torch.manual_seed(SEED)

sc.settings.verbosity = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SDAN with DE pre-selection by FDR (uncapped) or all genes."
    )
    parser.add_argument("--no-cuda", action="store_true", default=True)
    parser.add_argument("--n_comp", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--graph_weight", type=float, default=2.0)
    parser.add_argument("--start_patience", type=int, default=500)
    parser.add_argument("--epochs_min", type=int, default=2000)
    parser.add_argument("--data_dir", type=str, default="./Zheng_2017/",
                        help="Directory containing sc9_train.h5ad and sc9_test.h5ad")
    parser.add_argument("--cell_types", type=str, nargs="+",
                        default=["cd4_t_helper", "naive_t"])
    parser.add_argument("--fdr", type=float, default=None,
                        help="Optional FDR threshold for DE pre-selection (e.g., 0.05). "
                             "If omitted, use all genes.")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.mc_weight = args.graph_weight
    args.o_weight = args.graph_weight
    return args


def load_biogrid_edges():
    candidates = [
        "./Annotation/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt.gz",
        "./Annotation/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt",
    ]
    biogrid_path = next((p for p in candidates if os.path.exists(p)), None)
    if biogrid_path is None:
        raise FileNotFoundError(
            "Could not find BIOGRID file in ./Annotation/ (.txt.gz or .txt)."
        )
    return pd.read_csv(
        biogrid_path,
        sep="\t",
        low_memory=False,
        usecols=["Official Symbol Interactor A", "Official Symbol Interactor B"],
        dtype="string",
    )


def construct_gene_graph_cached(gene_list, biogrid_edges):
    mapping = pd.Series(range(len(gene_list)), index=gene_list)
    edge_list = biogrid_edges[
        (biogrid_edges["Official Symbol Interactor A"].isin(gene_list)) &
        (biogrid_edges["Official Symbol Interactor B"].isin(gene_list))
    ]
    edge_index = torch.as_tensor(
        np.vstack([
            edge_list.iloc[:, 0].map(mapping).to_numpy(dtype=np.int64),
            edge_list.iloc[:, 1].map(mapping).to_numpy(dtype=np.int64),
        ]),
        dtype=torch.long
    )
    edge_index = torch.unique(edge_index, dim=1)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    return edge_index


def construct_gene_list_de_uncapped(data, cell_type_list, alpha=0.10, method="fdr_bh"):
    gene_list = pd.Index([])
    for cell_type in cell_type_list:
        de_pval = construct_DE_gene(data=data, cell_type=cell_type, cell_type_list=cell_type_list)
        de_rej, _, _, _ = multipletests(de_pval, alpha=alpha, method=method)
        de_gene = data.var_names[de_rej]
        print(f"[de_uncapped] DE genes for {cell_type}: {len(de_gene)}")
        gene_list = gene_list.append(de_gene)
    gene_list = gene_list.unique()
    print(f"[de_uncapped] Total unique genes: {len(gene_list)}")
    return gene_list


def run_experiment(
    train_data, val_data, test_data,
    gene_list, cell_type_list,
    args, d, label,
    train_labels, val_labels, test_labels,
    biogrid_edges,
):
    cell_type_str = "-".join(cell_type_list)
    run_str = f"{cell_type_str}_{label}"

    print(f"\n{'='*60}")
    print(f"  Running setting: {label} ({len(gene_list)} genes)")
    print(f"{'='*60}\n")

    edge_list = construct_gene_graph_cached(gene_list, biogrid_edges)
    train_GNN = construct_GNN(train_data, gene_list, edge_list)
    val_GNN = construct_GNN(val_data, gene_list, edge_list)
    test_GNN = construct_GNN(test_data, gene_list, edge_list)

    in_channels = train_GNN.num_features
    out_channels = len(cell_type_list)

    model, train_s, loss_list, auc_list = train_with_args(
        [train_GNN, val_GNN, test_GNN],
        [train_labels, val_labels, test_labels],
        in_channels, out_channels,
        args, d, run_str,
    )

    torch.save(model, f"{d}output/model_{run_str}.pth")
    np.save(f"{d}output/train_s_{run_str}.npy", train_s.detach().cpu().numpy())
    pd.Series(gene_list).to_csv(f"{d}output/gene_list_{run_str}.txt", header=False, index=False)

    plot_loss(loss_list, run_str, d)
    plot_auc(auc_list, run_str, d)

    eval_device = train_s.device
    val_data_reduced = val_GNN.x.to(eval_device).t() @ train_s
    test_data_reduced = test_GNN.x.to(eval_device).t() @ train_s

    _, val_loss, val_auc = test_model(model, val_data_reduced, val_labels.to(eval_device))
    _, test_loss, test_auc = test_model(model, test_data_reduced, test_labels.to(eval_device))

    print(f"[{label}] Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    return {
        "label": label,
        "n_genes": int(len(gene_list)),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "val_loss": float(val_loss.item()),
        "test_loss": float(test_loss.item()),
        "loss_list": loss_list,
        "auc_list": auc_list,
    }


def plot_comparison(results, d, cell_type_str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for res in results:
        [_, val_auc_list, test_auc_list] = res["auc_list"]
        axes[0].plot(val_auc_list, label=res["label"])
        axes[1].plot(test_auc_list, label=res["label"])

    axes[0].set_title("Validation AUC")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].set_title("Test AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{d}figures/comparison_auc_{cell_type_str}.pdf")
    plt.close()


def main():
    args = parse_args()
    print("Using CPU." if not args.cuda else "Using CUDA.")

    d = args.data_dir
    cell_type_list = list(args.cell_types)
    cell_type_str = "-".join(cell_type_list)

    os.makedirs(f"{d}figures/", exist_ok=True)
    os.makedirs(f"{d}output/", exist_ok=True)

    train_data = sc.read(f"{d}sc9_train.h5ad")
    test_data = sc.read(f"{d}sc9_test.h5ad")

    qc(train_data)
    qc(test_data)

    train_data = train_data[train_data.obs.cell_type.isin(cell_type_list), :]
    test_data = test_data[test_data.obs.cell_type.isin(cell_type_list), :]

    np.random.seed(SEED)
    val_mask = np.zeros(len(train_data), dtype=bool)
    val_mask[np.random.choice(len(train_data), math.floor(0.1 * len(train_data)), replace=False)] = True
    val_data = train_data[val_mask]
    train_data = train_data[~val_mask]

    train_labels = construct_labels(train_data, cell_type_list)
    val_labels = construct_labels(val_data, cell_type_list)
    test_labels = construct_labels(test_data, cell_type_list)

    print("Loading BIOGRID interactions once...")
    biogrid_edges = load_biogrid_edges()

    if args.fdr is None:
        label = "all_genes"
        gene_list = train_data.var_names
        print(f"[all_genes] Genes selected: {len(gene_list)}")
    else:
        if not (0.0 < args.fdr <= 1.0):
            raise ValueError("--fdr must be in (0, 1].")
        label = f"de_fdr_{str(args.fdr).replace('.', 'p')}"
        print(f"Building uncapped DE gene list with FDR <= {args.fdr} ...")
        gene_list = construct_gene_list_de_uncapped(train_data.copy(), cell_type_list, alpha=args.fdr)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    results = [run_experiment(
        train_data, val_data, test_data,
        gene_list, cell_type_list,
        args, d, label=label,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        biogrid_edges=biogrid_edges,
    )]

    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("loss_list", "auc_list")}
        for r in results
    ])
    summary_path = f"{d}output/comparison_preselection_{cell_type_str}.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
