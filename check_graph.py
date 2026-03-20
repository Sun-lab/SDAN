"""
Robustness test for annotation graph in SDAN.

Single-run robustness setting for annotation graph in SDAN.

Gene selection:
- DE + HVG selection using the shared SDAN function
  (`construct_gene_list`, same pathway as scripts like Su_2020.py).

Graph perturbation:
- controlled by `--perturb_frac` in [0, 1]
  (0.0 = original annotation graph, 1.0 = fully rewired edge set size).
"""

import os
import math
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch_geometric
import matplotlib.pyplot as plt

from SDAN.preprocess import qc, construct_gene_list, construct_GNN, construct_labels
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
        description="Compare SDAN robustness to annotation graph perturbations."
    )
    parser.add_argument("--no-cuda", action="store_true", default=True)
    parser.add_argument("--n_top_genes", type=int, default=1000,
                        help="Number of DE-HVG genes per cell type (shared SDAN selection).")
    parser.add_argument("--perturb_frac", type=float, default=0.0,
                        help="Fraction of annotation edges to perturb (0.0 to 1.0).")
    parser.add_argument("--n_comp", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--graph_weight", type=float, default=2.0,
                        help="Graph loss weight for this run.")
    parser.add_argument("--start_patience", type=int, default=500)
    parser.add_argument("--epochs_min", type=int, default=2000)
    parser.add_argument("--data_dir", type=str, default="./Zheng_2017/",
                        help="Directory containing sc9_train.h5ad and sc9_test.h5ad.")
    parser.add_argument("--cell_types", type=str, nargs="+",
                        default=["cd4_t_helper", "naive_t"])
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
        raise FileNotFoundError("Could not find BIOGRID file in ./Annotation/ (.txt.gz or .txt)")
    return pd.read_csv(
        biogrid_path,
        sep="\t",
        low_memory=False,
        usecols=["Official Symbol Interactor A", "Official Symbol Interactor B"],
        dtype="string",
    )


def construct_gene_graph_cached(gene_list, biogrid_edges):
    mapping = pd.Series(range(len(gene_list)), index=gene_list)
    edge_df = biogrid_edges[
        (biogrid_edges["Official Symbol Interactor A"].isin(gene_list)) &
        (biogrid_edges["Official Symbol Interactor B"].isin(gene_list))
    ]
    edge_index = torch.as_tensor(
        np.vstack([
            edge_df.iloc[:, 0].map(mapping).to_numpy(dtype=np.int64),
            edge_df.iloc[:, 1].map(mapping).to_numpy(dtype=np.int64),
        ]),
        dtype=torch.long,
    )
    edge_index = torch.unique(edge_index, dim=1)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    return edge_index


def _unique_undirected_pairs(edge_index):
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    pairs = np.stack([src, dst], axis=1)
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.unique(pairs, axis=0)
    return pairs


def perturb_edge_index(edge_index, num_nodes, frac, seed):
    if frac <= 0:
        return edge_index

    pairs = _unique_undirected_pairs(edge_index)
    m = len(pairs)
    if m == 0:
        return edge_index

    rng = np.random.default_rng(seed)
    k = int(round(frac * m))
    k = min(max(k, 0), m)

    if k == 0:
        out_pairs = pairs
    else:
        replace_idx = rng.choice(m, size=k, replace=False)
        keep_mask = np.ones(m, dtype=bool)
        keep_mask[replace_idx] = False
        keep_pairs = pairs[keep_mask]

        existing = {tuple(p) for p in keep_pairs.tolist()}
        new_pairs = []
        while len(new_pairs) < k:
            need = k - len(new_pairs)
            a = rng.integers(0, num_nodes, size=need * 4 + 32, dtype=np.int64)
            b = rng.integers(0, num_nodes, size=need * 4 + 32, dtype=np.int64)
            mask = a != b
            a = a[mask]
            b = b[mask]
            lo = np.minimum(a, b)
            hi = np.maximum(a, b)
            for u, v in zip(lo, hi):
                key = (int(u), int(v))
                if key in existing:
                    continue
                existing.add(key)
                new_pairs.append(key)
                if len(new_pairs) >= k:
                    break

        out_pairs = np.vstack([keep_pairs, np.array(new_pairs, dtype=np.int64)])

    src = np.concatenate([out_pairs[:, 0], out_pairs[:, 1]])
    dst = np.concatenate([out_pairs[:, 1], out_pairs[:, 0]])
    out = torch.as_tensor(np.vstack([src, dst]), dtype=torch.long)
    out, _ = torch_geometric.utils.remove_self_loops(out)
    out = torch.unique(out, dim=1)
    return out


def run_experiment(
    train_data, val_data, test_data,
    edge_index, gene_list, cell_type_list,
    args, d, label,
    train_labels, val_labels, test_labels,
    graph_weight,
):
    run_str = f"{'-'.join(cell_type_list)}_{label}"

    print(f"\n{'='*68}")
    print(f"Running setting: {label}")
    print(f"genes={len(gene_list)} | undirected_edges={len(_unique_undirected_pairs(edge_index))}")
    print(f"graph_weight={graph_weight} (used for both mc_weight/o_weight)")
    print(f"{'='*68}\n")

    train_GNN = construct_GNN(train_data, gene_list, edge_index)
    val_GNN = construct_GNN(val_data, gene_list, edge_index)
    test_GNN = construct_GNN(test_data, gene_list, edge_index)

    run_args = argparse.Namespace(**vars(args))
    run_args.mc_weight = graph_weight
    run_args.o_weight = graph_weight

    model, train_s, loss_list, auc_list = train_with_args(
        [train_GNN, val_GNN, test_GNN],
        [train_labels, val_labels, test_labels],
        train_GNN.num_features,
        len(cell_type_list),
        run_args,
        d,
        run_str,
    )

    torch.save(model, f"{d}output/model_{run_str}.pth")
    np.save(f"{d}output/train_s_{run_str}.npy", train_s.detach().cpu().numpy())
    pd.Series(gene_list).to_csv(f"{d}output/gene_list_{run_str}.txt", header=False, index=False)

    plot_loss(loss_list, run_str, d)
    plot_auc(auc_list, run_str, d)

    eval_device = train_s.device
    val_reduced = val_GNN.x.to(eval_device).t() @ train_s
    test_reduced = test_GNN.x.to(eval_device).t() @ train_s

    _, val_loss, val_auc = test_model(model, val_reduced, val_labels.to(eval_device))
    _, test_loss, test_auc = test_model(model, test_reduced, test_labels.to(eval_device))

    return {
        "setting": label,
        "n_genes": int(len(gene_list)),
        "n_undirected_edges": int(len(_unique_undirected_pairs(edge_index))),
        "graph_weight": float(graph_weight),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "val_loss": float(val_loss.item()),
        "test_loss": float(test_loss.item()),
        "loss_list": loss_list,
        "auc_list": auc_list,
    }


def plot_single_auc(auc_list, d, run_str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    [_, val_auc_list, test_auc_list] = auc_list
    axes[0].plot(val_auc_list, label="val")
    axes[1].plot(test_auc_list, label="test")

    axes[0].set_title("Validation AUC")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Test AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{d}figures/annotation_robustness_{run_str}.pdf")
    plt.close()


def main():
    args = parse_args()
    if not (0.0 <= args.perturb_frac <= 1.0):
        raise ValueError("--perturb_frac must be in [0, 1].")

    d = args.data_dir
    cell_type_list = list(args.cell_types)
    cell_type_str = "-".join(cell_type_list)

    os.makedirs(f"{d}figures/", exist_ok=True)
    os.makedirs(f"{d}output/", exist_ok=True)

    print("Loading data...")
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

    gene_list = construct_gene_list(
        train_data,
        cell_type_list,
        n_top_genes=args.n_top_genes,
        alpha=0.05,
    )
    gene_mode_label = f"dehvg_top{args.n_top_genes}"

    print("Loading annotation graph...")
    biogrid_edges = load_biogrid_edges()
    base_edge_index = construct_gene_graph_cached(gene_list, biogrid_edges)
    frac_label = str(args.perturb_frac).replace(".", "p")
    run_label = f"annotation_perturb{frac_label}"

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    edge_index = perturb_edge_index(
        base_edge_index,
        num_nodes=len(gene_list),
        frac=args.perturb_frac,
        seed=SEED,
    )

    res = run_experiment(
        train_data, val_data, test_data,
        edge_index=edge_index,
        gene_list=gene_list,
        cell_type_list=cell_type_list,
        args=args,
        d=d,
        label=run_label,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        graph_weight=args.graph_weight,
    )
    res["perturb_fraction"] = args.perturb_frac

    run_str = f"{cell_type_str}_{run_label}"
    plot_single_auc(res["auc_list"], d, run_str)

    summary = pd.DataFrame([
        {k: v for k, v in res.items() if k not in ("loss_list", "auc_list")}
    ])
    out_csv = f"{d}output/annotation_robustness_{cell_type_str}_{gene_mode_label}_{run_label}.csv"
    summary.to_csv(out_csv, index=False)

    print("\nSummary:")
    print(summary[[
        "setting", "perturb_fraction", "n_genes", "n_undirected_edges", "graph_weight", "test_auc", "val_auc"
    ]].to_string(index=False))
    print(f"\nSaved summary: {out_csv}")


if __name__ == "__main__":
    main()
