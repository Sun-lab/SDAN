import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--cell_type', type=str,
                        help='Cell type of the data.')
    parser.add_argument('--n_top_genes', type=int, default=1000,
                        help='Number of DE genes to use for each type.')
    parser.add_argument('--n_comp', type=int, default=40,
                        help='Number of components.')
    parser.add_argument('--epochs', type=int, default=50000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--hidden1', type=int, default=64,
                        help='Number of first hidden units.')
    parser.add_argument('--hidden2', type=int, default=64,
                        help='Number of second hidden units.')
    parser.add_argument('--graph_weight', type=float, default=1,
                        help='Weight of the graph loss.')
    parser.add_argument('--mc_weight', type=float, default=1,
                        help='Weight of the minCUT loss.')
    parser.add_argument('--o_weight', type=float, default=1,
                        help='Weight of the orthogonality loss')
    parser.add_argument('--start_patience', type=int, default=3000,
                        help='Number of patience for early stopping.')
    parser.add_argument('--epochs_min', type=int, default=10000,
                        help='Minimum number of epochs to train.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
