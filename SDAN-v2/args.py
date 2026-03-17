# args.py
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="SDAN: single-cell classification with GNN, Spectra, or sciRED backends"
    )

    # ------------------ Shared options ------------------
    shared = parser.add_argument_group("Shared options")
    shared.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disable CUDA even if available.")
    shared.add_argument("--cell_type", type=str, required=True,
                        help="Cell type / subset identifier (e.g., cd4_BL, Astro).")
    shared.add_argument("--n_top_genes", type=int, default=1000,
                        help="Number of DE/HVG genes to use for each cell type.")

    # ------------------ GNN options ------------------
    gnn = parser.add_argument_group("GNN options")
    gnn.add_argument("--n_comp", type=int, default=40,
                     help="[GNN/sciRED] Number of components (clusters or factors).")
    gnn.add_argument("--epochs", type=int, default=50000,
                     help="[GNN] Number of training epochs (default=50000).")
    gnn.add_argument("--lr", type=float, default=1e-4,
                     help="[GNN] Learning rate (default=1e-4).")
    gnn.add_argument("--hidden1", type=int, default=64,
                     help="[GNN] Hidden size 1 (default=64).")
    gnn.add_argument("--hidden2", type=int, default=64,
                     help="[GNN] Hidden size 2 (default=64).")
    gnn.add_argument("--graph_weight", type=float, default=1.0,
                     help="[GNN] Weight of graph-related losses (default=1.0).")
    gnn.add_argument("--start_patience", type=int, default=3000,
                     help="[GNN] Patience for early stopping (default=3000).")
    gnn.add_argument("--epochs_min", type=int, default=10000,
                     help="[GNN] Minimum epochs before early stopping (default=10000).")

    # ------------------ Backend selector (positional) ------------------
    parser.add_argument("backend", nargs="?", default="GNN",
                        choices=["GNN", "Spectra", "sciRED"],
                        help="Choose backend: GNN (default), Spectra, or sciRED.")

    # ------------------ Spectra options ------------------
    spectra = parser.add_argument_group("Spectra options")
    spectra.add_argument("--spectra_epochs", type=int, default=10000,
                         help="[Spectra] Number of training epochs (default=10000).")
    spectra.add_argument("--spectra_lam", type=float, default=0.1,
                         help="[Spectra] Lambda: weight of priors (default=0.1).")
    spectra.add_argument("--spectra_delta", type=float, default=0.001,
                         help="[Spectra] Lower bound for gene scaling δ (default=0.001).")
    spectra.add_argument("--spectra_rho", type=float, default=0.001,
                         help="[Spectra] Background rate of 0s ρ (default=0.001).")
    spectra.add_argument("--spectra_L", type=int, default=40,
                         help="[Spectra] Number of factors (default=40).")

    # ------------------ sciRED (Varimax) options ------------------
    scired = parser.add_argument_group("sciRED options")
    scired.add_argument("--varimax_residualize", type=str, default="poisson_glm",
                        choices=["poisson_glm", "none"],
                        help="[sciRED] Residualize counts by library size via Poisson GLM (default) or skip.")
    scired.add_argument("--libsize_key", type=str, default="auto",
                        help='[sciRED] obs column for library size (e.g., "nCount_RNA"). '
                             'Use "auto" to auto-detect among common keys.')
    scired.add_argument("--pca_solver", type=str, default="auto",
                        choices=["auto", "full", "arpack", "randomized"],
                        help="[sciRED] PCA solver (sklearn).")
    scired.add_argument("--random_state", type=int, default=888,
                        help="[sciRED] Random seed for PCA (default=888).")

    # ------------------ Convenience ------------------
    parser.add_argument("--components", type=int, default=None,
                        help="If set, overrides n_comp (GNN/sciRED) and spectra_L (Spectra).")

    args = parser.parse_args()

    # ------------------ Post-processing ------------------
    # GPU flag
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    # Normalize GNN weights if backend=GNN
    if args.backend == "GNN":
        args.mc_weight = args.graph_weight
        args.o_weight  = args.graph_weight

    # Broadcast components if provided
    if args.components is not None:
        args.n_comp    = args.components   # GNN / sciRED
        args.spectra_L = args.components   # Spectra

    # Quick echo for sanity
    print(f"[args] backend={args.backend}  cell_type={args.cell_type}")
    print(f"[args] GNN: n_comp={args.n_comp}, epochs={args.epochs}")
    print(f"[args] Spectra: L={args.spectra_L}, epochs={args.spectra_epochs}, "
          f"lam={args.spectra_lam}, delta={args.spectra_delta}, rho={args.spectra_rho}")
    print(f"[args] sciRED: k={args.n_comp}, residualize={args.varimax_residualize}, "
          f"libsize_key={args.libsize_key}, pca_solver={args.pca_solver}, "
          f"random_state={args.random_state}")

    return args
