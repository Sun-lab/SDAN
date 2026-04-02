# args.py
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="SDAN: single-cell classification with SDAN, Spectra, sciRED, or scNET backends"
    )

    # ------------------ Shared options ------------------
    shared = parser.add_argument_group("Shared options")
    shared.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disable CUDA even if available.")
    shared.add_argument("--cell_type", type=str, required=True,
                        help="Cell type / subset identifier (e.g., cd4_BL, Astro).")
    shared.add_argument("--n_top_genes", type=int, default=1000,
                        help="Number of DE/HVG genes to use for each cell type.")

    # ------------------ SDAN options ------------------
    SDAN = parser.add_argument_group("SDAN options")
    SDAN.add_argument("--n_comp", type=int, default=40,
                     help="[SDAN/sciRED/scNET] Number of components (clusters or factors).")
    SDAN.add_argument("--epochs", type=int, default=50000,
                     help="[SDAN] Number of training epochs (default=50000).")
    SDAN.add_argument("--lr", type=float, default=1e-4,
                     help="[SDAN] Learning rate (default=1e-4).")
    SDAN.add_argument("--hidden1", type=int, default=64,
                     help="[SDAN] Hidden size 1 (default=64).")
    SDAN.add_argument("--hidden2", type=int, default=64,
                     help="[SDAN] Hidden size 2 (default=64).")
    SDAN.add_argument("--graph_weight", type=float, default=1.0,
                     help="[SDAN] Weight of graph-related losses (default=1.0).")
    SDAN.add_argument("--mc_weight", type=float, default=1.0,
                     help="[SDAN] Weight of minCUT loss (overridden in scripts).")
    SDAN.add_argument("--o_weight", type=float, default=1.0,
                     help="[SDAN] Weight of orthogonality loss (overridden in scripts).")
    SDAN.add_argument("--start_patience", type=int, default=3000,
                     help="[SDAN] Patience for early stopping (default=3000).")
    SDAN.add_argument("--epochs_min", type=int, default=10000,
                     help="[SDAN] Minimum epochs before early stopping (default=10000).")

    # ------------------ Backend selector (positional) ------------------
    parser.add_argument("backend", nargs="?", default="SDAN",
                        choices=["SDAN", "Spectra", "sciRED", "scNET"],
                        help="Choose backend: SDAN (default), Spectra, sciRED, or scNET.")

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

    # ------------------ scNET options ------------------
    scnet = parser.add_argument_group("scNET options")
    scnet.add_argument("--scnet_epochs", type=int, default=300,
                    help="[scNET] Number of training epochs (default=300).")
    scnet.add_argument("--scnet_batches", type=int, default=5,
                    help="[scNET] Number of mini-batches (default=5).")

    # ------------------ Convenience ------------------
    parser.add_argument("--components", type=int, default=None,
                        help="If set, overrides n_comp (SDAN/sciRED) and spectra_L (Spectra).")

    args = parser.parse_args()

    # ------------------ Post-processing ------------------
    # GPU flag
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    # Normalize SDAN weights if backend=SDAN
    if args.backend == "SDAN":
        args.mc_weight = args.graph_weight
        args.o_weight  = args.graph_weight

    # Broadcast components if provided
    if args.components is not None:
        args.n_comp    = args.components   # SDAN / sciRED
        args.spectra_L = args.components   # Spectra

    if args.backend == "scNET" and args.n_comp == 40:
        args.n_comp = 75

    if args.backend == "SDAN":
        print(f"[args] SDAN: n_comp={args.n_comp}, epochs={args.epochs}")

    elif args.backend == "Spectra":
        print(f"[args] Spectra: L={args.spectra_L}, epochs={args.spectra_epochs}, "
            f"lam={args.spectra_lam}, delta={args.spectra_delta}, rho={args.spectra_rho}")

    elif args.backend == "sciRED":
        print(f"[args] sciRED: k={args.n_comp}, residualize={args.varimax_residualize}, "
            f"libsize_key={args.libsize_key}, pca_solver={args.pca_solver}, "
            f"random_state={args.random_state}")

    elif args.backend == "scNET":
        print(f"[args] scNET: n_comp={args.n_comp}, epochs={args.scnet_epochs}, batches={args.scnet_batches}")
        
    return args
