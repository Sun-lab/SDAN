import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scanpy as sc
import torch
from SDAN.utils import plot_contingency, plot_confusion, plot_s
from sklearn.metrics import roc_auc_score

d = "./Yost_2019/"
weight_list = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
for weight in weight_list:
    score_ind = np.load(f'{d}output/score_ind_CD8T_{weight}.npy')
    score = np.load(f'{d}output/score_CD8T_{weight}.npy')

    print(f'Weight: {weight}, AUC: {roc_auc_score(score[:, 2], score[:, 1]):.3f}, '
          f'Ind AUC: {roc_auc_score(score_ind[:, 0], score_ind[:, 1]):.3f}')

    plt.figure(figsize=(4, 3), dpi=300)
    plt.hist([score[:, 1][score[:, 2] == 0], score[:, 1][score[:, 2] == 1]],
             density=True, bins=10, label=["No", "Yes"])
    plt.legend(loc='upper right')
    plt.title('Prediction scores, Cell')
    plt.savefig(f'{d}figures/score_CD8T_{weight}.pdf')
    plt.close()

    type_ind = np.array(["No", "Yes"])

    plt.figure(figsize=(4, 3), dpi=300)
    sns.boxplot(x=type_ind[score_ind[:, 0].astype(int)], y=score_ind[:, 1], showfliers=False,
                order=["No", "Yes"]).set(ylabel=None, title='Prediction scores, Ind')
    sns.stripplot(x=type_ind[score_ind[:, 0].astype(int)], y=score_ind[:, 1],
                  color='black', order=["No", "Yes"])
    plt.savefig(f'{d}figures/boxplot_score_CD8T_{weight}.pdf')
    plt.close()

    train_s_dir = f'{d}output/train_s_CD8T_{weight}.npy'
    train_s = torch.tensor(np.load(train_s_dir))
    test_reduced = sc.read(f'{d}output/test_reduced_CD8T_{weight}.h5ad')
    test_reduced.obs['cell_type'] = pd.Categorical(test_reduced.obs['cell_type'], categories=["No", "Yes"],
                                                   ordered=True)
    test_reduced = test_reduced[test_reduced.obs['cell_type'].argsort()]

    sc.settings.set_figure_params(figsize=(3, 3), dpi=300)
    sc.pl.tsne(test_reduced, color=["cell_type", "leiden"], title=['disease status', 'leiden'], return_fig=True,
               wspace=0.5)
    plt.savefig(f'{d}figures/tsne_CD8T_{weight}.pdf')
    plt.close()

    plt.figure(figsize=(2, 3), dpi=300)
    plot_contingency(test_reduced, ["No", "Yes"], f"CD8T_{weight}", d)
    plt.close()

    plt.figure()
    plot_confusion(test_reduced, f"CD8T_{weight}", d)
    plt.tight_layout()
    plt.close()

    plt.figure(figsize=(3, 3), dpi=300)
    plot_s(train_s, f"CD8T_{weight}", d)
    plt.close()
