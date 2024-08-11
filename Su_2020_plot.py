import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scanpy as sc
import torch
from SDAN.utils import plot_contingency, plot_confusion, plot_s
from sklearn.metrics import roc_auc_score

d = "./Su_2020/"
weight_list = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
for weight in weight_list:
    score_ind_CD4 = np.load(f'{d}output/score_ind_cd4_BL_{weight}.npy')
    score_ind_CD8 = np.load(f'{d}output/score_ind_cd8_BL_{weight}.npy')
    score_CD4 = np.load(f'{d}output/score_cd4_BL_{weight}.npy')
    score_CD8 = np.load(f'{d}output/score_cd8_BL_{weight}.npy')

    print(f'Weight: {weight}, CD4 AUC: {roc_auc_score(score_CD4[:, 2], score_CD4[:, 1]):.3f}, '
          f'CD8 AUC: {roc_auc_score(score_CD8[:, 2], score_CD8[:, 1]):.3f},'
          f'CD4 Ind AUC: {roc_auc_score(score_ind_CD4[:, 0], score_ind_CD4[:, 1]):.3f}, '
          f'CD8 Ind AUC: {roc_auc_score(score_ind_CD8[:, 0], score_ind_CD8[:, 1]):.3f}')

    plt.figure(figsize=(4, 3), dpi=300)
    plt.hist([score_CD4[:, 1][score_CD4[:, 2] == 0], score_CD4[:, 1][score_CD4[:, 2] == 1]],
             density=True, bins=10, label=["mild", "severe"])
    plt.legend(loc='upper right')
    plt.title('Prediction scores, Cell')
    plt.savefig(f'{d}figures/score_test_cd4_BL_{weight}.pdf')
    plt.close()

    plt.figure(figsize=(4, 3), dpi=300)
    plt.hist([score_CD8[:, 1][score_CD8[:, 2] == 0], score_CD8[:, 1][score_CD8[:, 2] == 1]],
             density=True, bins=10, label=["mild", "severe"])
    plt.legend(loc='upper right')
    plt.title('Prediction scores, Cell')
    plt.savefig(f'{d}figures/score_test_cd8_BL_{weight}.pdf')
    plt.close()

    type_ind = np.array(["mild", "severe"])

    plt.figure(figsize=(4, 3), dpi=300)
    sns.boxplot(x=type_ind[score_ind_CD4[:, 0].astype(int)], y=score_ind_CD4[:, 1], showfliers=False,
                order=["mild", "severe"]).set(ylabel=None, title='Prediction scores, Ind')
    sns.stripplot(x=type_ind[score_ind_CD4[:, 0].astype(int)], y=score_ind_CD4[:, 1],
                  color='black', order=["mild", "severe"])
    plt.savefig(f'{d}figures/boxplot_score_cd4_BL_{weight}.pdf')
    plt.close()

    plt.figure(figsize=(4, 3), dpi=300)
    sns.boxplot(x=type_ind[score_ind_CD8[:, 0].astype(int)], y=score_ind_CD8[:, 1], showfliers=False,
                order=["mild", "severe"]).set(ylabel=None, title='Prediction scores, Ind')
    sns.stripplot(x=type_ind[score_ind_CD8[:, 0].astype(int)], y=score_ind_CD8[:, 1],
                  color='black', order=["mild", "severe"])
    plt.savefig(f'{d}figures/boxplot_score_cd8_BL_{weight}.pdf')
    plt.close()

    plt.figure(figsize=(5, 4), dpi=300)
    plt.scatter(score_ind_CD4[score_ind_CD4[:, 0] == 0, 1], score_ind_CD8[score_ind_CD8[:, 0] == 0, 1])
    plt.scatter(score_ind_CD4[score_ind_CD4[:, 0] == 1, 1], score_ind_CD8[score_ind_CD8[:, 0] == 1, 1])
    plt.legend(["mild", "severe"], loc="lower right")
    plt.xlabel("CD4")
    plt.ylabel("CD8")
    plt.title("Prediction scores, Ind")
    plt.savefig(f'{d}figures/score_cross_{weight}.pdf')
    plt.close()

    for cell_type_str in [f"cd4_BL_{weight}", f"cd8_BL_{weight}"]:
        train_s_dir = f'{d}output/train_s_{cell_type_str}.npy'
        train_s = torch.tensor(np.load(train_s_dir))
        test_reduced = sc.read(f'{d}output/test_reduced_{cell_type_str}.h5ad')
        test_reduced.obs['cell_type'] = pd.Categorical(test_reduced.obs['cell_type'], categories=["mild", "severe"],
                                                       ordered=True)
        test_reduced = test_reduced[test_reduced.obs['cell_type'].argsort()]

        sc.settings.set_figure_params(figsize=(3, 3), dpi=300)
        sc.pl.tsne(test_reduced, color=["cell_type", "leiden"], title=['disease status', 'leiden'], return_fig=True, wspace=0.5)
        plt.savefig(f'{d}figures/tsne_{cell_type_str}.pdf')
        plt.close()

        plt.figure(figsize=(2, 3), dpi=300)
        plot_contingency(test_reduced, ["mild", "severe"], cell_type_str, d)
        plt.close()

        plt.figure()
        plot_confusion(test_reduced, cell_type_str, d)
        plt.tight_layout()
        plt.close()

        plt.figure(figsize=(3, 3), dpi=300)
        plot_s(train_s, cell_type_str, d)
        plt.close()
