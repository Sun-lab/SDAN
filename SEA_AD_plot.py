import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scanpy as sc
import torch
from SDAN.utils import plot_contingency, plot_confusion, plot_s
from sklearn.metrics import roc_auc_score

d = "./SEA_AD/"
weight_list = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
for weight in weight_list:
    score_ind_Astro = np.load(f'{d}output/score_ind_Astro_{weight}.npy')
    score_ind_Micro = np.load(f'{d}output/score_ind_Micro-PVM_{weight}.npy')
    score_Astro = np.load(f'{d}output/score_Astro_{weight}.npy')
    score_Micro = np.load(f'{d}output/score_Micro-PVM_{weight}.npy')

    print(f'Weight: {weight}, Astro AUC: {roc_auc_score(score_Astro[:, 2], score_Astro[:, 1]):.3f}, '
          f'Micro AUC: {roc_auc_score(score_Micro[:, 2], score_Micro[:, 1]):.3f},'
          f'Astro Ind AUC: {roc_auc_score(score_ind_Astro[:, 0], score_ind_Astro[:, 1]):.3f}, '
          f'Micro Ind AUC: {roc_auc_score(score_ind_Micro[:, 0], score_ind_Micro[:, 1]):.3f}')

    plt.figure(figsize=(4, 3), dpi=300)
    plt.hist([score_Astro[:, 1][score_Astro[:, 2] == 0], score_Astro[:, 1][score_Astro[:, 2] == 1]],
             density=True, bins=10, label=["No dementia", "Dementia"])
    plt.legend(loc='upper right')
    plt.title('Prediction scores, Cell')
    plt.savefig(f'{d}figures/score_test_Astro_{weight}.pdf')

    plt.figure(figsize=(4, 3), dpi=300)
    plt.hist([score_Micro[:, 1][score_Micro[:, 2] == 0], score_Micro[:, 1][score_Micro[:, 2] == 1]],
             density=True, bins=10, label=["No dementia", "Dementia"])
    plt.legend(loc='upper right')
    plt.title('Prediction scores, Cell')
    plt.savefig(f'{d}figures/score_test_Micro-PVM_{weight}.pdf')

    type_ind = np.array(["No dementia", "Dementia"])

    plt.figure(figsize=(4, 3), dpi=300)
    sns.boxplot(x=type_ind[score_ind_Astro[:, 0].astype(int)], y=score_ind_Astro[:, 1], showfliers=False,
                order=["No dementia", "Dementia"]).set(ylabel=None, title='Prediction scores, Ind')
    sns.stripplot(x=type_ind[score_ind_Astro[:, 0].astype(int)], y=score_ind_Astro[:, 1],
                  color='black', order=["No dementia", "Dementia"])
    plt.savefig(f'{d}figures/boxplot_score_Astro_{weight}.pdf')
    plt.close()

    plt.figure(figsize=(4, 3), dpi=300)
    sns.boxplot(x=type_ind[score_ind_Micro[:, 0].astype(int)], y=score_ind_Micro[:, 1], showfliers=False,
                order=["No dementia", "Dementia"]).set(ylabel=None, title='Prediction scores, Ind')
    sns.stripplot(x=type_ind[score_ind_Micro[:, 0].astype(int)], y=score_ind_Micro[:, 1],
                  color='black', order=["No dementia", "Dementia"])
    plt.savefig(f'{d}figures/boxplot_score_Micro-PVM_{weight}.pdf')
    plt.close()

    plt.figure(figsize=(5, 4), dpi=300)
    plt.scatter(score_ind_Astro[score_ind_Astro[:, 0] == 0, 1], score_ind_Micro[score_ind_Micro[:, 0] == 0, 1])
    plt.scatter(score_ind_Astro[score_ind_Astro[:, 0] == 1, 1], score_ind_Micro[score_ind_Micro[:, 0] == 1, 1])
    plt.legend(["No dementia", "Dementia"], loc="lower right")
    plt.xlabel("Astro")
    plt.ylabel("Micro-PVM")
    plt.title("Prediction scores, Ind")
    plt.savefig(f'{d}figures/score_cross_{weight}.pdf')
    plt.close()

    for cell_type_str in [f"Astro_{weight}", f"Micro-PVM_{weight}"]:
        train_s_dir = f'{d}output/train_s_{cell_type_str}.npy'
        train_s = torch.tensor(np.load(train_s_dir))
        test_reduced = sc.read(f'{d}output/test_reduced_{cell_type_str}.h5ad')
        test_reduced.obs['cell_type'] = pd.Categorical(test_reduced.obs['cell_type'], categories=["No dementia", "Dementia"],
                                                       ordered=True)
        test_reduced = test_reduced[test_reduced.obs['cell_type'].argsort()]

        sc.settings.set_figure_params(figsize=(3, 3), dpi=300)
        sc.pl.tsne(test_reduced, color=["cell_type", "leiden"], title=['status', 'leiden'], return_fig=True, wspace=0.5)
        plt.savefig(f'{d}figures/tsne_{cell_type_str}.pdf')
        plt.close()

        plt.figure(figsize=(2, 3), dpi=300)
        plot_contingency(test_reduced, ["No dementia", "Dementia"], cell_type_str, d)
        plt.close()

        plt.figure()
        plot_confusion(test_reduced, cell_type_str, d)
        plt.tight_layout()
        plt.close()

        plt.figure(figsize=(3, 3), dpi=300)
        plot_s(train_s, cell_type_str, d)
        plt.close()
