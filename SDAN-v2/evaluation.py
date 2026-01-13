# evaluation.py
from typing import Sequence, Iterable
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings

# ---- Global label names (set from backends via set_eval_labels) ----
POS_NAME = "Dementia"      # positive class label in obs['cell_type']
NEG_NAME = "No dementia"   # negative class label in obs['cell_type']

def set_eval_labels(pos_label: str, neg_label: str) -> None:
    """Set global positive/negative class names expected in obs['cell_type']."""
    global POS_NAME, NEG_NAME
    POS_NAME, NEG_NAME = pos_label, neg_label


def _build_models():
    return {
        "logreg": SkPipeline([("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))]),
        # "rf":     SkPipeline([("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1))]),
    }

def _to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)

def _get_estimator(pipeline_or_est):
    return pipeline_or_est.named_steps["clf"] if isinstance(pipeline_or_est, SkPipeline) else pipeline_or_est

def _pos_col_idx(estimator_or_pipeline, pos_id: int) -> int:
    """Return the predict_proba column index for our positive class id."""
    est = _get_estimator(estimator_or_pipeline)
    if hasattr(est, "classes_"):
        classes_est = np.asarray(est.classes_)
        try:
            classes_cmp = classes_est.astype(int)
        except Exception:
            classes_cmp = classes_est
        if pos_id in classes_cmp:
            return int(np.where(classes_cmp == pos_id)[0][0])
        if set(classes_est.tolist()) == {0, 1}:
            return int(np.where(classes_est == 1)[0][0])
        raise RuntimeError(f"Positive class id {pos_id} not in estimator.classes_={classes_est.tolist()}")
    warnings.warn("Estimator has no classes_. Assuming binary; using column 1 as positive.")
    return 1

def eval_and_save(
    train_ad: AnnData,
    test_ad: AnnData,
    feat_prefix: str,
    out_csv: str,
    coef_csv: str,
    cell_type_list: Sequence[str],
    mild_ind: Iterable,  
    severe_ind: Iterable,  
):
    """
    Train classifiers on reduced features and evaluate on test set.
    Computes cell-level AUC (POS_NAME vs others) and individual-level AUC
    by averaging per-cell positive probabilities per 'individual'.

    Parameters
    ----------
    cell_type_list : ordered list of class names present in obs['cell_type'].
                     Must include NEG_NAME and POS_NAME.
    mild_ind : iterable of NEGATIVE individuals (kept for backward compat).
    severe_ind : iterable of POSITIVE individuals (kept for backward compat).
    """
    models = _build_models()

    # Features
    Ztr = _to_dense(train_ad.X)
    Zte = _to_dense(test_ad.X)
    feat_names = [f"{feat_prefix}_{i}" for i in range(Ztr.shape[1])]

    # Map cell_type strings to ids per provided ordering
    mapping_cell = {ct: i for i, ct in enumerate(cell_type_list)}
    if POS_NAME not in mapping_cell or NEG_NAME not in mapping_cell:
        raise ValueError(
            f"cell_type_list must contain NEG_NAME={NEG_NAME!r} and POS_NAME={POS_NAME!r}; "
            f"got {list(cell_type_list)}"
        )
    pos_idx = mapping_cell[POS_NAME]

    y_tr_ids = train_ad.obs['cell_type'].astype(str).map(mapping_cell).to_numpy()
    y_te_ids = test_ad.obs['cell_type'].astype(str).map(mapping_cell).to_numpy()

    # Also keep a binary vector for cell-level AUC
    y_te_bin = (test_ad.obs['cell_type'].astype(str).to_numpy() == POS_NAME).astype(int)

    indiv_te = test_ad.obs['individual'].astype(str).to_numpy()
    neg_set = set(map(str, mild_ind))
    pos_set = set(map(str, severe_ind))

    results = []
    coef_dict = {}

    for name, clf in models.items():

        # Train
        clf.fit(Ztr, y_tr_ids)

        # Align positive class to the correct probability column
        pos_id = mapping_cell[POS_NAME]
        pos_col = _pos_col_idx(clf, pos_id)

        # Predict probabilities for the positive class
        if not hasattr(clf, "predict_proba"):
            raise RuntimeError(f"Model '{name}' does not support predict_proba.")
        probs = clf.predict_proba(Zte)[:, pos_col]

        # Cell-level AUC 
        auc_cell = float(roc_auc_score(y_te_bin, probs))

        # Individual-level: mean prob per individual, label via donor lists
        test_score_ind = pd.Series(probs, index=indiv_te).groupby(level=0).mean()
        inds_known = [ind for ind in test_score_ind.index if (ind in pos_set) or (ind in neg_set)]
        scores_known = test_score_ind.loc[inds_known].values
        y_ind_known = np.array([1 if ind in pos_set else 0 for ind in inds_known], dtype=int)
        auc_ind = float(roc_auc_score(y_ind_known, scores_known))

        print(f"{name} | Cell AUC: {auc_cell:.4f} | Ind AUC: {auc_ind:.4f}")

        # Save per-individual scores (only labeled inds)
        ind_scores_df = pd.DataFrame({
            "individual": inds_known,
            "score_pos": scores_known,
            "label_bin": y_ind_known,
            "label_str": [POS_NAME if v == 1 else NEG_NAME for v in y_ind_known],
        })
        ind_scores_path = out_csv.replace(".csv", f"_{feat_prefix}_{name}_ind_scores.csv")
        ind_scores_df.to_csv(ind_scores_path, index=False)

        results.append({
            "backend": feat_prefix,
            "model": name,
            "auc_cell": float(auc_cell),
            "auc_ind": float(auc_ind),
        })
        # Save coefficients for logistic regression
        if name == "logreg":
            est = _get_estimator(clf)
            if hasattr(est, "coef_"):
                coef = est.coef_
                if coef.ndim == 1 or coef.shape[0] == 1:
                    coef_vec = coef.ravel()
                else:
                    # pick the row corresponding to our positive class id
                    row = int(np.where(est.classes_ == pos_id)[0][0])
                    coef_vec = coef[row, :]
                coef_dict[name] = pd.DataFrame({"feature": feat_names, "coef": coef_vec})

    # Save evaluation summary
    pd.DataFrame(results).to_csv(out_csv, index=False)

    # Save coefficients (if any)
    if coef_dict:
        coef_df = pd.concat(coef_dict.values(), keys=coef_dict.keys(), names=["model"])
        coef_df.to_csv(coef_csv)

    return results
