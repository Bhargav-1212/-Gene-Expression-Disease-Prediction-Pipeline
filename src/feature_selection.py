"""
feature_selection.py
--------------------
Implements multiple gene-selection strategies:
  • VarianceThreshold   – removes near-constant genes
  • SelectKBest         – ANOVA F-test or mutual information
  • L1 / Lasso          – embedded linear selection
  • "all"               – runs all three, takes the union
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    method: str = "selectkbest",
    top_k: int = 100,
    variance_threshold: float = 0.01,
    selectkbest_score: str = "f_classif",
    lasso_alpha: float = 0.001,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Run the requested feature-selection method.

    Parameters
    ----------
    X_train / X_test      : Scaled feature matrices.
    y_train               : Training labels.
    method                : "variance" | "selectkbest" | "lasso" | "all"
    top_k                 : Target number of features for SelectKBest / Lasso.
    variance_threshold    : Cut-off for VarianceThreshold.
    selectkbest_score     : "f_classif" or "mutual_info_classif".
    lasso_alpha           : Regularisation strength for Lasso selection.

    Returns
    -------
    X_train_sel, X_test_sel : DataFrames with selected columns only.
    selected_genes          : List of selected gene names.
    """
    logger.info(f"Feature selection — method: '{method}', top_k: {top_k}")

    dispatch = {
        "variance":    _variance_selection,
        "selectkbest": _selectkbest_selection,
        "lasso":       _lasso_selection,
        "all":         _all_methods,
    }

    if method not in dispatch:
        raise ValueError(
            f"Unknown feature selection method '{method}'. "
            f"Choose from: {list(dispatch.keys())}"
        )

    selector_fn = dispatch[method]

    # Keyword arguments vary per method – pass all, each fn takes what it needs
    X_train_sel, selected_genes = selector_fn(
        X_train, y_train,
        top_k=top_k,
        variance_threshold=variance_threshold,
        score_func=selectkbest_score,
        lasso_alpha=lasso_alpha,
    )

    X_test_sel = X_test[selected_genes]

    logger.info(
        f"Genes selected: {len(selected_genes)} "
        f"(from {X_train.shape[1]} input features)"
    )
    return X_train_sel, X_test_sel, selected_genes


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

def _variance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    variance_threshold: float = 0.01,
    top_k: int = 100,
    **_,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove features whose variance falls below the threshold."""
    sel = VarianceThreshold(threshold=variance_threshold)
    sel.fit(X)
    mask = sel.get_support()
    selected = X.columns[mask].tolist()

    # If too many survive, keep top_k by variance
    if len(selected) > top_k:
        variances = X[selected].var()
        selected = variances.nlargest(top_k).index.tolist()

    logger.info(
        f"VarianceThreshold(threshold={variance_threshold}) "
        f"→ {len(selected)} genes"
    )
    return X[selected], selected


def _selectkbest_selection(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    top_k: int = 100,
    score_func: str = "f_classif",
    **_,
) -> Tuple[pd.DataFrame, List[str]]:
    """Select top-k genes by ANOVA F-score or mutual information."""
    fn = mutual_info_classif if score_func == "mutual_info_classif" else f_classif
    k = min(top_k, X.shape[1])
    sel = SelectKBest(score_func=fn, k=k)
    sel.fit(X, y)
    mask = sel.get_support()
    selected = X.columns[mask].tolist()

    logger.info(f"SelectKBest({score_func}, k={k}) → {len(selected)} genes")
    return X[selected], selected


def _lasso_selection(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    lasso_alpha: float = 0.001,
    top_k: int = 100,
    **_,
) -> Tuple[pd.DataFrame, List[str]]:
    """Use L1-penalised Logistic Regression as an embedded selector."""
    # Ensure labels are integer-encoded
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    estimator = LogisticRegression(
        penalty="l1",
        C=1.0 / max(lasso_alpha, 1e-9),
        solver="liblinear",
        max_iter=1000,
        random_state=42,
        multi_class="auto",
    )
    sel = SelectFromModel(estimator, max_features=top_k, threshold=-np.inf)
    sel.fit(X, y)
    mask = sel.get_support()
    selected = X.columns[mask].tolist()

    # Edge case: if lasso collapses everything, fall back to top_k by coefficient norm
    if len(selected) == 0:
        logger.warning("Lasso selected 0 features — falling back to SelectKBest.")
        return _selectkbest_selection(X, y, top_k=top_k)

    logger.info(f"Lasso(alpha={lasso_alpha}) → {len(selected)} genes")
    return X[selected], selected


def _all_methods(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    top_k: int = 100,
    variance_threshold: float = 0.01,
    score_func: str = "f_classif",
    lasso_alpha: float = 0.001,
    **_,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run all three selectors and take the union (capped at top_k)."""
    _, var_genes   = _variance_selection(X, y, variance_threshold=variance_threshold, top_k=top_k)
    _, kb_genes    = _selectkbest_selection(X, y, top_k=top_k, score_func=score_func)
    _, lasso_genes = _lasso_selection(X, y, lasso_alpha=lasso_alpha, top_k=top_k)

    # Union, then cap at top_k (prioritise genes appearing in multiple methods)
    from collections import Counter
    vote = Counter(var_genes + kb_genes + lasso_genes)
    # Sort by vote count desc, then alphabetically for determinism
    selected = [g for g, _ in sorted(vote.items(), key=lambda kv: (-kv[1], kv[0]))]
    selected = selected[:top_k]

    logger.info(
        f"Union of all methods → {len(selected)} genes "
        f"(var={len(var_genes)}, kb={len(kb_genes)}, lasso={len(lasso_genes)})"
    )
    return X[selected], selected
