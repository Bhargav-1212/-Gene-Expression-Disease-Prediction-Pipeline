"""
imbalance.py
------------
Handles class imbalance at the training-data level.

Strategies:
  • "smote"         – Synthetic Minority Over-sampling (SMOTE)
  • "class_weight"  – Delegate to model's class_weight param (no resampling)
  • "both"          – SMOTE + class_weight on the model
  • "none"          – No intervention

SMOTE is applied AFTER the train/test split and ONLY on training data,
never on the test set, to avoid data leakage.
"""

import logging
import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple

logger = logging.getLogger(__name__)


def apply_imbalance_strategy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    strategy: str = "smote",
    smote_k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply the configured imbalance-handling strategy to training data.

    Parameters
    ----------
    X_train           : Training feature matrix (post feature-selection).
    y_train           : Training labels.
    strategy          : "smote" | "class_weight" | "both" | "none"
    smote_k_neighbors : k for SMOTE; auto-capped to minority class size - 1.
    random_state      : RNG seed.

    Returns
    -------
    X_resampled, y_resampled  (unchanged if strategy is "class_weight" or "none")
    """
    _log_distribution("Before resampling", y_train)

    if strategy in ("class_weight", "none"):
        logger.info(
            f"Imbalance strategy='{strategy}' — no resampling applied. "
            "Class weights are handled at model level."
        )
        return X_train, y_train

    if strategy in ("smote", "both"):
        X_res, y_res = _apply_smote(
            X_train, y_train,
            k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )
        _log_distribution("After SMOTE", y_res)
        return X_res, y_res

    raise ValueError(
        f"Unknown imbalance strategy '{strategy}'. "
        "Choose: 'smote', 'class_weight', 'both', 'none'."
    )


def get_class_weight_param(strategy: str) -> str | None:
    """
    Return the class_weight value to pass into sklearn models.
    Returns "balanced" if class weights should be active, else None.
    """
    return "balanced" if strategy in ("class_weight", "both") else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    k_neighbors: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.error(
            "imbalanced-learn not installed. "
            "Run: pip install imbalanced-learn"
        )
        return X, y

    # SMOTE requires k_neighbors < minority class size
    min_class_count = min(Counter(y).values())
    k = min(k_neighbors, min_class_count - 1)
    if k < 1:
        logger.warning(
            f"Minority class has only {min_class_count} sample(s) — "
            "SMOTE skipped (need at least 2)."
        )
        return X, y

    if k != k_neighbors:
        logger.warning(
            f"SMOTE k_neighbors reduced {k_neighbors} → {k} "
            f"(minority class size={min_class_count})."
        )

    smote = SMOTE(k_neighbors=k, random_state=random_state)
    X_arr, y_arr = smote.fit_resample(X.values, y.values)

    X_res = pd.DataFrame(X_arr, columns=X.columns)
    y_res = pd.Series(y_arr, name=y.name)
    return X_res, y_res


def _log_distribution(label: str, y: pd.Series) -> None:
    counts  = Counter(y)
    total   = len(y)
    parts   = ", ".join(
        f"{cls}: {cnt} ({cnt/total*100:.1f}%)"
        for cls, cnt in sorted(counts.items(), key=lambda x: str(x[0]))
    )
    logger.info(f"Class distribution [{label}] — {parts}")
