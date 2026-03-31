"""
trainer.py
----------
Orchestrates model training + optional hyperparameter tuning.
Accepts class_weight parameter injected from imbalance config.
"""

import logging
import time
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .models import build_model

logger = logging.getLogger(__name__)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_names: List[str],
    tuning_enabled: bool = True,
    tuning_method: str = "randomized",
    n_iter: int = 20,
    cv_folds: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
) -> Dict[str, Any]:
    """
    Train and optionally tune each requested model.

    class_weight is forwarded to sklearn models; XGBoost handles imbalance
    through scale_pos_weight (binary) or sample_weight (multi-class) —
    for simplicity we rely on SMOTE upstream for XGBoost.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fitted_models: Dict[str, Any] = {}

    # XGBoost requires integer-encoded labels
    le = LabelEncoder()
    y_int = pd.Series(le.fit_transform(y_train), name=y_train.name)

    for name in model_names:
        logger.info(f"{'='*60}")
        logger.info(f"Training: {name.upper()}")
        estimator, param_grid = build_model(
            name, random_state=random_state, class_weight=class_weight
        )

        y_fit = y_int if isinstance(estimator, XGBClassifier) else y_train

        t0 = time.time()
        if tuning_enabled and param_grid:
            best_est = _tune(
                estimator, param_grid, X_train, y_fit,
                method=tuning_method, cv=cv, n_iter=n_iter,
                scoring=scoring, n_jobs=n_jobs, random_state=random_state,
            )
        else:
            logger.info("  Tuning disabled — fitting with default params")
            best_est = estimator.fit(X_train, y_fit)

        logger.info(f"  ✓ Done in {time.time() - t0:.1f}s")
        fitted_models[name] = best_est

    logger.info(f"{'='*60}")
    logger.info(f"All {len(fitted_models)} model(s) trained.")
    return fitted_models


def _tune(estimator, param_grid, X, y, method, cv, n_iter, scoring, n_jobs, random_state):
    Cls = GridSearchCV if method == "grid" else RandomizedSearchCV
    kwargs = dict(
        estimator=estimator, cv=cv, scoring=scoring,
        n_jobs=n_jobs, refit=True, verbose=0,
    )
    if method != "grid":
        kwargs.update(param_distributions=param_grid,
                      n_iter=n_iter, random_state=random_state)
    else:
        kwargs["param_grid"] = param_grid

    search = Cls(**kwargs)
    search.fit(X, y)
    logger.info(f"  Best params : {search.best_params_}")
    logger.info(f"  CV {scoring}: {search.best_score_:.4f}")
    return search.best_estimator_
