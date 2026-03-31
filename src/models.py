"""
models.py
---------
Model factories and hyperparameter search spaces for:
  • Logistic Regression  (baseline)
  • SVM
  • Random Forest
  • XGBoost
  • VotingClassifier ensemble  (built from the above)

class_weight is injected dynamically based on the imbalance strategy.
"""

import logging
from typing import Tuple, Dict, Any, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

ModelSpec = Tuple[Any, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_specs(
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
) -> Dict[str, ModelSpec]:
    """Return {model_name: (estimator, param_grid)} for all base models."""
    return {
        "logistic_regression": _logistic_regression(random_state, class_weight),
        "svm":                 _svm(random_state, class_weight),
        "random_forest":       _random_forest(random_state, class_weight),
        "xgboost":             _xgboost(random_state),   # XGBoost uses scale_pos_weight instead
    }


def build_model(
    name: str,
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
) -> ModelSpec:
    specs = get_model_specs(random_state, class_weight)
    if name not in specs:
        raise ValueError(f"Unknown model '{name}'. Available: {list(specs.keys())}")
    return specs[name]


def build_ensemble(
    fitted_models: Dict[str, Any],
    voting: str = "soft",
) -> Any:
    """
    Build a VotingClassifier from already-fitted base estimators.
    Uses soft voting by default (requires predict_proba on all models).
    Falls back to hard voting if any model lacks predict_proba.
    """
    estimators = []
    for name, model in fitted_models.items():
        if voting == "soft" and not hasattr(model, "predict_proba"):
            logger.warning(
                f"'{name}' lacks predict_proba — switching ensemble to hard voting."
            )
            voting = "hard"
        estimators.append((name, model))

    ensemble = VotingClassifier(estimators=estimators, voting=voting)
    logger.info(
        f"Ensemble built: VotingClassifier({voting}) "
        f"over [{', '.join(fitted_models.keys())}]"
    )
    return ensemble


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _logistic_regression(rs: int, cw: Optional[str]) -> ModelSpec:
    estimator = LogisticRegression(
        max_iter=1000,
        random_state=rs,
        class_weight=cw,
    )
    param_grid = {
        "C":      [0.001, 0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    }
    return estimator, param_grid


def _svm(rs: int, cw: Optional[str]) -> ModelSpec:
    estimator = SVC(
        probability=True,
        random_state=rs,
        class_weight=cw,
    )
    param_grid = {
        "C":      [0.1, 1.0, 10.0, 100.0],
        "kernel": ["rbf", "linear"],
        "gamma":  ["scale", "auto"],
    }
    return estimator, param_grid


def _random_forest(rs: int, cw: Optional[str]) -> ModelSpec:
    estimator = RandomForestClassifier(
        random_state=rs,
        class_weight=cw,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }
    return estimator, param_grid


def _xgboost(rs: int) -> ModelSpec:
    estimator = XGBClassifier(
        random_state=rs,
        eval_metric="mlogloss",
        n_jobs=-1,
        verbosity=0,
    )
    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 5, 7],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma":            [0, 0.1, 0.2],
        "reg_alpha":        [0, 0.1, 0.5],
    }
    return estimator, param_grid
