"""
evaluator.py
------------
Computes all evaluation metrics for each trained model and
returns structured results ready for reporting and visualization.

Handles the XGBoost edge-case where the model was trained on integer-encoded
labels but y_test may still hold the original string labels.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    average: str = "weighted",
    label_encoder=None,
) -> Dict[str, Any]:
    """
    Compute a full evaluation report for a single model.

    Automatically aligns label types: if the model predicts integers but
    y_test contains strings (XGBoost case), y_test is re-encoded to integers
    for metric computation and the original string labels are kept for display.
    """
    y_pred = model.predict(X_test)

    # ── Align dtypes between y_test and y_pred ────────────────────────────
    y_true_aligned, y_pred_aligned, display_le = _align_labels(y_test, y_pred)

    classes   = np.unique(np.concatenate([y_true_aligned, y_pred_aligned]))
    n_classes = len(classes)

    # ── Scalar metrics ────────────────────────────────────────────────────
    acc  = accuracy_score(y_true_aligned, y_pred_aligned)
    prec = precision_score(y_true_aligned, y_pred_aligned, average=average, zero_division=0)
    rec  = recall_score(y_true_aligned,   y_pred_aligned, average=average, zero_division=0)
    f1   = f1_score(y_true_aligned,       y_pred_aligned, average=average, zero_division=0)

    roc_auc = _compute_roc_auc(model, X_test, y_true_aligned, classes, average)

    # ── Curves ────────────────────────────────────────────────────────────
    roc_data = _compute_roc_curve(model, X_test, y_true_aligned, classes)
    pr_data  = _compute_pr_curve(model, X_test, y_true_aligned, classes)

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true_aligned, y_pred_aligned, labels=classes)

    # ── Human-readable class names ────────────────────────────────────────
    # Priority: outer label_encoder > internal display_le > str(class)
    if label_encoder is not None:
        try:
            class_labels = label_encoder.inverse_transform(classes.astype(int))
        except Exception:
            class_labels = classes.astype(str)
    elif display_le is not None:
        class_labels = display_le.inverse_transform(classes.astype(int))
    else:
        class_labels = classes.astype(str)

    clf_report = classification_report(
        y_true_aligned, y_pred_aligned,
        labels=classes,
        target_names=class_labels,
        zero_division=0,
    )

    results = {
        "model_name":  model_name,
        "accuracy":    round(acc,     4),
        "precision":   round(prec,    4),
        "recall":      round(rec,     4),
        "f1":          round(f1,      4),
        "roc_auc":     round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix":      cm,
        "classes":               classes,
        "class_labels":          class_labels,
        "roc_data":              roc_data,
        "pr_data":               pr_data,
        "classification_report": clf_report,
    }

    _log_results(results)
    return results


def compare_models(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Build a leaderboard DataFrame sorted by F1 descending."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  r["accuracy"],
            "Precision": r["precision"],
            "Recall":    r["recall"],
            "F1":        r["f1"],
            "ROC-AUC":   r["roc_auc"],
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align_labels(y_test: pd.Series, y_pred: np.ndarray):
    """
    Ensure y_test and y_pred share the same dtype.

    Returns (y_true_aligned, y_pred_aligned, display_le_or_None).
    display_le is set when we had to encode string labels to integers,
    so the caller can decode them back for display.
    """
    y_true = y_test.values
    test_is_str = y_true.dtype.kind in ("U", "O")   # string / object
    pred_is_int = np.issubdtype(y_pred.dtype, np.integer)

    if test_is_str and pred_is_int:
        # XGBoost case: encode string labels to int to match predictions
        le = LabelEncoder()
        y_true_enc = le.fit_transform(y_true)
        return y_true_enc, y_pred, le

    if not test_is_str and not pred_is_int:
        # Both numeric — cast to same type
        return y_true.astype(y_pred.dtype), y_pred, None

    # Types already match (both str or both int)
    return y_true, y_pred, None


def _compute_roc_auc(model, X_test, y_true, classes, average) -> Optional[float]:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X_test)
        if len(classes) == 2:
            return roc_auc_score(y_true, proba[:, 1])
        return roc_auc_score(y_true, proba, multi_class="ovr", average=average)
    except Exception as e:
        logger.warning(f"ROC-AUC failed: {e}")
        return None


def _compute_roc_curve(model, X_test, y_true, classes) -> Optional[dict]:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X_test)
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_true, proba[:, 1])
            auc = roc_auc_score(y_true, proba[:, 1])
            return {"fpr": fpr, "tpr": tpr, "auc": auc, "binary": True}
        y_bin = label_binarize(y_true, classes=classes)
        curves = {}
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            auc = roc_auc_score(y_bin[:, i], proba[:, i])
            curves[str(cls)] = {"fpr": fpr, "tpr": tpr, "auc": auc}
        return {"curves": curves, "binary": False}
    except Exception as e:
        logger.warning(f"ROC curve error: {e}")
        return None


def _compute_pr_curve(model, X_test, y_true, classes) -> Optional[dict]:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X_test)
        if len(classes) == 2:
            prec, rec, _ = precision_recall_curve(y_true, proba[:, 1])
            ap = average_precision_score(y_true, proba[:, 1])
            return {"precision": prec, "recall": rec, "ap": ap, "binary": True}
        y_bin = label_binarize(y_true, classes=classes)
        curves = {}
        for i, cls in enumerate(classes):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
            ap = average_precision_score(y_bin[:, i], proba[:, i])
            curves[str(cls)] = {"precision": prec, "recall": rec, "ap": ap}
        return {"curves": curves, "binary": False}
    except Exception as e:
        logger.warning(f"PR curve error: {e}")
        return None


def _log_results(r: dict) -> None:
    logger.info(
        f"\n  ── {r['model_name'].upper()} ──\n"
        f"  Accuracy  : {r['accuracy']:.4f}\n"
        f"  Precision : {r['precision']:.4f}\n"
        f"  Recall    : {r['recall']:.4f}\n"
        f"  F1        : {r['f1']:.4f}\n"
        f"  ROC-AUC   : {r['roc_auc']}\n"
        f"\n{r['classification_report']}"
    )
