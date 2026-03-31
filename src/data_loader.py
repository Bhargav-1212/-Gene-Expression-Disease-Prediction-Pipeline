"""
data_loader.py
--------------
Handles all data ingestion, validation, and preprocessing for the gene
expression disease prediction pipeline.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder
)
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    file_path: str,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a CSV gene-expression dataset and return features + labels.

    Parameters
    ----------
    file_path     : Path to the input CSV file.
    target_column : Name of the column that holds disease labels.

    Returns
    -------
    X : pd.DataFrame  – feature matrix (genes as columns)
    y : pd.Series     – label vector
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)[:10]} ..."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Drop non-numeric columns that can't be gene expression values
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(
            f"Dropping {len(non_numeric)} non-numeric column(s): {non_numeric[:5]}"
        )
        X = X.drop(columns=non_numeric)

    logger.info(
        f"Dataset loaded — samples: {len(X)}, genes: {X.shape[1]}, "
        f"classes: {y.nunique()}"
    )
    _log_class_distribution(y)
    return X, y


def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    normalization: str = "standard",
    handle_missing: str = "median",
    encode_labels: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Full preprocessing pipeline:
      1. Missing-value imputation
      2. Label encoding (optional)
      3. Stratified train/test split
      4. Feature scaling (fitted on train only)

    Parameters
    ----------
    X               : Raw feature matrix.
    y               : Raw label vector.
    normalization   : "standard" | "minmax" | "none"
    handle_missing  : "mean" | "median" | "drop"
    encode_labels   : Whether to LabelEncode string targets.
    test_size       : Proportion of data held out for testing.
    random_state    : RNG seed.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test,
        scaler, label_encoder, feature_names
    """
    logger.info("Starting preprocessing …")

    # ── 1. Handle missing values ──────────────────────────────────────────
    X = _impute(X, strategy=handle_missing)

    # ── 2. Encode labels ──────────────────────────────────────────────────
    label_encoder = None
    if encode_labels and y.dtype == object:
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=y.name)
        logger.info(
            f"Labels encoded: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
        )

    # ── 3. Stratified split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        f"Split — train: {len(X_train)}, test: {len(X_test)}"
    )

    # ── 4. Scaling (fit on train, transform both) ─────────────────────────
    scaler = _build_scaler(normalization)
    if scaler is not None:
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        logger.info(f"Normalization applied: {normalization}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": list(X_train.columns),
    }


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _impute(X: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Impute missing values or drop rows with any NaN."""
    missing = X.isnull().sum().sum()
    if missing == 0:
        logger.info("No missing values detected.")
        return X

    logger.info(f"Handling {missing} missing value(s) with strategy='{strategy}'")

    if strategy == "drop":
        X = X.dropna()
    else:
        imputer = SimpleImputer(strategy=strategy)
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )
    return X


def _build_scaler(normalization: str) -> Optional[object]:
    """Return a fitted-ready scaler or None."""
    if normalization == "standard":
        return StandardScaler()
    if normalization == "minmax":
        return MinMaxScaler()
    if normalization in ("none", None):
        return None
    raise ValueError(
        f"Unknown normalization '{normalization}'. "
        "Use 'standard', 'minmax', or 'none'."
    )


def _log_class_distribution(y: pd.Series) -> None:
    counts = y.value_counts()
    logger.info("Class distribution:")
    for label, cnt in counts.items():
        pct = cnt / len(y) * 100
        logger.info(f"  {label}: {cnt} ({pct:.1f}%)")
