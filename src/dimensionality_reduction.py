"""
dimensionality_reduction.py
---------------------------
Optional PCA step:
  • Reduces the selected-gene matrix for faster training
  • Produces a 2-D scatter plot coloured by disease class
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional

from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    n_components: int = 50,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Fit PCA on training data and transform both splits.

    Parameters
    ----------
    X_train / X_test : Feature matrices after feature selection.
    n_components     : Number of principal components to keep.
    random_state     : RNG seed.

    Returns
    -------
    X_train_pca, X_test_pca : DataFrames with PC columns.
    pca                     : Fitted PCA object (for variance explained).
    """
    n_components = min(n_components, X_train.shape[1], X_train.shape[0])
    logger.info(
        f"Applying PCA: {X_train.shape[1]} features → {n_components} components"
    )

    pca = PCA(n_components=n_components, random_state=random_state)
    train_arr = pca.fit_transform(X_train)
    test_arr  = pca.transform(X_test)

    cols = [f"PC{i+1}" for i in range(n_components)]
    X_train_pca = pd.DataFrame(train_arr, columns=cols, index=X_train.index)
    X_test_pca  = pd.DataFrame(test_arr,  columns=cols, index=X_test.index)

    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info(
        f"PCA explained variance ({n_components} PCs): {explained:.1f}%"
    )
    return X_train_pca, X_test_pca, pca


def plot_pca_2d(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    output_dir: str = "outputs",
    label_encoder=None,
) -> str:
    """
    Generate a 2-D PCA scatter plot coloured by class label.

    Returns
    -------
    Path to the saved PNG.
    """
    logger.info("Generating PCA 2-D scatter plot …")

    # Always re-fit a fresh 2-component PCA for visualization
    pca2 = PCA(n_components=2, random_state=42)
    coords = pca2.fit_transform(X_train)

    labels = y_train.values
    if label_encoder is not None:
        labels = label_encoder.inverse_transform(labels.astype(int))

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))

    for i, cls in enumerate(np.unique(labels)):
        mask = labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=str(cls),
            color=palette[i],
            alpha=0.65,
            edgecolors="white",
            linewidths=0.4,
            s=40,
        )

    var1 = pca2.explained_variance_ratio_[0] * 100
    var2 = pca2.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)", fontsize=12)
    ax.set_title("PCA 2-D Projection — Gene Expression Data", fontsize=14)
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_path = Path(output_dir) / "pca_2d_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PCA scatter saved → {out_path}")
    return str(out_path)
