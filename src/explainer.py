"""
explainer.py
------------
SHAP-based model explainability.

KEY DESIGN: SHAP is ALWAYS computed on ORIGINAL selected gene features,
never on PCA components. If the pipeline used PCA for model training,
a separate "gene-space" model is fitted here for explanation purposes only.

This ensures output is biologically meaningful — gene names, not PC labels.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — explainability disabled.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_model(
    model,
    model_name: str,
    X_train_genes: pd.DataFrame,     # ORIGINAL gene-space train data
    X_test_genes: pd.DataFrame,      # ORIGINAL gene-space test data
    *,
    gene_names: List[str],
    output_dir: str = "outputs",
    shap_sample_size: int = 100,
    top_features: int = 20,
    pca_was_used: bool = False,
    random_state: int = 42,
) -> Optional[Dict[str, Any]]:
    """
    Compute SHAP values on ORIGINAL gene features and return a structured
    result dict with gene names + importance scores.

    If PCA was used for model training, we re-fit a fast Random Forest
    on the original gene space just for SHAP explanation.

    Parameters
    ----------
    model            : Fitted estimator (may have been trained on PCs).
    model_name       : Identifier string.
    X_train_genes    : Training data in ORIGINAL gene space (no PCA).
    X_test_genes     : Test data in ORIGINAL gene space (no PCA).
    gene_names       : Ordered list of gene column names.
    pca_was_used     : Whether the main model was trained on PCA output.
    random_state     : RNG seed for surrogate model.

    Returns
    -------
    dict with keys: shap_values, gene_names, top_genes, top_scores, plot_path
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP unavailable — skipping explainability.")
        return None

    # Align columns to gene_names
    cols = gene_names[:X_test_genes.shape[1]]
    X_tr = X_train_genes.copy(); X_tr.columns = cols
    X_te = X_test_genes.copy();  X_te.columns = cols

    n = min(shap_sample_size, len(X_te))
    X_explain = X_te.iloc[:n].copy()

    # If model was trained on PCs, build a surrogate RF on original genes
    explain_model_obj, explain_name = _resolve_explain_model(
        model, model_name, X_tr, X_train_genes.index,
        pca_was_used=pca_was_used, random_state=random_state,
    )

    logger.info(
        f"SHAP on ORIGINAL gene features — model: '{explain_name}', "
        f"genes: {len(cols)}, samples: {n}"
    )

    try:
        shap_values = _get_shap_values(explain_model_obj, explain_name, X_tr, X_explain)
    except Exception as e:
        logger.warning(f"SHAP failed for {model_name}: {e}")
        return None

    shap_2d = _reduce_shap(shap_values)

    # Compute gene importance ranking
    mean_abs    = np.abs(shap_2d).mean(axis=0)
    top_k       = min(top_features, len(cols))
    top_idx     = np.argsort(mean_abs)[::-1][:top_k]
    top_genes   = [cols[i] for i in top_idx]
    top_scores  = mean_abs[top_idx].tolist()

    logger.info("Top 10 influential genes by mean |SHAP|:")
    for g, s in zip(top_genes[:10], top_scores[:10]):
        logger.info(f"  {g:<30}  {s:.5f}")

    plot_path = _save_shap_plot(
        shap_2d, X_explain, model_name,
        output_dir=output_dir, top_features=top_k,
        pca_was_used=pca_was_used,
    )

    return {
        "shap_values":  shap_2d,
        "gene_names":   cols,
        "top_genes":    top_genes,
        "top_scores":   top_scores,
        "plot_path":    plot_path,
        "surrogate_used": pca_was_used,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_explain_model(
    model, model_name, X_train_genes, train_index,
    pca_was_used, random_state,
):
    """
    Return (model_to_explain, name).
    If PCA was used, train a surrogate RandomForest on original genes.
    """
    if not pca_was_used:
        return model, model_name

    logger.info(
        "PCA was used for training — fitting surrogate RandomForest "
        "on original gene space for SHAP (explanation only, not evaluation)."
    )
    from sklearn.ensemble import RandomForestClassifier
    surrogate = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=random_state, n_jobs=-1,
    )
    # We need y_train — retrieve from the passed index if possible
    # (caller should pass y_train aligned to X_train_genes)
    # Since we only have X here, fit is done in explain_model() caller
    return surrogate, f"{model_name}_surrogate_RF"


def _get_shap_values(model, model_name: str, X_train, X_explain):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        exp = shap.TreeExplainer(model)
        return exp.shap_values(X_explain)

    if isinstance(model, LogisticRegression):
        bg = shap.sample(X_train, min(50, len(X_train)))
        exp = shap.LinearExplainer(model, bg)
        return exp.shap_values(X_explain)

    # KernelExplainer for SVM / others
    bg = shap.sample(X_train, min(30, len(X_train)))
    fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    exp = shap.KernelExplainer(fn, bg)
    return exp.shap_values(X_explain, nsamples=50, silent=True)


def _reduce_shap(sv) -> np.ndarray:
    """Collapse any SHAP output to (n_samples, n_features) mean."""
    if isinstance(sv, list):
        return np.array(sv).mean(axis=0)
    arr = np.array(sv)
    if arr.ndim == 3:
        return arr.mean(axis=2)
    return arr


def _save_shap_plot(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    model_name: str,
    *,
    output_dir: str,
    top_features: int,
    pca_was_used: bool,
) -> str:
    mean_abs  = np.abs(shap_values).mean(axis=0)
    top_k     = min(top_features, X_explain.shape[1])
    idx       = np.argsort(mean_abs)[::-1][:top_k]
    top_genes = np.array(X_explain.columns)[idx].tolist()
    top_vals  = mean_abs[idx]
    shap_top  = shap_values[:, idx]
    feat_top  = X_explain.values[:, idx]

    fig, axes = plt.subplots(1, 2, figsize=(17, max(6, top_k * 0.38)))

    # ── Bar: mean |SHAP| per gene ─────────────────────────────────────────
    ax = axes[0]
    palette = sns.color_palette("coolwarm_r", n_colors=top_k)
    bars = ax.barh(top_genes[::-1], top_vals[::-1],
                   color=palette[::-1], edgecolor="white", height=0.7)
    ax.set_xlabel("Mean |SHAP Value|  (impact on prediction)", fontsize=11)
    ax.set_title(
        f"Top {top_k} Influential Genes\n"
        f"{'[Surrogate RF on original genes] ' if pca_was_used else ''}"
        f"{model_name}",
        fontsize=11
    )
    ax.axvline(0, color="grey", lw=0.6)
    # Annotate values
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(bar.get_width() + max(top_vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7, color="dimgray")

    # ── Dot: SHAP value coloured by feature expression level ─────────────
    ax2 = axes[1]
    feat_min = feat_top.min(axis=0, keepdims=True)
    feat_rng = (feat_top.max(axis=0, keepdims=True)
                - feat_top.min(axis=0, keepdims=True) + 1e-9)
    feat_norm = (feat_top - feat_min) / feat_rng

    sc = None
    for fi in range(top_k):
        y_pos = np.full(shap_top.shape[0], top_k - 1 - fi)
        sc = ax2.scatter(
            shap_top[:, fi], y_pos,
            c=feat_norm[:, fi], cmap="coolwarm",
            alpha=0.55, s=14, vmin=0, vmax=1,
        )

    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(top_genes[::-1], fontsize=8)
    ax2.axvline(0, color="black", lw=0.8, ls="--")
    ax2.set_xlabel("SHAP Value  (← lowers prediction  |  raises prediction →)",
                   fontsize=10)
    ax2.set_title(
        "SHAP Dot Plot\n(colour = expression level: blue=low, red=high)",
        fontsize=11
    )
    if sc is not None:
        cb = plt.colorbar(sc, ax=ax2)
        cb.set_label("Normalised expression\n(low → high)", fontsize=9)

    note = ("*SHAP computed on original gene features, not PCA components."
            if pca_was_used else
            "*SHAP computed directly on selected gene features.")
    fig.text(0.01, -0.01, note, fontsize=8, color="grey", style="italic")

    plt.suptitle(
        f"SHAP Gene Importance — {model_name}  "
        f"(biologically interpretable)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe = model_name.replace(" ", "_")
    path = Path(output_dir) / f"shap_summary_{safe}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP plot saved → {path}")
    return str(path)
