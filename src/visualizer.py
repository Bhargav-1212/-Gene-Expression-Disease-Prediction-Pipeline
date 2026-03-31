"""
visualizer.py
-------------
Generates and saves all pipeline visualizations:
  • Confusion matrix heatmap
  • ROC curve(s)
  • Precision-Recall curve(s)
  • Feature importance bar chart (tree models)
  • Model comparison table plot
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Global style
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = sns.color_palette("tab10")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_results(
    results: Dict[str, Dict[str, Any]],
    comparison_df: pd.DataFrame,
    selected_genes: List[str],
    fitted_models: Dict[str, Any],
    *,
    output_dir: str = "outputs",
) -> Dict[str, str]:
    """
    Generate all evaluation plots and save them to output_dir.

    Returns
    -------
    paths : {plot_name: file_path}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    # ── Per-model plots ───────────────────────────────────────────────────
    for name, r in results.items():
        safe = name.replace(" ", "_")

        p = _plot_confusion_matrix(r, out, safe)
        paths[f"confusion_matrix_{safe}"] = p

        if r.get("roc_data"):
            p = _plot_roc_curve(r, out, safe)
            paths[f"roc_curve_{safe}"] = p

        if r.get("pr_data"):
            p = _plot_pr_curve(r, out, safe)
            paths[f"pr_curve_{safe}"] = p

    # ── Feature importance (best tree model) ─────────────────────────────
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    if hasattr(best_model, "feature_importances_"):
        p = _plot_feature_importance(
            best_model, selected_genes, best_model_name, out
        )
        paths["feature_importance"] = p

    # ── Model comparison table ────────────────────────────────────────────
    p = _plot_comparison_table(comparison_df, out)
    paths["model_comparison"] = p

    logger.info(f"All plots saved to '{output_dir}/'")
    return paths


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    r: dict,
    out: Path,
    safe_name: str,
) -> str:
    cm     = r["confusion_matrix"]
    labels = r["class_labels"]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2),
                                    max(4, len(labels) * 1.0)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(f"Confusion Matrix — {r['model_name']}", fontsize=13)
    plt.tight_layout()

    path = out / f"confusion_matrix_{safe_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_roc_curve(r: dict, out: Path, safe_name: str) -> str:
    roc = r["roc_data"]
    fig, ax = plt.subplots(figsize=(7, 6))

    if roc["binary"]:
        ax.plot(roc["fpr"], roc["tpr"],
                label=f"AUC = {roc['auc']:.3f}", lw=2, color=PALETTE[0])
    else:
        for i, (cls, d) in enumerate(roc["curves"].items()):
            ax.plot(d["fpr"], d["tpr"],
                    label=f"Class {cls} (AUC={d['auc']:.3f})",
                    lw=1.5, color=PALETTE[i % 10])

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve — {r['model_name']}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    path = out / f"roc_curve_{safe_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_pr_curve(r: dict, out: Path, safe_name: str) -> str:
    pr = r["pr_data"]
    fig, ax = plt.subplots(figsize=(7, 6))

    if pr["binary"]:
        ax.plot(pr["recall"], pr["precision"],
                label=f"AP = {pr['ap']:.3f}", lw=2, color=PALETTE[1])
    else:
        for i, (cls, d) in enumerate(pr["curves"].items()):
            ax.plot(d["recall"], d["precision"],
                    label=f"Class {cls} (AP={d['ap']:.3f})",
                    lw=1.5, color=PALETTE[i % 10])

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {r['model_name']}", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    path = out / f"pr_curve_{safe_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_feature_importance(
    model,
    gene_names: List[str],
    model_name: str,
    out: Path,
    top_n: int = 30,
) -> str:
    importances = model.feature_importances_
    # Align lengths (after PCA, gene_names may be PC labels)
    if len(importances) != len(gene_names):
        gene_names = [f"Feature_{i}" for i in range(len(importances))]

    idx = np.argsort(importances)[::-1][:top_n]
    top_genes  = [gene_names[i] for i in idx]
    top_import = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    colors = sns.color_palette("viridis", n_colors=top_n)
    ax.barh(top_genes[::-1], top_import[::-1], color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(
        f"Top {top_n} Feature Importances — {model_name}", fontsize=13
    )
    plt.tight_layout()

    path = out / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_comparison_table(df: pd.DataFrame, out: Path) -> str:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(df) * 0.9 + 2)))

    # ── Bar chart ─────────────────────────────────────────────────────────
    ax = axes[0]
    x  = np.arange(len(df))
    w  = 0.15
    for i, metric in enumerate(metrics):
        vals = df[metric].fillna(0).values
        ax.bar(x + i * w, vals, w, label=metric, color=PALETTE[i])

    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim([0, 1.12])
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — All Metrics", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7)

    # ── Table ─────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")
    display_df = df.copy()
    for m in metrics:
        display_df[m] = display_df[m].apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "N/A"
        )

    cell_colours = [["#f0f4ff"] * len(display_df.columns)] * len(display_df)
    # Highlight best row
    cell_colours[0] = ["#d4edda"] * len(display_df.columns)

    tbl = ax2.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        cellColours=cell_colours,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax2.set_title("Leaderboard (Best = green row)", fontsize=12, pad=20)

    plt.tight_layout()
    path = out / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Gene annotation plot (added in v3)
# ---------------------------------------------------------------------------

def plot_gene_annotation(
    enriched_genes: List[Dict],
    *,
    output_dir: str = "outputs",
    top_n: int = 20,
) -> str:
    """
    Plot SHAP importance with gene symbols + disease annotation badges.
    Returns path to saved PNG.
    """
    import matplotlib.patches as mpatches

    top = enriched_genes[:top_n]
    if not top:
        return ""

    symbols   = [g["symbol"] for g in top]
    scores    = [g["mean_abs_shap"] for g in top]
    has_assoc = [bool(g.get("diseases")) for g in top]

    fig, ax = plt.subplots(figsize=(13, max(6, top_n * 0.42)))
    colors  = ["#2196F3" if h else "#90CAF9" for h in has_assoc]
    bars    = ax.barh(symbols[::-1], scores[::-1],
                      color=colors[::-1], edgecolor="white", height=0.65)

    # Annotate bars with disease count
    for bar, g in zip(bars, top[::-1]):
        nd = len(g.get("diseases", []))
        if nd:
            ax.text(bar.get_width() + max(scores) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"  {nd} disease assoc.", va="center",
                    fontsize=7.5, color="#1565C0")

    ax.set_xlabel("Mean |SHAP Value| — Predictive Importance", fontsize=11)
    ax.set_title(
        "Top Influential Genes with Disease Associations\n"
        "(blue = known disease gene · light = unannotated)",
        fontsize=12,
    )
    ax.axvline(0, color="grey", lw=0.5)

    legend_patches = [
        mpatches.Patch(color="#2196F3", label="Known disease association"),
        mpatches.Patch(color="#90CAF9", label="No curated association"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()

    path = Path(output_dir) / "gene_annotation_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Gene annotation plot saved → {path}")
    return str(path)
