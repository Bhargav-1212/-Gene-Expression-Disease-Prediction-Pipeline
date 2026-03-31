"""
reporter.py  v3.0
-----------------
Saves model, selected features, and a research-grade evaluation report.

Report sections:
  1. Pipeline configuration summary
  2. Best model — metrics + per-class breakdown
  3. Full model leaderboard
  4. Top influential genes (with symbol, description, disease associations, SHAP score)
  5. Biological interpretation notes
  6. Reproducibility block (random seed, versions)
"""

import json
import logging
import pickle
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_model(model, model_name: str, output_dir: str = "outputs") -> str:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model_name": model_name, "estimator": model}, f)
    logger.info(f"Model saved → {path}")
    return str(path)


def save_features(selected_genes: List[str], output_dir: str = "outputs") -> str:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "selected_features.txt"
    with open(path, "w", encoding="utf-8") as f:
        for g in selected_genes:
            f.write(g + "\n")
    logger.info(f"Selected features ({len(selected_genes)}) saved → {path}")
    return str(path)


def save_report(
    results: Dict[str, Dict[str, Any]],
    comparison_df: pd.DataFrame,
    best_model_name: str,
    config: dict,
    *,
    shap_result: Optional[Dict[str, Any]] = None,
    enriched_genes: Optional[List[Dict]] = None,
    output_dir: str = "outputs",
    fmt: str = "json",
) -> str:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    report = _build_report(
        results, comparison_df, best_model_name, config,
        shap_result, enriched_genes,
    )

    # Always save both formats
    json_path = out / "evaluation_report.json"
    txt_path  = out / "evaluation_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_serial)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_fmt_txt(report))

    logger.info(f"Report saved → {json_path}  +  {txt_path}")
    return str(json_path if fmt == "json" else txt_path)


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------

def _build_report(results, comparison_df, best_model_name, config,
                  shap_result, enriched_genes):
    scalar_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    best_r      = results[best_model_name]
    fs_cfg      = config["feature_selection"]
    imb_cfg     = config.get("class_imbalance", {})
    ens_cfg     = config["models"].get("ensemble", {})

    # Gene importance section
    gene_section = None
    if shap_result and enriched_genes:
        gene_section = {
            "method":         "SHAP (on original gene features — biologically interpretable)",
            "surrogate_used": shap_result.get("surrogate_used", False),
            "top_genes":      enriched_genes,
        }

    return {
        "pipeline_version": "3.0.0",
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "reproducibility": {
            "python_version": platform.python_version(),
            "platform":       platform.system(),
            "random_seed":    config["data"].get("random_state", 42),
        },
        "configuration": {
            "normalization":           config["preprocessing"]["normalization"],
            "imbalance_strategy":      imb_cfg.get("strategy", "none"),
            "feature_selection_method": fs_cfg["method"],
            "top_k_genes":             fs_cfg["top_k"],
            "pca_enabled":             config["dimensionality_reduction"]["enabled"],
            "shap_on_original_genes":  config["explainability"].get("use_original_features", True),
            "ensemble_enabled":        ens_cfg.get("enabled", False),
            "hyperparameter_tuning":   config["models"]["tuning"]["enabled"],
            "cv_folds":                config["models"]["tuning"]["cv_folds"],
            "models_trained":          list(results.keys()),
        },
        "best_model": {
            "name":                   best_model_name,
            "accuracy":               best_r["accuracy"],
            "precision":              best_r["precision"],
            "recall":                 best_r["recall"],
            "f1_score":               best_r["f1"],
            "roc_auc":                best_r["roc_auc"],
            "classification_report":  best_r["classification_report"],
        },
        "leaderboard":    comparison_df.to_dict(orient="records"),
        "all_model_metrics": {
            name: {k: r[k] for k in scalar_keys}
            for name, r in results.items()
        },
        "gene_importance": gene_section,
    }


# ---------------------------------------------------------------------------
# Human-readable TXT report
# ---------------------------------------------------------------------------

def _fmt_txt(r: dict) -> str:
    W   = 78
    SEP = "═" * W
    sep = "─" * W
    cfg = r["configuration"]
    bm  = r["best_model"]
    rep = r["reproducibility"]

    def _centre(txt): return txt.center(W)
    def _kv(k, v):    return f"  {k:<36} {v}"

    lines = [
        SEP,
        _centre("GENE EXPRESSION DISEASE PREDICTION PIPELINE"),
        _centre("Research-Grade Evaluation Report  •  v3.0.0"),
        SEP,
        "",
        "  ┌─ RUN INFO " + "─" * (W - 13) + "┐",
        _kv("Timestamp:",          r["timestamp"]),
        _kv("Python version:",     rep["python_version"]),
        _kv("Platform:",           rep["platform"]),
        _kv("Random seed:",        rep["random_seed"]),
        "  └" + "─" * (W - 3) + "┘",
        "",
        "  ┌─ PIPELINE CONFIGURATION " + "─" * (W - 28) + "┐",
        _kv("Normalisation:",      cfg["normalization"]),
        _kv("Imbalance strategy:", cfg["imbalance_strategy"]),
        _kv("Feature selection:",  f"{cfg['feature_selection_method']}  (top_k = {cfg['top_k_genes']})"),
        _kv("PCA (model only):",   "enabled" if cfg["pca_enabled"] else "disabled"),
        _kv("SHAP on orig. genes:",str(cfg["shap_on_original_genes"])),
        _kv("Ensemble:",           "enabled" if cfg["ensemble_enabled"] else "disabled"),
        _kv("Hyperparameter tune:",str(cfg["hyperparameter_tuning"])),
        _kv("CV folds:",           cfg["cv_folds"]),
        _kv("Models trained:",     ", ".join(cfg["models_trained"])),
        "  └" + "─" * (W - 3) + "┘",
        "",
        SEP,
        _centre(f"🏆  BEST MODEL :  {bm['name'].upper()}"),
        SEP,
        _kv("Accuracy:",   f"{bm['accuracy']:.4f}"),
        _kv("Precision:",  f"{bm['precision']:.4f}"),
        _kv("Recall:",     f"{bm['recall']:.4f}"),
        _kv("F1 Score:",   f"{bm['f1_score']:.4f}"),
        _kv("ROC-AUC:",    f"{bm['roc_auc']}" if bm["roc_auc"] else "N/A"),
        "",
        "  Per-class breakdown:",
        sep,
        bm["classification_report"],
    ]

    # Leaderboard
    lines += [SEP, _centre("MODEL LEADERBOARD"), SEP]
    hdr = f"  {'Rank':<5} {'Model':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'ROC-AUC'}"
    lines += [hdr, "  " + sep]
    for row in r["leaderboard"]:
        auc = f"{row['ROC-AUC']:.4f}" if row["ROC-AUC"] else "  N/A  "
        marker = " ← BEST" if row["Rank"] == 1 else ""
        lines.append(
            f"  {row['Rank']:<5} {row['Model']:<25} {row['Accuracy']:<10.4f} "
            f"{row['Precision']:<11.4f} {row['Recall']:<8.4f} {row['F1']:<8.4f} {auc}{marker}"
        )

    # Gene importance
    gi = r.get("gene_importance")
    if gi and gi.get("top_genes"):
        lines += [
            "",
            SEP,
            _centre("TOP INFLUENTIAL GENES — SHAP ANALYSIS"),
            _centre("(Biologically interpretable — original gene features)"),
            SEP,
        ]
        if gi.get("surrogate_used"):
            lines.append(
                "  * Surrogate Random Forest fitted on original gene space "
                "(pipeline trained on PCA)."
            )
            lines.append("  * SHAP scores reflect gene-level importance for disease prediction.")
            lines.append("")

        lines.append(
            f"  {'Rk':<4} {'Symbol':<12} {'Gene ID':<24} {'|SHAP|':<10} "
            f"{'Description & Disease Associations'}"
        )
        lines.append("  " + "─" * (W - 2))

        for g in gi["top_genes"]:
            diseases = "; ".join(g["diseases"][:2]) if g["diseases"] else "No curated association"
            if len(g.get("diseases", [])) > 2:
                diseases += f" (+{len(g['diseases'])-2})"
            desc_str = g["description"]
            if desc_str and desc_str != "—" and len(desc_str) > 35:
                desc_str = desc_str[:32] + "…"

            lines.append(
                f"  {g['rank']:<4} {g['symbol']:<12} {g['gene_id']:<24} "
                f"{g['mean_abs_shap']:<10.6f} {desc_str}"
            )
            if diseases and diseases != "No curated association":
                lines.append(f"  {'':<50} ↳ {diseases}")

        lines += [
            "",
            "  INTERPRETATION GUIDE:",
            "  • Mean |SHAP| score = average magnitude of a gene's impact on predictions.",
            "  • Higher score → gene expression strongly influences disease classification.",
            "  • Positive SHAP → gene expression pushes prediction toward a disease class.",
            "  • Negative SHAP → gene expression pushes prediction away from that class.",
            "  • Genes with known disease associations validate biological plausibility.",
            "  • Use these genes as candidate biomarkers for downstream wet-lab validation.",
        ]

    lines += ["", SEP,
              _centre("END OF REPORT"),
              SEP, ""]
    return "\n".join(lines)


def _serial(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return str(obj)
