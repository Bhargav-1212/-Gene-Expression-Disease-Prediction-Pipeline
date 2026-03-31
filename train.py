"""
train.py  v3.0
--------------
CLI entry point — Gene Expression Disease Prediction Pipeline.

v3 additions:
  • ENSG ID → gene symbol + disease association mapping (gene_annotator.py)
  • Annotated SHAP report with biological interpretation
  • Gene annotation importance plot
  • Both JSON + TXT reports always saved
  • --no-annotation flag to skip network lookup

Usage
-----
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --no-tuning
    python train.py --config configs/config.yaml --no-annotation
    python train.py --config configs/config.yaml --models svm xgboost
"""

import argparse
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils                    import setup_logging, load_config, set_global_seed, print_banner
from src.data_loader              import load_data, preprocess_data
from src.feature_selection        import select_features
from src.dimensionality_reduction import apply_pca, plot_pca_2d
from src.imbalance                import apply_imbalance_strategy, get_class_weight_param
from src.trainer                  import train_models
from src.models                   import build_ensemble
from src.evaluator                import evaluate_model, compare_models
from src.visualizer               import plot_results, plot_gene_annotation
from src.explainer                import explain_model
from src.gene_annotator           import annotate_genes, enrich_shap_results, format_gene_table
from src.reporter                 import save_model, save_features, save_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: dict, cli_overrides: dict = None) -> None:
    t_total = time.time()

    if cli_overrides:
        if cli_overrides.get("no_tuning"):
            cfg["models"]["tuning"]["enabled"] = False
        if cli_overrides.get("models"):
            cfg["models"]["train"] = cli_overrides["models"]

    random_state   = cfg["data"].get("random_state", 42)
    output_dir     = cfg["outputs"]["dir"]
    use_annotation = not (cli_overrides or {}).get("no_annotation", False)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    set_global_seed(random_state)

    # ── 1. Load ───────────────────────────────────────────────────────────
    logger.info("STEP 1 ── Data Loading")
    X, y = load_data(
        file_path=cfg["data"]["file_path"],
        target_column=cfg["data"]["target_column"],
    )

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    logger.info("STEP 2 ── Preprocessing")
    prep = preprocess_data(
        X, y,
        normalization=cfg["preprocessing"]["normalization"],
        handle_missing=cfg["preprocessing"]["handle_missing"],
        encode_labels=cfg["preprocessing"]["encode_labels"],
        test_size=cfg["data"]["test_size"],
        random_state=random_state,
    )
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    label_encoder   = prep["label_encoder"]

    # ── 3. Feature Selection ──────────────────────────────────────────────
    logger.info("STEP 3 ── Feature Selection")
    fs_cfg = cfg["feature_selection"]
    X_train_sel, X_test_sel, selected_genes = select_features(
        X_train, y_train, X_test,
        method=fs_cfg["method"],
        top_k=fs_cfg["top_k"],
        variance_threshold=fs_cfg["variance_threshold"],
        selectkbest_score=fs_cfg["selectkbest_score"],
        lasso_alpha=fs_cfg["lasso_alpha"],
    )

    # ── 4. Gene Annotation ────────────────────────────────────────────────
    logger.info("STEP 4 ── Gene Annotation (ENSG → Symbol + Disease Associations)")
    ann_cfg     = cfg.get("annotation", {})
    annotations = annotate_genes(
        selected_genes,
        use_api=ann_cfg.get("use_api", True) and use_annotation,
        api_timeout=ann_cfg.get("api_timeout", 10),
    )
    # Summary log
    known = sum(1 for g in selected_genes
                if annotations.get(g, {}).get("diseases"))
    logger.info(
        f"  {known}/{len(selected_genes)} selected genes have known disease associations"
    )

    # ── 5. Class Imbalance Handling ───────────────────────────────────────
    logger.info("STEP 5 ── Class Imbalance Handling")
    imb_cfg   = cfg.get("class_imbalance", {})
    imb_strat = imb_cfg.get("strategy", "none")
    cw_param  = get_class_weight_param(imb_strat)

    X_train_bal, y_train_bal = apply_imbalance_strategy(
        X_train_sel, y_train,
        strategy=imb_strat,
        smote_k_neighbors=imb_cfg.get("smote_k_neighbors", 5),
        random_state=random_state,
    )
    X_train_genes_for_shap = X_train_bal
    X_test_genes_for_shap  = X_test_sel

    # ── 6. PCA (model training only) ──────────────────────────────────────
    dr_cfg   = cfg["dimensionality_reduction"]
    pca_used = dr_cfg["enabled"]

    if pca_used:
        logger.info("STEP 6 ── PCA  (model training only — SHAP uses original genes)")
        X_train_model, X_test_model, _ = apply_pca(
            X_train_bal, X_test_sel,
            n_components=dr_cfg["n_components"],
            random_state=random_state,
        )
        plot_pca_2d(X_train_sel, y_train,
                    output_dir=output_dir, label_encoder=label_encoder)
        model_feature_names = list(X_train_model.columns)
    else:
        logger.info("STEP 6 ── PCA skipped")
        X_train_model       = X_train_bal
        X_test_model        = X_test_sel
        model_feature_names = selected_genes

    # ── 7. Train Base Models ──────────────────────────────────────────────
    logger.info("STEP 7 ── Model Training")
    tune_cfg = cfg["models"]["tuning"]
    fitted_models = train_models(
        X_train_model, y_train_bal,
        model_names=cfg["models"]["train"],
        tuning_enabled=tune_cfg["enabled"],
        tuning_method=tune_cfg["method"],
        n_iter=tune_cfg["n_iter"],
        cv_folds=tune_cfg["cv_folds"],
        scoring=tune_cfg["scoring"],
        n_jobs=tune_cfg["n_jobs"],
        random_state=random_state,
        class_weight=cw_param,
    )

    # ── 8. Ensemble ───────────────────────────────────────────────────────
    ens_cfg = cfg["models"].get("ensemble", {})
    if ens_cfg.get("enabled", False):
        logger.info("STEP 8 ── Building Ensemble")
        ensemble = build_ensemble(fitted_models, voting=ens_cfg.get("voting", "soft"))
        from sklearn.preprocessing import LabelEncoder as _LE
        from xgboost import XGBClassifier as _XGB
        _le   = _LE()
        _yi   = pd.Series(_le.fit_transform(y_train_bal))
        y_ens = _yi if any(isinstance(m, _XGB) for m in fitted_models.values()) else y_train_bal
        ensemble.fit(X_train_model, y_ens)
        fitted_models["ensemble"] = ensemble
        logger.info("  ✓ Ensemble fitted")
    else:
        logger.info("STEP 8 ── Ensemble skipped")

    # ── 9. Evaluate ───────────────────────────────────────────────────────
    logger.info("STEP 9 ── Evaluation")
    eval_cfg = cfg["evaluation"]
    results  = {}
    for name, model in fitted_models.items():
        results[name] = evaluate_model(
            model, name, X_test_model, y_test,
            average=eval_cfg["average"],
            label_encoder=label_encoder,
        )

    comparison_df   = compare_models(results)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model      = fitted_models[best_model_name]
    _print_leaderboard(comparison_df)

    # ── 10. Visualisations ────────────────────────────────────────────────
    if cfg["outputs"]["save_plots"]:
        logger.info("STEP 10 ── Generating Plots")
        plot_results(results, comparison_df, model_feature_names, fitted_models,
                     output_dir=output_dir)

    # ── 11. SHAP on original gene features ───────────────────────────────
    exp_cfg     = cfg["explainability"]
    shap_result = None
    if exp_cfg["enabled"]:
        logger.info("STEP 11 ── SHAP Explainability (original gene space)")

        shap_model_name = best_model_name
        shap_model      = best_model
        if best_model_name == "ensemble":
            base_df         = comparison_df[comparison_df["Model"] != "ensemble"]
            shap_model_name = base_df.iloc[0]["Model"]
            shap_model      = fitted_models[shap_model_name]
            logger.info(f"  Ensemble is best — using '{shap_model_name}' for SHAP.")

        if pca_used:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder as _LE2
            logger.info("  Fitting gene-space surrogate RF for SHAP …")
            surrogate = RandomForestClassifier(
                n_estimators=100, max_depth=15,
                random_state=random_state, n_jobs=-1,
            )
            _le2 = _LE2()
            surrogate.fit(X_train_genes_for_shap,
                          pd.Series(_le2.fit_transform(y_train_bal)))
            shap_model      = surrogate
            shap_model_name = f"{shap_model_name} (gene-space surrogate)"

        shap_result = explain_model(
            shap_model, shap_model_name,
            X_train_genes_for_shap, X_test_genes_for_shap,
            gene_names=selected_genes,
            output_dir=output_dir,
            shap_sample_size=exp_cfg["shap_sample_size"],
            top_features=exp_cfg["top_features"],
            pca_was_used=False,
            random_state=random_state,
        )

    # ── 12. Enrich SHAP results with gene annotations ─────────────────────
    enriched_genes = None
    if shap_result and annotations:
        logger.info("STEP 12 ── Enriching SHAP results with gene annotations")
        enriched_genes = enrich_shap_results(
            shap_result["top_genes"],
            shap_result["top_scores"],
            annotations,
        )

        # Print annotated gene table to log
        table_str = format_gene_table(enriched_genes, max_rows=20)
        logger.info("\n  TOP INFLUENTIAL GENES WITH ANNOTATIONS:" + table_str)

        # Gene annotation importance plot
        if cfg["outputs"]["save_plots"]:
            plot_gene_annotation(
                enriched_genes,
                output_dir=output_dir,
                top_n=exp_cfg.get("top_features", 20),
            )

    # ── 13. Save outputs ──────────────────────────────────────────────────
    logger.info("STEP 13 ── Saving Outputs")
    if cfg["outputs"]["save_model"]:
        save_model(best_model, best_model_name, output_dir)
    if cfg["outputs"]["save_features"]:
        save_features(selected_genes, output_dir)
    if cfg["outputs"]["save_report"]:
        save_report(
            results, comparison_df, best_model_name, cfg,
            shap_result=shap_result,
            enriched_genes=enriched_genes,
            output_dir=output_dir,
            fmt=cfg["outputs"]["report_format"],
        )

    elapsed = time.time() - t_total
    logger.info(f"\n✅  Pipeline complete in {elapsed:.1f}s")
    logger.info(f"   All outputs saved to '{output_dir}/'")
    _print_output_summary(output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_leaderboard(df: pd.DataFrame) -> None:
    logger.info("\n" + "=" * 68)
    logger.info("  MODEL LEADERBOARD")
    logger.info("=" * 68)
    logger.info("\n" + df.to_string(index=False))
    best = df.iloc[0]
    auc  = f"{best['ROC-AUC']:.4f}" if best["ROC-AUC"] else "N/A"
    logger.info(f"\n  🏆  BEST MODEL  :  {best['Model'].upper()}")
    logger.info(f"      F1 = {best['F1']:.4f}  |  ROC-AUC = {auc}  |  Accuracy = {best['Accuracy']:.4f}")
    logger.info("=" * 68)


def _print_output_summary(output_dir: str) -> None:
    out   = Path(output_dir)
    files = sorted(out.glob("*"))
    logger.info(f"\n  Output files ({len(files)}):")
    for f in files:
        size = f.stat().st_size
        sz   = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
        logger.info(f"    {f.name:<45} {sz:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Gene Expression Disease Prediction Pipeline v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with gene annotation
  python train.py --config configs/config.yaml

  # Fast mode — no hyperparameter tuning
  python train.py --config configs/config.yaml --no-tuning

  # Offline mode — skip mygene.info API lookup
  python train.py --config configs/config.yaml --no-annotation

  # Train specific models only
  python train.py --config configs/config.yaml --models svm xgboost

  # Combine flags
  python train.py --config configs/config.yaml --no-tuning --no-annotation
        """,
    )
    ap.add_argument("--config",        "-c", required=True,
                    help="Path to YAML configuration file")
    ap.add_argument("--no-tuning",     action="store_true",
                    help="Disable hyperparameter tuning (faster runs)")
    ap.add_argument("--no-annotation", action="store_true",
                    help="Skip mygene.info API lookup (fully offline)")
    ap.add_argument("--models",        nargs="+",
                    choices=["logistic_regression","svm","random_forest","xgboost"],
                    help="Override which models to train")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"),
                  log_file=log_cfg.get("log_file"))
    print_banner()
    logger.info(f"Pipeline v3.0  |  Config: {args.config}")
    run_pipeline(
        cfg,
        cli_overrides={
            "no_tuning":     args.no_tuning,
            "no_annotation": args.no_annotation,
            "models":        args.models,
        },
    )
