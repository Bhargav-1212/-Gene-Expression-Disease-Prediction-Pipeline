# Gene Expression Disease Prediction Pipeline v3.0

A **research-grade, fully modular ML system** for disease classification from gene expression data (RNA-seq / microarray). Designed to be accurate, robust, and biologically interpretable.

---

## 🚀 What Makes This Pipeline Production-Ready?

| Feature | Implementation |
|---|---|
| **Gene Annotation Lookup** | Converts ENSG IDs and probe IDs to human-readable gene symbols + real biological disease associations via curated tables and the `mygene.info` API. |
| **SHAP Interpretability** | Explainability algorithms always use the original gene-space features (not PCA dimensions) so doctors can understand *which specific genes* caused a prediction. |
| **Class Imbalance Handling** | Automatically applies **SMOTE** (Synthetic Minority Over-sampling Technique) and class weighting to ensure rare diseases aren't ignored by the model. |
| **Ensemble Modeling** | Trains 4 advanced algorithms (Logistic Regression, SVM, Random Forest, XGBoost) concurrently, then fuses them into a highly robust Soft-Voting Ensemble classifier. |
| **Rich Evaluation Reports** | Dumps both JSON (for machines/dashboards) and TXT (for human researchers) reports with full metrics, leaderboard rankings, and interpretation guides. |

---

## 💻 Quick Start Guide

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Prepare Your Data**
Place your real CSV dataset into the `data/` folder. Your CSV must have:
- A `disease_label` or `CLASS` column (configure this in `config.yaml`).
- All other columns must be numeric gene expression values.

*(Alternatively, run `python generate_synthetic_data.py` to test the pipeline with dummy data).*

**3. Run the Full ML Pipeline**
```bash
python train.py --config configs/config.yaml
```

**4. Additional Run Modes:**
```bash
# Fast mode (skips heavy hyperparameter tuning)
python train.py --config configs/config.yaml --no-tuning

# Offline mode (skips the mygene.info internet API lookup)
python train.py --config configs/config.yaml --no-annotation

# Train specific algorithms only
python train.py --config configs/config.yaml --models xgboost random_forest
```

---

## 📊 Visual Outputs (Plots & Charts Generated)

This pipeline automatically generates several high-quality, publication-ready visualizations saved directly to your `outputs/` folder. **These pictures are required for proper clinical and data science evaluation:**

### 1. Model Performance Plots
* **`model_comparison.png`**
  A stylized bar chart leaderboard comparing Accuracy, F1 Score, and ROC-AUC across all trained models (SVM, XGBoost, Ensemble, etc.). Use this to instantly see the winner.
* **`confusion_matrix_<model>.png`**
  Heatmaps for each model showing exactly where the model got predictions right vs. where it got confused (e.g. predicting Tumor_A when the patient actually had Tumor_B).
* **`roc_curve_<model>.png` & `pr_curve_<model>.png`**
  Receiver Operating Characteristic and Precision-Recall curves. These graphs demonstrate the trade-off between sensitivity and specificity across different threshold values. Crucial for assessing performance on imbalanced medical datasets.

### 2. Biology & Data Interpretability Plots
* **`pca_2d_scatter.png`**
  A 2D scatter plot generated using Principal Component Analysis. It condenses thousands of genes down to an X/Y axis so you can visually see if the disease classes naturally separate from healthy patients in the raw data.
* **`shap_summary_<model>.png`**
  A standard SHAP importance plot showing the top 20 most impactful genes. It breaks down how pushing a gene's expression level high/low shifts the model's disease prediction.
* **`gene_annotation_importance.png`** ⭐ *(Unique to this pipeline)*
  A customized horizontal bar chart merging machine learning math with biology. It charts the SHAP importance alongside actual disease associations parsed from global biological databases, instantly highlighting which top features are known cancer/disease biomarkers.

---

## ⚙️ Configuration File (`configs/config.yaml`)

Control the absolute entirety of the pipeline without touching Python code:

```yaml
data:
  file_path: "data/golub_leukemia.csv"
  target_column: "CLASS"

class_imbalance:
  strategy: "smote"     # Select: smote | class_weight | both | none

feature_selection:
  method: "selectkbest" # Select: variance | selectkbest | lasso | all
  top_k: 50             # How many top genes to filter down to

dimensionality_reduction:
  enabled: true         # Condense features using PCA before training

models:
  tuning:
    enabled: true       # Run heavy GridSearch optimization
    cv_folds: 5         # 5-fold cross validation
```

---

## 📦 What gets saved? (Outputs Directory)

After running, check the `outputs/` folder for your deliverables:

| File | Description |
|---|---|
| `model.pkl` | **Your production-ready model.** Developers can load this pickle file in their own apps to make predictions on new patient data! |
| `selected_features.txt` | The list of the exact genes (e.g., top 50) the model ended up using. |
| `evaluation_report.txt` | The human-readable summary containing metrics, leaderboards, and the gene biological lookup tables. |
| `evaluation_report.json`| The exact same report, but formatted for developers building frontend dashboards. |
| `pipeline.log` | Raw execution logs (useful for debugging). |

---

## 🧬 Interpreting the SHAP Biology Results

When you read the generated reports or view the SHAP picture outputs, use this guide:
- **Mean \|SHAP\| Score:** The average magnitude of a gene's impact on predictions. High score = major factor.
- **Positive SHAP:** High expression of this gene pushes the prediction *toward* the disease class.
- **Negative SHAP:** High expression pushes the prediction *away* from the disease.
- **Biological Validation:** If the pipeline successfully tags a top SHAP gene with known disease associations (e.g. "Ovarian Cancer"), it acts as strong validation that the model has learned genuine biological reality, not just statistical noise.
