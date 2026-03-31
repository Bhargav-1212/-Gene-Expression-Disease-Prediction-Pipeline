# Gene Expression Disease Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn%20%7C%20xgboost-orange)
![Interpretability](https://img.shields.io/badge/Interpretability-SHAP-brightgreen)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

A complete, end-to-end, research-grade machine learning pipeline engineered specifically for **Bioinformatics and Computational Biology**. This system ingests highly-dimensional gene expression data (e.g., RNA-Seq or Microarray readings), applies advanced statistical preprocessing, handles severe class imbalances, and trains an ensemble of state-of-the-art predictive models to classify disease states (e.g., Tumor classification, Leukemia subtyping).

Crucially, the pipeline is designed to be a "glass box," utilizing **SHAP (SHapley Additive exPlanations)** and the **MyGene.info REST API** to uncover the biological mechanisms behind its predictions, providing researchers with interpretable candidate biomarkers rather than just statistical accuracies.

---

## 🔬 System Overview & Capabilities

Standard machine learning approaches often fail on genomic data due to the "Curse of Dimensionality" (thousands of genes, but only dozens of patients). This pipeline is purpose-built to overcome this through a robust, multi-stage architecture:

1. **Intelligent Feature Selection:** Uses statistical heuristics (`SelectKBest` with ANOVA F-values, Lasso Regularization, or Variance Thresholding) to isolate only the core predictive genes out of tens of thousands of background variables.
2. **Synthetic Minority Over-sampling (SMOTE):** Clinical datasets are rarely balanced. The pipeline automatically interpolates minority patient classes (e.g., a rare tumor variant) in the generated latent space to neutralize algorithm bias.
3. **Dimensionality Reduction (PCA):** Condenses vast genetic profiles into dense principal components prior to model training, drastically reducing overfitting while retaining over 90% of the dataset's variance.
4. **Automated Hyperparameter Tuning:** Executes high-cardinality `RandomizedSearchCV` or `GridSearchCV` cross-validation across multiple algorithms to autonomously discover optimal algorithmic configurations.
5. **Soft Voted Ensembling:** Instead of relying on a single algorithm, the system fuses predictions from Logistic Regression, Support Vector Machines (SVM), Random Forests, and Gradient Boosting (XGBoost) to achieve a highly resilient consensus prediction.

---

## 🚀 Installation & Setup

Ensure you are running Python 3.9 or higher. 

**1. Clone the repository and install the dependencies:**
```bash
git clone https://github.com/Bhargav-1212/-Gene-Expression-Disease-Prediction-Pipeline.git
cd -Gene-Expression-Disease-Prediction-Pipeline
pip install -r requirements.txt
```

**2. Prepare your Data:**
Place your generic dataset in the `data/` directory. The CSV should contain:
- A clear target class column (e.g., `CLASS` or `disease_label`).
- Thousands of columns containing raw/normalized continuous expression values, where column headers are valid Gene Symbols (e.g., `TP53`), Transcript IDs, or Microarray Probes (e.g., `M23197_at`).

*(Note: You can run `python generate_synthetic_data.py` to test the pipeline on auto-generated dummy genomic noise).*

---

## 🕹️ Command Line Usage

The entire computational pipeline is triggered via a single entry point, governed entirely by strict YAML configuration mappings.

```bash
# Standard Production Run
python train.py --config configs/config.yaml

# Fast Iteration Mode (bypasses the lengthy CV Hyperparameter grid search)
python train.py --config configs/config.yaml --no-tuning

# Completely Offline Mode (Bypasses external API/Gene Database lookups)
python train.py --config configs/config.yaml --no-annotation

# Train exclusively specific ML algorithms based on config definitions
python train.py --config configs/config.yaml --models xgboost svm
```

---

## 📊 Detailed Output Artifacts

Upon successful completion, the pipeline serializes the final model and generates a suite of research-ready, high-resolution diagnostic plots within the `outputs/` directory.

### 1. Clinical Evaluation Metrics
* **`evaluation_report.txt`**  
  A comprehensively structured report designed for researchers. It includes runtime configuration, per-class breakdown metrics (Precision, Recall, F1-Score, Support), and a unified Model Leaderboard tracking the top performers ranked by ROC-AUC and F1.
* **`evaluation_report.json`**  
  The JSON-serialized sibling of the `.txt` report. Perfect for parsing into external web dashboards, CI/CD pipelines, or MongoDB databases.
* **`model.pkl`**  
  The final serialized implementation of the **Best Performing Estimator**. Application developers can dynamically load this Pickle file in production backends to run live genomic inference against new incoming patient blood/tissue samples.

### 2. Algorithmic Diagnostics
* **`model_comparison.png`**  
  A composite bar clustered graph that visually tracks validation performance across the various tested models. Excellent for publication abstracts to summarily justify model selection.
* **`confusion_matrix_<model>.png`**  
  High-contrast seaborn heatmaps constructed for every trained algorithm. They explicitly chart True Positives against False Negatives, allowing researchers to evaluate where the algorithm suffers from distinct clinical "blind spots".
* **`roc_curve_<model>.png` & `pr_curve_<model>.png`**  
  Standard Receiver Operating Characteristic and Precision-Recall Area Under Curve integrations. ROC validates probabilistic discriminative capacity, whereas the PR curve specifically isolates algorithmic robustness on severely imbalanced datasets.

### 3. Biological Interpretability (Explainable AI)
* **`pca_2d_scatter.png`**  
  Projects the N-dimensional expression matrix (post-feature-selection) onto a flat 2D topological plane. Visually demonstrates whether the isolated gene features naturally cluster distinct disease types together.
* **`shap_summary_<model>.png`**  
  Calculates Game-Theoretic SHAP values mapping precisely how individual genes influence clinical predictions. It dictates magnitude (how important the gene is) and directionality (does upregulation of this gene actively push the prediction towards Disease A vs Disease B?).
* **`gene_annotation_importance.png`** 🌟 *(System Highlight)*  
  The true innovation of the pipeline. It maps the aforementioned SHAP statistical impact scores against **biological reality**. The pipeline parses the `MyGene.info` databases to pull the specific gene variants, then graphs them, explicitly stamping disease-associate badges onto genes known to be pathogenic in wider medical literature. 
* **`selected_features.txt`**  
  The flat text array of the Top *N* genes the architecture identified as the primary drivers of the disease state. This acts as the fundamental "Candidate Biomarker" list to be handed back to laboratory technicians for empirical wet-lab validation.

---

## ⚙️ Pipeline Configuration

The `configs/config.yaml` controls all internal architecture mechanisms.

```yaml
data:
  file_path: "data/golub_leukemia.csv"
  target_column: "CLASS"          
  test_size: 0.2                  # 80/20 Train/Test Validation Split

class_imbalance:
  strategy: "smote"               # Applies spatial minority oversampling 

feature_selection:
  method: "selectkbest"           # Applies ANOVA F-Value variance thresholds
  top_k: 50                       # Isolates the top 50 genomic features 

models:
  tuning:
    enabled: true                 # Instructs architecture to optimize sub-algorithms
    cv_folds: 5                   # Straticated 5-fold cross validation split
```

---

## 📜 License & Citation

This codebase is provided as an open-source clinical research tool. When utilizing `gene_annotator.py` outputs in published biological research, please ensure you properly cite the [MyGene.info API](http://mygene.info/) query backbone.
