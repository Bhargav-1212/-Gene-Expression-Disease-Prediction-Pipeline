"""
generate_synthetic_data.py  v2
------------------------------
Creates a realistic gene-expression dataset that includes real cancer gene
symbols (BRCA1, TP53, EGFR …) alongside ENSG IDs so the annotation module
has meaningful genes to resolve and display.

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --samples 500 --genes 2000 --classes 3
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Real disease-relevant gene symbols to embed in the dataset
CANCER_GENES = [
    "BRCA1","BRCA2","TP53","PTEN","RB1","APC","VHL","MLH1","MSH2","CDKN2A",
    "KRAS","NRAS","BRAF","MYC","MYCN","EGFR","ERBB2","ALK","MET","RET",
    "BCL2","MDM2","CDK4","CCND1","FLT3","JAK2","ABL1","KIT","CTNNB1",
    "NOTCH1","FBXW7","IDH1","IDH2","DNMT3A","TET2","EZH2","ARID1A",
    "PIK3CA","AKT1","MTOR","ATM","CHEK2","PALB2","RAD51","STK11",
    "CDKN1A","CDKN1B","CDK6","CCNE1","TERT","SMAD4","NF1","RUNX1",
]

def generate(
    n_samples:    int  = 300,
    n_genes:      int  = 1000,
    n_classes:    int  = 3,
    n_informative:int  = 150,
    random_state: int  = 42,
    output_path:  str  = "data/gene_expression.csv",
) -> str:
    rng = np.random.default_rng(random_state)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build gene name list: real symbols first, then ENSG IDs
    n_real    = min(len(CANCER_GENES), n_genes)
    real_syms = CANCER_GENES[:n_real]
    ensg_ids  = [f"ENSG{i:011d}" for i in range(1, n_genes - n_real + 1)]
    gene_names = real_syms + ensg_ids          # total = n_genes

    # Class labels
    class_names = {0: "Healthy", 1: "Tumor_TypeA", 2: "Tumor_TypeB",
                   3: "Metastatic", 4: "Benign"}
    raw_labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    raw_labels = np.concatenate(
        [raw_labels, np.arange(n_samples - len(raw_labels))]
    )
    rng.shuffle(raw_labels)
    labels = np.array([class_names.get(l, f"Class_{l}") for l in raw_labels])

    # Base expression: log-normal (mimics RNA-seq count distributions)
    X = rng.lognormal(mean=4.0, sigma=1.5, size=(n_samples, n_genes))

    # Inject class-specific signal into informative genes
    # Prioritise real cancer genes as informative
    n_real_informative = min(n_real, n_informative // 2)
    real_info_idx  = np.arange(n_real_informative)
    ensg_info_idx  = rng.choice(
        np.arange(n_real, n_genes),
        size=n_informative - n_real_informative,
        replace=False,
    )
    informative_idx = np.concatenate([real_info_idx, ensg_info_idx])

    for cls in range(n_classes):
        mask   = raw_labels == cls
        signal = rng.normal(loc=cls * 2.8, scale=1.0,
                            size=(mask.sum(), len(informative_idx)))
        X[np.ix_(mask, informative_idx)] += signal

    # ~2% missing values
    X[rng.random(X.shape) < 0.02] = np.nan

    df = pd.DataFrame(X, columns=gene_names)
    df.insert(0, "disease_label", labels)
    df.to_csv(output_path, index=False)

    print(f"[OK] Dataset → {output_path}")
    print(f"     Samples: {n_samples}  |  Genes: {n_genes}  "
          f"(incl. {n_real} real cancer gene symbols)")
    print(f"     Classes: {n_classes}  |  Informative: {n_informative}  |  Missing: ~2%")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples",     type=int, default=300)
    ap.add_argument("--genes",       type=int, default=1000)
    ap.add_argument("--classes",     type=int, default=3)
    ap.add_argument("--informative", type=int, default=150)
    ap.add_argument("--output",      type=str, default="data/gene_expression.csv")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()
    generate(
        n_samples=args.samples, n_genes=args.genes,
        n_classes=args.classes, n_informative=args.informative,
        random_state=args.seed, output_path=args.output,
    )
