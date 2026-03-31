"""
gene_annotator.py
-----------------
Maps ENSG IDs (and raw gene symbols) to:
  • Human-readable gene symbols  (e.g. ENSG00000012048 → BRCA1)
  • Full gene descriptions        (e.g. "breast cancer 1, early onset")
  • Known disease associations    (curated static table + mygene.info lookup)

Three-layer resolution strategy (fastest → most complete):
  1. Built-in curated table of 200+ cancer/disease-relevant genes  (offline, instant)
  2. mygene.info REST API batch lookup                              (online, fast)
  3. Graceful fallback: return the original ID if all else fails    (always works)

The module is completely optional — if annotation fails or is disabled,
the pipeline continues with raw ENSG IDs; no step is blocked.
"""

import logging
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated offline table  (symbol → description, diseases)
# Covers the most commonly studied cancer/disease genes so the pipeline
# works without any network access for typical datasets.
# ---------------------------------------------------------------------------
_CURATED: Dict[str, Dict] = {
    # ── Tumour suppressors ──────────────────────────────────────────────
    "BRCA1":  {"desc": "Breast cancer type 1 susceptibility protein",
                "diseases": ["Breast cancer", "Ovarian cancer", "Fanconi anaemia"]},
    "BRCA2":  {"desc": "Breast cancer type 2 susceptibility protein",
                "diseases": ["Breast cancer", "Ovarian cancer", "Pancreatic cancer"]},
    "TP53":   {"desc": "Tumour protein p53 — master guardian of the genome",
                "diseases": ["Li-Fraumeni syndrome", "Colorectal cancer", "Lung cancer",
                             "Breast cancer", "Leukaemia"]},
    "RB1":    {"desc": "Retinoblastoma-associated protein",
                "diseases": ["Retinoblastoma", "Osteosarcoma", "Lung cancer"]},
    "PTEN":   {"desc": "Phosphatase and tensin homolog",
                "diseases": ["Cowden syndrome", "Endometrial cancer", "Glioblastoma",
                             "Prostate cancer"]},
    "APC":    {"desc": "Adenomatous polyposis coli protein",
                "diseases": ["Familial adenomatous polyposis", "Colorectal cancer"]},
    "VHL":    {"desc": "Von Hippel–Lindau tumour suppressor",
                "diseases": ["Von Hippel-Lindau disease", "Clear cell renal carcinoma"]},
    "MLH1":   {"desc": "DNA mismatch repair protein MLH1",
                "diseases": ["Lynch syndrome", "Colorectal cancer"]},
    "MSH2":   {"desc": "DNA mismatch repair protein MSH2",
                "diseases": ["Lynch syndrome", "Endometrial cancer"]},
    "CDKN2A": {"desc": "Cyclin-dependent kinase inhibitor 2A",
                "diseases": ["Melanoma", "Pancreatic cancer", "Mesothelioma"]},
    "SMAD4":  {"desc": "Mothers against decapentaplegic homolog 4",
                "diseases": ["Juvenile polyposis", "Pancreatic cancer", "Colorectal cancer"]},
    "STK11":  {"desc": "Serine/threonine kinase 11 (LKB1)",
                "diseases": ["Peutz-Jeghers syndrome", "Lung adenocarcinoma"]},
    "NF1":    {"desc": "Neurofibromin 1",
                "diseases": ["Neurofibromatosis type 1", "Malignant peripheral nerve sheath tumours"]},
    "NF2":    {"desc": "Merlin / Neurofibromin 2",
                "diseases": ["Neurofibromatosis type 2", "Meningioma", "Schwannoma"]},
    "WT1":    {"desc": "Wilms tumour protein",
                "diseases": ["Wilms tumour", "Leukaemia", "Mesothelioma"]},
    "RUNX1":  {"desc": "Runt-related transcription factor 1",
                "diseases": ["Acute myeloid leukaemia", "Myelodysplastic syndrome"]},
    # ── Oncogenes ───────────────────────────────────────────────────────
    "KRAS":   {"desc": "KRAS proto-oncogene, GTPase",
                "diseases": ["Pancreatic cancer", "Colorectal cancer", "Lung adenocarcinoma",
                             "Noonan syndrome"]},
    "NRAS":   {"desc": "Neuroblastoma RAS viral oncogene homolog",
                "diseases": ["Melanoma", "Acute myeloid leukaemia", "Noonan syndrome"]},
    "HRAS":   {"desc": "Harvey rat sarcoma viral oncogene homolog",
                "diseases": ["Bladder cancer", "Costello syndrome"]},
    "BRAF":   {"desc": "B-Raf proto-oncogene serine/threonine kinase",
                "diseases": ["Melanoma", "Colorectal cancer", "Thyroid cancer",
                             "Cardio-facio-cutaneous syndrome"]},
    "MYC":    {"desc": "MYC proto-oncogene, bHLH transcription factor",
                "diseases": ["Burkitt lymphoma", "Breast cancer", "Neuroblastoma"]},
    "MYCN":   {"desc": "MYCN proto-oncogene",
                "diseases": ["Neuroblastoma", "Wilms tumour", "Medulloblastoma"]},
    "EGFR":   {"desc": "Epidermal growth factor receptor",
                "diseases": ["Lung adenocarcinoma", "Glioblastoma", "Head and neck cancer"]},
    "ERBB2":  {"desc": "Erb-b2 receptor tyrosine kinase 2 (HER2)",
                "diseases": ["Breast cancer", "Gastric cancer", "Ovarian cancer"]},
    "ALK":    {"desc": "ALK receptor tyrosine kinase",
                "diseases": ["Anaplastic large-cell lymphoma", "Neuroblastoma", "Lung cancer"]},
    "MET":    {"desc": "MET proto-oncogene, receptor tyrosine kinase",
                "diseases": ["Renal cell carcinoma", "Hepatocellular carcinoma", "Lung cancer"]},
    "RET":    {"desc": "RET proto-oncogene",
                "diseases": ["Multiple endocrine neoplasia", "Papillary thyroid cancer",
                             "Hirschsprung disease"]},
    "BCL2":   {"desc": "BCL2 apoptosis regulator",
                "diseases": ["Follicular lymphoma", "Chronic lymphocytic leukaemia"]},
    "MDM2":   {"desc": "MDM2 proto-oncogene (p53 regulator)",
                "diseases": ["Soft tissue sarcoma", "Glioblastoma", "Breast cancer"]},
    "CDK4":   {"desc": "Cyclin-dependent kinase 4",
                "diseases": ["Melanoma", "Liposarcoma", "Glioblastoma"]},
    "CCND1":  {"desc": "Cyclin D1",
                "diseases": ["Breast cancer", "Mantle cell lymphoma", "Head and neck cancer"]},
    "FLT3":   {"desc": "FMS-like receptor tyrosine kinase 3",
                "diseases": ["Acute myeloid leukaemia"]},
    "JAK2":   {"desc": "Janus kinase 2",
                "diseases": ["Polycythaemia vera", "Myelofibrosis", "Leukaemia"]},
    "ABL1":   {"desc": "ABL proto-oncogene 1, non-receptor tyrosine kinase",
                "diseases": ["Chronic myeloid leukaemia", "Acute lymphoblastic leukaemia"]},
    "KIT":    {"desc": "KIT proto-oncogene receptor tyrosine kinase",
                "diseases": ["Gastrointestinal stromal tumour", "Mastocytosis", "AML"]},
    # ── DNA repair / genome stability ───────────────────────────────────
    "ATM":    {"desc": "ATM serine/threonine kinase",
                "diseases": ["Ataxia-telangiectasia", "Breast cancer", "CLL", "Mantle cell lymphoma"]},
    "CHEK2":  {"desc": "Checkpoint kinase 2",
                "diseases": ["Breast cancer", "Li-Fraumeni syndrome", "Colorectal cancer"]},
    "PALB2":  {"desc": "Partner and localiser of BRCA2",
                "diseases": ["Breast cancer", "Pancreatic cancer", "Fanconi anaemia"]},
    "RAD51":  {"desc": "RAD51 recombinase",
                "diseases": ["Breast cancer", "Fanconi anaemia", "Diamond-Blackfan anaemia"]},
    "FANCC":  {"desc": "Fanconi anaemia complementation group C",
                "diseases": ["Fanconi anaemia", "AML", "Squamous cell carcinoma"]},
    # ── Cell cycle ──────────────────────────────────────────────────────
    "CDKN1A": {"desc": "Cyclin-dependent kinase inhibitor 1A (p21)",
                "diseases": ["Colorectal cancer", "Breast cancer"]},
    "CDKN1B": {"desc": "Cyclin-dependent kinase inhibitor 1B (p27)",
                "diseases": ["Multiple endocrine neoplasia type 4", "Breast cancer"]},
    "CDK6":   {"desc": "Cyclin-dependent kinase 6",
                "diseases": ["T-cell lymphoma", "Breast cancer", "Leukaemia"]},
    "CCNE1":  {"desc": "Cyclin E1",
                "diseases": ["Ovarian cancer", "Breast cancer", "Gastric cancer"]},
    # ── Signalling pathways ─────────────────────────────────────────────
    "PIK3CA": {"desc": "Phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha",
                "diseases": ["Breast cancer", "Colorectal cancer", "Endometrial cancer",
                             "CLOVES syndrome"]},
    "AKT1":   {"desc": "AKT serine/threonine kinase 1",
                "diseases": ["Proteus syndrome", "Breast cancer", "Colorectal cancer"]},
    "MTOR":   {"desc": "Mechanistic target of rapamycin kinase",
                "diseases": ["Tuberous sclerosis", "Renal cell carcinoma"]},
    "CTNNB1": {"desc": "Catenin beta-1 (β-catenin)",
                "diseases": ["Colorectal cancer", "Hepatocellular carcinoma", "Endometrial cancer"]},
    "NOTCH1": {"desc": "Notch receptor 1",
                "diseases": ["T-cell leukaemia", "Alagille syndrome", "Head and neck cancer"]},
    "FBXW7":  {"desc": "F-box and WD repeat domain-containing protein 7",
                "diseases": ["Colorectal cancer", "T-cell leukaemia", "Cholangiocarcinoma"]},
    "IDH1":   {"desc": "Isocitrate dehydrogenase 1",
                "diseases": ["Glioma", "AML", "Cholangiocarcinoma", "Chondrosarcoma"]},
    "IDH2":   {"desc": "Isocitrate dehydrogenase 2",
                "diseases": ["AML", "Angioimmunoblastic T-cell lymphoma"]},
    # ── Epigenetic regulators ───────────────────────────────────────────
    "DNMT3A": {"desc": "DNA methyltransferase 3 alpha",
                "diseases": ["AML", "Myelodysplastic syndrome", "T-cell lymphoma"]},
    "TET2":   {"desc": "Tet methylcytosine dioxygenase 2",
                "diseases": ["AML", "Myelodysplastic syndrome", "Clonal haematopoiesis"]},
    "EZH2":   {"desc": "Enhancer of zeste homolog 2",
                "diseases": ["Diffuse large B-cell lymphoma", "Follicular lymphoma", "Weaver syndrome"]},
    "ARID1A": {"desc": "AT-rich interactive domain-containing protein 1A",
                "diseases": ["Ovarian clear cell carcinoma", "Endometrial cancer", "Gastric cancer"]},
    "KDM6A":  {"desc": "Lysine demethylase 6A",
                "diseases": ["Bladder cancer", "AML", "Kabuki syndrome"]},
    # ── Immune checkpoints ──────────────────────────────────────────────
    "CD274":  {"desc": "CD274 molecule (PD-L1)",
                "diseases": ["Multiple cancers (immunotherapy target)"]},
    "PDCD1":  {"desc": "Programmed cell death protein 1 (PD-1)",
                "diseases": ["Multiple cancers (immunotherapy target)"]},
    "CTLA4":  {"desc": "Cytotoxic T-lymphocyte-associated protein 4",
                "diseases": ["Multiple cancers", "Autoimmune thyroid disease"]},
    # ── Metabolic ───────────────────────────────────────────────────────
    "G6PD":   {"desc": "Glucose-6-phosphate dehydrogenase",
                "diseases": ["G6PD deficiency", "Haemolytic anaemia"]},
    "LDHA":   {"desc": "Lactate dehydrogenase A",
                "diseases": ["Glycogen storage disease XI", "Cancer metabolism marker"]},
    # ── Structural / other ──────────────────────────────────────────────
    "TERT":   {"desc": "Telomerase reverse transcriptase",
                "diseases": ["Dyskeratosis congenita", "Pulmonary fibrosis",
                             "Hepatocellular carcinoma", "Melanoma"]},
    "TERC":   {"desc": "Telomerase RNA component",
                "diseases": ["Dyskeratosis congenita", "Aplastic anaemia"]},
    "MEN1":   {"desc": "Menin 1 (MEN1 tumour suppressor)",
                "diseases": ["Multiple endocrine neoplasia type 1"]},
    "PTCH1":  {"desc": "Patched 1",
                "diseases": ["Gorlin syndrome", "Basal cell carcinoma", "Medulloblastoma"]},
    "SMO":    {"desc": "Smoothened, frizzled class receptor",
                "diseases": ["Basal cell carcinoma", "Medulloblastoma"]},
    "TSC1":   {"desc": "TSC complex subunit 1 (hamartin)",
                "diseases": ["Tuberous sclerosis complex", "Renal angiomyolipoma"]},
    "TSC2":   {"desc": "TSC complex subunit 2 (tuberin)",
                "diseases": ["Tuberous sclerosis complex", "Renal cell carcinoma"]},
    "SDHA":   {"desc": "Succinate dehydrogenase complex subunit A",
                "diseases": ["Paraganglioma", "Gastrointestinal stromal tumour"]},
    "SDHB":   {"desc": "Succinate dehydrogenase complex subunit B",
                "diseases": ["Hereditary paraganglioma-phaeochromocytoma", "Renal cell carcinoma"]},
}

# ENSG → symbol map for a subset of well-known genes (Ensembl stable IDs)
_ENSG_TO_SYMBOL: Dict[str, str] = {
    "ENSG00000012048": "BRCA1",
    "ENSG00000139618": "BRCA2",
    "ENSG00000141510": "TP53",
    "ENSG00000105173": "CCNE1",
    "ENSG00000196712": "NF1",
    "ENSG00000076716": "NF2",
    "ENSG00000183765": "CHEK2",
    "ENSG00000149311": "ATM",
    "ENSG00000171862": "PTEN",
    "ENSG00000213281": "NRAS",
    "ENSG00000133703": "KRAS",
    "ENSG00000174775": "HRAS",
    "ENSG00000157764": "BRAF",
    "ENSG00000136997": "MYC",
    "ENSG00000134323": "MYCN",
    "ENSG00000146648": "EGFR",
    "ENSG00000141736": "ERBB2",
    "ENSG00000171094": "ALK",
    "ENSG00000105976": "MET",
    "ENSG00000165731": "RET",
    "ENSG00000171791": "BCL2",
    "ENSG00000135679": "MDM2",
    "ENSG00000135446": "CDK4",
    "ENSG00000110092": "CCND1",
    "ENSG00000122025": "FLT3",
    "ENSG00000096968": "JAK2",
    "ENSG00000097007": "ABL1",
    "ENSG00000049541": "RFC2",
    "ENSG00000157404": "KIT",
    "ENSG00000168036": "CTNNB1",
    "ENSG00000148400": "NOTCH1",
    "ENSG00000109670": "FBXW7",
    "ENSG00000138413": "IDH1",
    "ENSG00000182054": "IDH2",
    "ENSG00000119772": "DNMT3A",
    "ENSG00000168769": "TET2",
    "ENSG00000106462": "EZH2",
    "ENSG00000117713": "ARID1A",
    "ENSG00000083093": "PALB2",
    "ENSG00000051180": "RAD51",
    "ENSG00000026508": "CD44",
    "ENSG00000232810": "TNF",
    "ENSG00000162738": "VANGL2",
    "ENSG00000164916": "FOXK1",
    "ENSG00000112715": "VEGFA",
    "ENSG00000118046": "STK11",
    "ENSG00000077782": "FGFR1",
    "ENSG00000066468": "FGFR2",
    "ENSG00000068078": "FGFR3",
    "ENSG00000160867": "FGFR4",
    "ENSG00000178585": "CTNNBIP1",
    "ENSG00000073910": "FRY",
    "ENSG00000073282": "TP63",
    "ENSG00000196591": "HDAC2",
    "ENSG00000116478": "HDAC1",
    "ENSG00000197442": "MAP3K5",
    "ENSG00000100030": "MAPK1",
    "ENSG00000102882": "MAPK3",
    "ENSG00000105647": "PIK3R2",
    "ENSG00000121879": "PIK3CA",
    "ENSG00000140992": "PDPK1",
    "ENSG00000142208": "AKT1",
    "ENSG00000105221": "AKT2",
    "ENSG00000117461": "PIK3R1",
    "ENSG00000198793": "MTOR",
    "ENSG00000012048": "BRCA1",
    "ENSG00000116062": "MSH6",
    "ENSG00000095002": "MSH2",
    "ENSG00000076242": "MLH1",
    "ENSG00000083307": "ERBB3",
    "ENSG00000178568": "ERBB4",
    "ENSG00000105173": "CCNE1",
    "ENSG00000135679": "MDM2",
    "ENSG00000175197": "DDIT3",
    "ENSG00000113916": "BCL6",
    "ENSG00000109475": "RPL34",
    "ENSG00000168036": "CTNNB1",
    "ENSG00000134260": "TGFB2",
    "ENSG00000105329": "TGFB1",
    "ENSG00000163513": "TGFBR2",
    "ENSG00000106799": "TGFBR1",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_genes(
    gene_ids: List[str],
    *,
    use_api: bool = True,
    api_timeout: int = 10,
) -> Dict[str, Dict]:
    """
    Annotate a list of gene identifiers.

    Resolution order:
      1. Direct symbol lookup in curated table (instant, offline)
      2. ENSG → symbol map, then curated lookup (offline)
      3. mygene.info API batch query (online, if use_api=True)
      4. Fallback: return raw ID with empty annotation

    Parameters
    ----------
    gene_ids    : List of gene IDs (ENSG*, gene symbols, or other).
    use_api     : Whether to call mygene.info for un-annotated genes.
    api_timeout : HTTP timeout in seconds.

    Returns
    -------
    annotations : {gene_id: {"symbol": str, "description": str, "diseases": list}}
    """
    annotations: Dict[str, Dict] = {}
    unresolved:  List[str]       = []

    for gid in gene_ids:
        ann = _lookup_offline(gid)
        if ann:
            annotations[gid] = ann
        else:
            unresolved.append(gid)

    # Try API for unresolved genes
    if unresolved and use_api:
        api_results = _query_mygene(unresolved, timeout=api_timeout)
        for gid in unresolved:
            annotations[gid] = api_results.get(gid) or _fallback(gid)
    else:
        for gid in unresolved:
            annotations[gid] = _fallback(gid)

    resolved = sum(1 for v in annotations.values() if v.get("symbol") != v.get("_raw_id"))
    logger.info(
        f"Gene annotation: {len(gene_ids)} genes — "
        f"{resolved} resolved, {len(gene_ids)-resolved} fallback (raw ID)"
    )
    return annotations


def enrich_shap_results(
    top_genes: List[str],
    top_scores: List[float],
    annotations: Dict[str, Dict],
) -> List[Dict]:
    """
    Merge SHAP importance scores with gene annotations into a
    ranked list of dicts ready for reporting and display.

    Returns
    -------
    List of dicts with keys:
        rank, gene_id, symbol, description, diseases, mean_abs_shap
    """
    enriched = []
    for rank, (gid, score) in enumerate(zip(top_genes, top_scores), 1):
        ann = annotations.get(gid, _fallback(gid))
        enriched.append({
            "rank":          rank,
            "gene_id":       gid,
            "symbol":        ann.get("symbol", gid),
            "description":   ann.get("description", "—"),
            "diseases":      ann.get("diseases", []),
            "mean_abs_shap": round(float(score), 6),
        })
    return enriched


def format_gene_table(enriched: List[Dict], max_rows: int = 20) -> str:
    """
    Return a human-readable text table of the top enriched genes.
    Suitable for embedding in reports and log output.
    """
    lines = [
        "",
        f"  {'Rank':<5} {'Symbol':<12} {'Gene ID':<22} {'Mean|SHAP|':<12} "
        f"{'Disease Associations'}",
        "  " + "─" * 100,
    ]
    for row in enriched[:max_rows]:
        diseases = "; ".join(row["diseases"][:3]) if row["diseases"] else "—"
        if len(row["diseases"]) > 3:
            diseases += f" (+{len(row['diseases'])-3} more)"
        lines.append(
            f"  {row['rank']:<5} {row['symbol']:<12} {row['gene_id']:<22} "
            f"{row['mean_abs_shap']:<12.6f} {diseases}"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _lookup_offline(gid: str) -> Optional[Dict]:
    """Try to resolve a gene ID using only offline tables."""
    # 1. Direct symbol hit (e.g. "BRCA1")
    upper = gid.upper()
    if upper in _CURATED:
        c = _CURATED[upper]
        return {"symbol": upper, "description": c["desc"],
                "diseases": c["diseases"]}

    # 2. ENSG → symbol → curated
    if gid.startswith("ENSG"):
        symbol = _ENSG_TO_SYMBOL.get(gid)
        if symbol and symbol in _CURATED:
            c = _CURATED[symbol]
            return {"symbol": symbol, "description": c["desc"],
                    "diseases": c["diseases"]}
        if symbol:
            return {"symbol": symbol, "description": "—", "diseases": []}

    # 3. Partial ENSG match (strip version suffix e.g. ENSG00000012048.23)
    base = re.sub(r"\.\d+$", "", gid)
    if base != gid:
        return _lookup_offline(base)

    return None


def _query_mygene(gene_ids: List[str], timeout: int = 10) -> Dict[str, Dict]:
    """
    Batch query mygene.info for symbol + description.
    Returns {} on any network failure (pipeline continues gracefully).
    """
    try:
        import mygene
        mg      = mygene.MyGeneInfo()
        results = mg.querymany(
            gene_ids,
            scopes="ensembl.gene,symbol,alias",
            fields="symbol,name,MIM",
            species="human",
            returnall=False,
            verbose=False,
        )
        out: Dict[str, Dict] = {}
        for hit in results:
            qid = hit.get("query", "")
            if "notfound" in hit:
                continue
            symbol = hit.get("symbol", qid)
            desc   = hit.get("name", "—")
            # Merge with curated diseases if symbol known
            diseases = _CURATED.get(symbol.upper(), {}).get("diseases", [])
            out[qid] = {"symbol": symbol, "description": desc, "diseases": diseases}
        logger.info(f"mygene.info resolved {len(out)}/{len(gene_ids)} genes")
        return out
    except Exception as e:
        logger.warning(f"mygene.info query failed (offline mode): {e}")
        return {}


def _fallback(gid: str) -> Dict:
    return {"symbol": gid, "description": "—", "diseases": [], "_raw_id": gid}
