"""TCGA multi-omics preprocessing pipeline – merged-aware version
=================================================================
Key change
----------

* **Two** RNA-seq preprocessing helpers are now available:

  * `preprocess_rnaseq_single()` –- run **inside** the cohort loop, exactly as
    before, so each cohort still gets its own QC’ed matrices written to
    `…/processed_data/`.

  * `preprocess_rnaseq_merged()` –- run **after** *all* cohorts have been scanned.
    It takes the *merged* log2(count + 1) matrix (all samples, all genes),
    returns a *single* pair of log2-CPM matrices (full + filtered) and writes
    them to `…/datasets_TCGA/merged/`.

* During the cohort loop the script **does not** collect the per-cohort
  log2-CPM tables any more; it only stacks the raw log2(count + 1) matrices in
  `merged_rna_raw`.  These are concatenated *once* at the end and fed to
  `preprocess_rnaseq_merged()`.

All other modalities (clinical, CNA, RPPA) are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

###############################################################################
# ------------------------------ configuration ------------------------------ #
###############################################################################
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT  = (SCRIPT_DIR / ".." / "datasets_TCGA").resolve()

MERGE_DIR  = DATA_ROOT / "merged"
MERGE_DIR.mkdir(exist_ok=True, parents=True)

COHORTS = [
    "BRCA", "GBM", "LUAD", "OV", "UCEC", "KIRC", "HNSC", "LGG", "THCA",
    "LUSC", "PRAD", "SKCM", "COAD", "STAD", "BLCA", "LIHC", "CESC",
    "KIRP", "TGCT", "SARC",
]

###############################################################################
# ------------------------------ helper funcs ------------------------------- #
###############################################################################

def clean_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Second-pass cleaning of clinical tables."""
    thr = len(df) / 2
    df = df.dropna(axis=1, thresh=thr)
    nunique = df.nunique()
    df = df.drop(columns=nunique[nunique == 1].index)
    if "sample" in df.columns:
        df = df.rename(columns={"sample": "sample_id"})
    return df


def transpose(df: pd.DataFrame, id_name: str = "sample_id") -> pd.DataFrame:
    """Rows → samples, columns → features."""
    out = df.T.copy()
    out.index.name = id_name
    return out.reset_index()


# --------------------------------------------------------------------------- #
#  RNA-seq (STAR log2(count+1)) helpers                                       #
# --------------------------------------------------------------------------- #

def _log2expr_to_log2cpm(
    log2_expr: pd.DataFrame,
    min_cpm: float = 1.0,
    min_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shared worker: convert log2(count+1) → raw → CPM → log2-CPM and filter.
    """
    counts_raw  = (2.0 ** log2_expr) - 1.0
    sample_sums = counts_raw.sum(axis=0)
    cpm         = counts_raw.div(sample_sums, axis=1) * 1e6

    keep = (cpm >= min_cpm).sum(axis=1) >= (min_frac * counts_raw.shape[1])
    cpm_filt = cpm.loc[keep]

    log2cpm_full = np.log2(cpm      + 1.0)
    log2cpm_filt = np.log2(cpm_filt + 1.0)
    return log2cpm_full, log2cpm_filt


def preprocess_rnaseq_single(
    log2_expr: pd.DataFrame,
    *,
    min_cpm: float = 1.0,
    min_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-cohort QC – identical to the original behaviour.
    """
    return _log2expr_to_log2cpm(log2_expr, min_cpm=min_cpm, min_frac=min_frac)


def preprocess_rnaseq_merged(
    merged_log2_expr: pd.DataFrame,
    *,
    min_cpm: float = 1.0,
    min_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Global QC on the *merged* matrix.  Gene-filtering is now applied
    once across **all** cohorts, guaranteeing a common gene set.
    """
    return _log2expr_to_log2cpm(merged_log2_expr, min_cpm=min_cpm, min_frac=min_frac)


# --------------------------------------------------------------------------- #
#  Other omics helpers (unchanged)                                            #
# --------------------------------------------------------------------------- #
def preprocess_cna_absolute(cna: pd.DataFrame, clip: Tuple[int, int] = (-2, 2)) -> pd.DataFrame:
    cna = cna.replace(0, np.nan)
    logratio = np.log2(cna / 2.0)
    return logratio.clip(*clip)


def preprocess_rppa(rppa: pd.DataFrame) -> pd.DataFrame:
    return rppa.sub(rppa.median(axis=0), axis=1)
###############################################################################
# ------------------------------ main routine ------------------------------- #
###############################################################################
merged_clin     : List[pd.DataFrame] = []
merged_rna_raw  : List[pd.DataFrame] = []   # NEW – store raw log2(count+1)
merged_cna      : List[pd.DataFrame] = []
merged_rppa     : List[pd.DataFrame] = []

for cohort in COHORTS:
    cohort_dir = DATA_ROOT / f"tcga_{cohort}_ucsc_xena"
    raw_dir  = cohort_dir / "raw_data"
    proc_dir = cohort_dir / "processed_data"
    proc_dir.mkdir(parents=True, exist_ok=True)

    f_clin = raw_dir / f"TCGA-{cohort}.clinical.tsv"
    f_rna  = raw_dir / f"TCGA-{cohort}.star_counts.tsv"
    f_cna  = raw_dir / f"TCGA-{cohort}.gene-level_absolute.tsv"
    f_rppa = raw_dir / f"TCGA-{cohort}.protein.tsv"

    print(f"\n▶ {cohort}")

    # ----------------------------- clinical ------------------------------ #
    if f_clin.exists():
        clin = pd.read_csv(f_clin, sep="\t")
        clin = clean_clinical(clin)
        clin.to_csv(proc_dir / "clinical_preproc.csv", index=False)
        merged_clin.append(clin.set_index("sample_id"))
    else:
        print("  – clinical file missing, skipping")

    # ------------------------------ RNA-seq ------------------------------ #
    if f_rna.exists():
        log2_counts = pd.read_csv(f_rna, sep="\t", index_col=0)

        # --- cohort-level QC (unchanged) -------------------------------- #
        logcpm_full, logcpm_filt = preprocess_rnaseq_single(log2_counts)
        transpose(logcpm_full).to_csv(proc_dir / "rnaseq_preproc.csv",     index=False)
        transpose(logcpm_filt).to_csv(proc_dir / "rnaseq_QC_preproc.csv", index=False)

        # --- store *raw* matrix for global QC --------------------------- #
        merged_rna_raw.append(log2_counts)

    else:
        print("  – RNA file missing, skipping")

    # ------------------------------ CNA --------------------------------- #
    if f_cna.exists():
        cna_raw  = pd.read_csv(f_cna, sep="\t", index_col=0)
        logratio = preprocess_cna_absolute(cna_raw)
        transpose(logratio).to_csv(proc_dir / "cna_preproc.csv", index=False)
        merged_cna.append(transpose(logratio).set_index("sample_id"))

    # ------------------------------ RPPA -------------------------------- #
    if f_rppa.exists():
        rppa_raw = pd.read_csv(f_rppa, sep="\t", index_col=0)
        centred  = preprocess_rppa(rppa_raw)
        transpose(centred).to_csv(proc_dir / "protein_preproc.csv", index=False)
        merged_rppa.append(transpose(centred).set_index("sample_id"))

###############################################################################
# ------------------------------ merge step ------------------------------- #
###############################################################################
print("\n▶ Writing merged matrices …")

# ---------- clinical (outer, then second-pass cleaning) ------------------- #
if merged_clin:
    merged = pd.concat(merged_clin, axis=0, join="outer")
    merged_clean = clean_clinical(merged.reset_index())
    merged_clean.to_csv(MERGE_DIR / "merged_clinical.csv", index=False)
print("clin done")

# ---------- RNA-seq (global QC) ------------------------------------------ #
if merged_rna_raw:
    merged_raw = pd.concat(merged_rna_raw, axis=1, join="outer")
    logcpm_full, logcpm_filt = preprocess_rnaseq_merged(merged_raw)

    transpose(logcpm_full).to_csv(MERGE_DIR / "merged_rnaseq.csv",      index=False)
    transpose(logcpm_filt).to_csv(MERGE_DIR / "merged_rnaseq_QC.csv",  index=False)
print("RNA & RNA-QC done")

# ---------- CNA ---------------------------------------------------------- #
if merged_cna:
    pd.concat(merged_cna, axis=0, join="outer").to_csv(MERGE_DIR / "merged_cna.csv")
print("CNA done")

# ---------- RPPA --------------------------------------------------------- #
if merged_rppa:
    pd.concat(merged_rppa, axis=0, join="outer").to_csv(MERGE_DIR / "merged_rppa.csv")
print("Prot done")

print("\nAll done ✓ – check", DATA_ROOT)
