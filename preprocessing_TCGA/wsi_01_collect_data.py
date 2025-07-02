#!/usr/bin/env python3
import shutil
import argparse
from pathlib import Path
import glob

# ─── Configuration ─────────────────────────────────────────────────────────────

# Source path pattern for the Pathology folder
pathology_pattern = "../Pathology/{cohort}/embeddings/trident_processed/*/20x_512px_0px_overlap/slide_features_titan/*.h5" # use '{cohort}' placeholder in pathology_pattern
cohorts = ["TCGA-GBM", "TCGA-LGG"] # Cohorts under Pathology to process

# Source path pattern for the Storage folder
storage_pattern   = "/storage/DSH/projects/data/TCGA/embeddings/*/trident_processed/20x_512px_0px_overlap/slide_features_titan/*.h5"

# Destination folder for all .h5 copies 
dest_root = Path("../datasets_TCGA/wsi_embeddings/individual_embeddings")
# ────────────────────────────────────────────────────────────────────────────────

def collect_files(patterns, dest_root):
    """
    Given a list of glob patterns (which may be absolute or relative), expand
    using the glob module and copy matching .h5 files flat into dest_root.
    """
    for pattern in patterns:
        for file_str in glob.glob(pattern, recursive=True):
            file_path = Path(file_str)
            if not file_path.is_file():
                continue
            target = dest_root / file_path.name
            shutil.copy2(file_path, target)
            print(f"Copied: {file_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect slide embeddings from specified locations into a single folder"
    )
    parser.add_argument(
        "--mode", choices=["pathology", "storage", "all"], default="all",
        help="Source location: 'pathology', 'storage', or 'all'"
    )
    args = parser.parse_args()

    # Build the list of patterns based on mode
    patterns = []
    if args.mode in ("pathology", "all"):
        patterns += [pathology_pattern.format(cohort=cohort) for cohort in cohorts]
    if args.mode in ("storage", "all"):
        patterns.append(storage_pattern)

    # Ensure destination exists
    dest_root.mkdir(parents=True, exist_ok=True)

    collect_files(patterns, dest_root)


