import os
import json
import pandas as pd
import numpy as np
from itertools import combinations

# === 1) Gather all test_metrics JSONs into a single DataFrame ===

# Make sure BASE_DIR points at the folder that contains "cna_from_…", "rnaseq_from_…", etc.
BASE_DIR = "../../results/32"
# Use a relative path for the output directory to save the CSV files
OUTPUT_DIR = "../../results/downstream/task_00_rsquared/" 
rows = []

print("Starting data collection...")

for root, dirs, files in os.walk(BASE_DIR):
    for fname in files:
        if not (fname.startswith("test_metrics") and fname.endswith(".json")):
            continue

        fullpath = os.path.join(root, fname)
        exp_folder = os.path.basename(os.path.dirname(root))

        # Determine sampling_type
        if "_from_coherent" in exp_folder:
            sampling_type = "coherent"
        elif "_from_multi" in exp_folder:
            sampling_type = "multi"
        else:
            sampling_type = "single"

        # Parse target and source
        if sampling_type == "single":
            if "_from_" not in exp_folder:
                continue
            target, src_part = exp_folder.split("_from_", 1)
            source = src_part
        else:
            target = exp_folder.split("_from_", 1)[0]
            prefix = "test_metrics_from_"
            suffix = "_best"
            # Handle potential file naming variations
            if fname.rfind(suffix) != -1:
                combo = fname[len(prefix): fname.rfind(suffix)]
            else:
                combo = fname[len(prefix):]
            source = combo

        # Which reference metric?
        if "best_mse" in fname:
            metric_used = "mse"
        elif "best_cosine" in fname:
            metric_used = "cosine"
        elif "best_timestep" in fname:
            metric_used = "timestep"
        else:
            metric_used = None

        try:
            with open(fullpath, "r") as f:
                data = json.load(f)
            
            rows.append({
                "experiment":      exp_folder,
                "target":          target,
                "source":          source,
                "sampling_type":   sampling_type,
                "metric_used":     metric_used,
                **data
            })
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not read or parse {fullpath}. Error: {e}")


if not rows:
    raise RuntimeError(f"No valid test_metrics JSON found under {BASE_DIR}.")

df_results = pd.DataFrame(rows)
print(f"Collected {len(df_results)} results.")

# Convert metric columns to numeric, coercing errors to NaN
for col in ["mse_mean", "mse_std", "r2_mean", "r2_std", "cos_mean", "cos_std"]:
    if col in df_results:
        df_results[col] = pd.to_numeric(df_results[col], errors="coerce")

# === 2) For each target modality, prepare data for CSV export ===

all_targets = sorted(df_results["target"].unique())

# Lists to hold data before converting to DataFrame
summary_stats_rows = []
plot_data_frames = []

print("Processing data for each target modality...")

for modality in all_targets:
    # 2a) Identify other modalities
    others = sorted([m for m in all_targets if m != modality])

    # 2b) Build label lists to represent each bar on the would-be plot
    single_labels = others.copy()
    
    coherent_labels = [f"coh_{'_'.join(combo)}"
                       for r in range(2, len(others) + 1)
                       for combo in combinations(others, r)]
    
    multi_labels = [f"mult_{'_'.join(combo)}"
                    for r in range(1, len(others) + 1)
                    for combo in combinations(others, r)]

    # All labels in the exact order they would be plotted
    labels = single_labels + coherent_labels + multi_labels
    
    # 2c) Fill in Test R² values for MSE metric only
    for i, lbl in enumerate(labels):
        is_full_conditioning = False
        
        # Determine sampling type and source combination from the label
        if lbl in single_labels:
            sampling_type = "single"
            source = lbl
        elif lbl.startswith("coh_"):
            sampling_type = "coherent"
            source = lbl.replace("coh_", "", 1)
            # Check if this is the "all conditionings" case for coherent
            if len(source.split('_')) == len(others):
                is_full_conditioning = True
        else: # must start with "mult_"
            sampling_type = "multi"
            source = lbl.replace("mult_", "", 1)
            # Check if this is the "all conditionings" case for multi
            if len(source.split('_')) == len(others):
                is_full_conditioning = True

        # Find the corresponding data row (using only MSE metric)
        df_row = df_results[
            (df_results["target"] == modality) &
            (df_results["sampling_type"] == sampling_type) &
            (df_results["source"] == source) &
            (df_results["metric_used"] == "mse")
        ]
        
        # Append the raw data to a list for the first CSV
        if not df_row.empty:
            plot_data_frames.append(df_row)

        # Prepare a structured row for the summary statistics CSV
        if not df_row.empty:
            row_data = df_row.iloc[0]
            summary_stats_rows.append({
                'target_modality': modality,
                'label': lbl,
                'source_combination': source,
                'conditioning_type': sampling_type,
                'is_full_conditioning': is_full_conditioning,
                'r2_mean': row_data.get("r2_mean", np.nan),
                'r2_std': row_data.get("r2_std", np.nan)
            })
        else:
            # Add a placeholder if no data was found for this combination
            summary_stats_rows.append({
                'target_modality': modality,
                'label': lbl,
                'source_combination': source,
                'conditioning_type': sampling_type,
                'is_full_conditioning': is_full_conditioning,
                'r2_mean': np.nan,
                'r2_std': np.nan
            })

# === 3) Create and save the DataFrames to CSV files ===

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CSV 1: Raw data used for all plots ---
if plot_data_frames:
    df_plot_data = pd.concat(plot_data_frames).drop_duplicates().reset_index(drop=True)
    plot_data_save_path = os.path.join(OUTPUT_DIR, "r2_data_complete.csv")
    df_plot_data.to_csv(plot_data_save_path, index=False)
    print(f"Successfully saved raw plot data to {plot_data_save_path}")
else:
    print("Warning: No plot data was found to save.")

# --- CSV 2: Summary statistics representing each bar ---
if summary_stats_rows:
    df_summary_stats = pd.DataFrame(summary_stats_rows)
    summary_stats_save_path = os.path.join(OUTPUT_DIR, "r2_summary_stats_complete.csv")
    df_summary_stats.to_csv(summary_stats_save_path, index=False)
    print(f"Successfully saved summary statistics to {summary_stats_save_path}")
else:
    print("Warning: No summary statistics were generated to save.")