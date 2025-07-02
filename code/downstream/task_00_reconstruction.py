import os
import json
import pandas as pd
import numpy as np

# === 1) Gather all test_metrics JSONs into a single DataFrame ===

# Make sure BASE_DIR points at the folder that contains "cna_from_…", "rnaseq_from_…", etc.
BASE_DIR = "../../results/32"
rows = []

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
            combo = fname[len(prefix): fname.rfind(suffix)]
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

df_results = pd.DataFrame(rows)
if df_results.empty:
    raise RuntimeError(f"No test_metrics JSON found under {BASE_DIR}.")

# Convert metric columns to numeric
for col in ["mse_mean", "mse_std", "r2_mean", "r2_std", "cos_mean", "cos_std"]:
    if col in df_results:
        df_results[col] = pd.to_numeric(df_results[col], errors="coerce")

# === 2) Pre-computation to get the data for the plot ===

all_targets = sorted(df_results["target"].unique())
conditions_to_plot = []

# Identify all data points that would have been part of the plot
for modality in all_targets:
    others = sorted([m for m in all_targets if m != modality])
    single_sources = others
    full_multi_source = '_'.join(others)
    full_coherent_source = '_'.join(others)

    for s in single_sources:
        conditions_to_plot.append((modality, s, 'single'))
    conditions_to_plot.append((modality, full_coherent_source, 'coherent'))
    conditions_to_plot.append((modality, full_multi_source, 'multi'))

# Create a boolean mask to filter the DataFrame
mask = pd.Series(False, index=df_results.index)
for target, source, sampling in conditions_to_plot:
    mask |= (
        (df_results["target"] == target) &
        (df_results["source"] == source) &
        (df_results["sampling_type"] == sampling) &
        (df_results["metric_used"] == "mse")
    )
df_plot_data = df_results[mask].copy()

# === 3) Save the data that would be used to create the plot in a single csv ===
plot_data_save_path = "../../results/downstream/task_00_rsquared/r2_data.csv"
df_plot_data.to_csv(plot_data_save_path, index=False)

print(f"Plot data saved to {plot_data_save_path}")

# === 4) Extract and save the means and std dev that are reported in the plot as a second csv ===

summary_stats = []

for i, modality in enumerate(all_targets):
    others = sorted([m for m in all_targets if m != modality])

    single_sources = others
    full_multi_source = '_'.join(others)
    full_coherent_source = '_'.join(others)

    labels = single_sources + ["Coherent", "Multi"]
    data_sources = [
        (s, 'single') for s in single_sources
    ]
    data_sources.append((full_coherent_source, 'coherent'))
    data_sources.append((full_multi_source, 'multi'))
    
    for j, (source, sampling) in enumerate(data_sources):
        df_row = df_plot_data[
            (df_plot_data["target"] == modality) &
            (df_plot_data["sampling_type"] == sampling) &
            (df_plot_data["source"] == source)
        ]
        
        if not df_row.empty:
            summary_stats.append({
                'target': modality,
                'source_label': labels[j],
                'r2_mean': df_row.iloc[0]["r2_mean"],
                'r2_std': df_row.iloc[0]["r2_std"],
                'sampling_type': sampling,
                'source': source
            })
        else:
             summary_stats.append({
                'target': modality,
                'source_label': labels[j],
                'r2_mean': np.nan,
                'r2_std': np.nan,
                'sampling_type': sampling,
                'source': source
            })

df_summary_stats = pd.DataFrame(summary_stats)

# === 5) Save the summary stats to a CSV file ===
summary_stats_save_path = "../../results/downstream/task_00_rsquared/r2_summary_stats.csv"
df_summary_stats.to_csv(summary_stats_save_path, index=False)

print(f"Summary statistics saved to {summary_stats_save_path}")