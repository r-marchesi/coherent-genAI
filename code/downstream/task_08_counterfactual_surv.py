import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
import os
import pathlib
import sys
from types import SimpleNamespace

# --- Package Installation and Imports ---
try:
    import sksurv
except ImportError:
    import subprocess
    print("sksurv not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-survival"])
    import sksurv

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# --- Original Task 07 Imports (minus classification) ---
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# 1. CONFIGURATION (Adapted for Survival)
# =============================================================================
# Define all modalities available in the dataset
# --- FIX 1: Corrected 'rna' to 'rnaseq' ---
ALL_MODALITIES = ['cna', 'rnaseq', 'rppa', 'wsi']

# --- Experiment Parameters ---
N_EXPERIMENT_REPEATS = 10
N_ITERATIONS = 30
ABLATION_STEPS = np.arange(0, 1.05, 0.05)
K_NEIGHBORS = 10

# --- File Paths ---
gen_path = '../../results/32/'
train_path = '../../datasets_TCGA/07_normalized/32/'
test_path = '../../datasets_TCGA/07_normalized/32/'
labels_path = '../../datasets_TCGA/downstream_labels/'
# NEW: Path for survival data
surv_dir_path = '../../datasets_TCGA/downstream_labels/survival'
# NEW: Output directory
output_dir = '../../results/downstream/task_07b_counterfactual_survival/'


# Mapping from the ablated modalities tuple to the corresponding generated data file.
# --- FIX 2: Corrected 'rna' to 'rnaseq' in keys and paths ---
GENERATED_DATA_MAPPING_coh = {
    ('rnaseq',): f'{gen_path}rnaseq_from_coherent/test/generated_samples_from_cna_rppa_wsi_best_mse.csv',
    ('wsi',): f'{gen_path}wsi_from_coherent/test/generated_samples_from_cna_rnaseq_rppa_best_mse.csv',
}

GENERATED_DATA_MAPPING_multi = {
    ('rnaseq',): f'{gen_path}rnaseq_from_multi/test/generated_samples_from_cna_rppa_wsi_best_mse.csv',
    ('wsi',): f'{gen_path}wsi_from_multi/test/generated_samples_from_cna_rnaseq_rppa_best_mse.csv',
}


# =============================================================================
# 2. HELPER FUNCTIONS (Adapted from Task 06 & 07)
# =============================================================================

def clean_and_unify_tables(df_dict, modalities):
    """
    Unifies multiple modality-specific dataframes into a single dataframe.
    (From original Task 07)
    """
    sample_ids = df_dict[modalities[0]]['sample_id'].values
    for m in modalities:
        assert np.all(df_dict[m]['sample_id'].values == sample_ids), f"Sample IDs for {m} do not match."

    processed_dfs = []
    for m in modalities:
        df = df_dict[m].drop(columns=['sample_id'])
        df.columns = [f'{m}_{i+1}' for i in range(df.shape[1])]
        processed_dfs.append(df)
        
    return pd.concat(processed_dfs, axis=1)

# --- NEW: Helper functions from Task 06b to load survival data ---

def to_survival_structured_array(df: pd.DataFrame) -> np.ndarray:
    """Converts a DataFrame with OS and OS.time into a structured array for scikit-survival."""
    event_indicator = df['OS'].astype(bool)
    event_time = df['OS.time'].astype(float)
    structured_array = np.array(
        list(zip(event_indicator, event_time)),
        dtype=[('event', bool), ('time', float)]
    )
    return structured_array

def load_survival_labels(labels_dir: str, surv_dir: str, train_index: pd.Index, test_index: pd.Index):
    """Loads and aligns survival data for train and test sets."""
    print("--- Loading and preparing survival labels... ---")
    
    # Load all survival files
    all_surv_files = [f for f in os.listdir(surv_dir) if f.endswith('.survival.tsv')]
    surv_dfs = []
    for fn in all_surv_files:
        try:
            cancer_surv = pd.read_csv(os.path.join(surv_dir, fn), sep="\t")
            cancer_surv = (cancer_surv.rename(columns={"sample": "sample_id"})
                           .assign(sample_id=lambda df_val: df_val.sample_id.str[:-1])
                           .set_index("sample_id").drop(columns=["_PATIENT"], errors='ignore'))
            cancer_surv = cancer_surv[~cancer_surv.index.duplicated(keep="first")]
            surv_dfs.append(cancer_surv)
        except Exception as e:
            print(f"Warning: Could not process survival file {fn}. Error: {e}")
            
    pan_surv = pd.concat(surv_dfs).dropna(subset=['OS', 'OS.time'])

    # Align with provided train/test indices
    train_labels_aligned = pan_surv.reindex(train_index).dropna()
    test_labels_aligned = pan_surv.reindex(test_index).dropna()
    
    # Convert to structured array
    y_train = to_survival_structured_array(train_labels_aligned)
    y_test = to_survival_structured_array(test_labels_aligned)
    
    # Return the labels and the indices of samples that had valid labels
    return y_train, y_test, train_labels_aligned.index, test_labels_aligned.index

# =============================================================================
# 3. CORE EXPERIMENT FUNCTION (Adapted for Survival)
# =============================================================================
def run_ablation_experiment(ablated_modalities: Tuple[str, ...]):
    """
    Runs the full progressive ablation experiment for a given set of ablated modalities
    using SURVIVAL ANALYSIS.
    """
    print(f"\n{'='*30}\nRUNNING SURVIVAL EXPERIMENT FOR ABLATED MODALITY: {str(ablated_modalities).upper()}\n{'='*30}")

    # --- 3a. Data Loading and Preparation (Task 07 logic) ---
    print("Loading datasets...")
    # --- FIX 3: Corrected 'rna' to 'rnaseq' in keys and paths ---
    train_raw = {
        'cna': pd.read_csv(f'{train_path}cna_train.csv', sep=','),
        'rnaseq': pd.read_csv(f'{train_path}rnaseq_train.csv', sep=','),
        'rppa': pd.read_csv(f'{train_path}rppa_train.csv', sep=','),
        'wsi': pd.read_csv(f'{train_path}wsi_train.csv', sep=','),
    }

    test_raw = {
        'cna': pd.read_csv(f'{test_path}cna_test.csv', sep=','),
        'rnaseq': pd.read_csv(f'{test_path}rnaseq_test.csv', sep=','),
        'rppa': pd.read_csv(f'{test_path}rppa_test.csv', sep=','),
        'wsi': pd.read_csv(f'{test_path}wsi_test.csv', sep=','),
    }
    
    # Store original indices
    train_sample_ids = train_raw[ALL_MODALITIES[0]]['sample_id']
    test_sample_ids = test_raw[ALL_MODALITIES[0]]['sample_id']

    print("Processing and unifying tables...")
    train_full = clean_and_unify_tables(train_raw, ALL_MODALITIES)
    test_full = clean_and_unify_tables(test_raw, ALL_MODALITIES)
    
    # Set original sample_id indices
    train_full.index = train_sample_ids
    test_full.index = test_sample_ids

    # --- NEW: Load survival labels and filter data ---
    train_labels, test_labels, train_valid_idx, test_valid_idx = load_survival_labels(
        labels_path, surv_dir_path, train_full.index, test_full.index
    )
    
    # Filter train/test data to only include samples with valid survival labels
    train_data = train_full.loc[train_valid_idx]
    test_data_full = test_full.loc[test_valid_idx].reset_index(drop=True) # Reset index for easy .loc
    test_labels_full = test_labels # This is already filtered
    
    print(f"  Found {len(train_data)} training samples and {len(test_data_full)} test samples with survival data.")


    # Load generated data
    generated_file_key = ablated_modalities
    if generated_file_key not in GENERATED_DATA_MAPPING_coh:
        raise ValueError(f"No generated data file specified for the modality combination: {generated_file_key}")

    try:
        gen_data_long_coh = {mod: pd.read_csv(GENERATED_DATA_MAPPING_coh[generated_file_key], sep=',') for mod in ablated_modalities}
        gen_data_long_multi = {mod: pd.read_csv(GENERATED_DATA_MAPPING_multi[generated_file_key], sep=',') for mod in ablated_modalities}
    except FileNotFoundError as e:
        print(f"Error: A generated data file was not found: {e}. Please check paths.")
        return None

    # --- 3b. Full Experiment Loop ---
    all_runs_gen_coh, all_runs_gen_multi, all_runs_random = [], [], []

    for run in range(N_EXPERIMENT_REPEATS):
        print(f"\n--- Starting Experiment Run {run + 1}/{N_EXPERIMENT_REPEATS} for {str(ablated_modalities).upper()} ---")
        
        # --- MODEL: Use RandomSurvivalForest ---
        rsf = RandomSurvivalForest(n_estimators=100, random_state=42 + run, oob_score=True, n_jobs=-1)
        print("Training Random Survival Forest model...")
        rsf.fit(train_data, train_labels) # Use DataFrame and structured array

        # Progressive Ablation Analysis
        test_data = test_data_full.copy()
        ablated_cols = [c for c in test_data.columns if c.split('_')[0] in ablated_modalities]
        
        # Prepare generated candidates
        n_full_test_samples = len(test_full) # Original size before filtering
        
        # Get original integer indices of the test samples we are using
        original_indices = test_full.index.get_indexer(test_valid_idx)
        n_test_samples_to_use = len(original_indices) # This is len(test_data)
        
        generated_candidates_coh = {mod: [] for mod in ablated_modalities}
        generated_candidates_multi = {mod: [] for mod in ablated_modalities}
        
        for mod in ablated_modalities:
            df_coh = gen_data_long_coh[mod]
            df_multi = gen_data_long_multi[mod]
            for i in original_indices:
                gather_indices = np.arange(K_NEIGHBORS) * n_full_test_samples + i
                generated_candidates_coh[mod].append(df_coh.iloc[gather_indices])
                generated_candidates_multi[mod].append(df_multi.iloc[gather_indices])
        
        # --- VARIANCE: Calculate based on risk scores ---
        print("Calculating counterfactual risk score variances...")
        prediction_variances_gen_coh, prediction_variances_gen_multi = [], []
        for i in range(n_test_samples_to_use):
            original_sample = test_data.iloc[[i]]
            ablated_sample = original_sample.copy()
            ablated_sample[ablated_cols] = np.nan # RSF handles NaNs
            
            # Predict risk score for the ablated sample
            ablated_prediction = rsf.predict(ablated_sample)[0]
            
            # Prepare batch of K counterfactuals
            prediction_input_coh = pd.concat([original_sample] * K_NEIGHBORS, ignore_index=True)
            prediction_input_multi = pd.concat([original_sample] * K_NEIGHBORS, ignore_index=True)
            for mod in ablated_modalities:
                mod_cols = [c for c in test_data.columns if c.startswith(f'{mod}_')]
                prediction_input_coh[mod_cols] = generated_candidates_coh[mod][i].values
                prediction_input_multi[mod_cols] = generated_candidates_multi[mod][i].values
            
            # Get K risk scores
            generated_predictions_coh = rsf.predict(prediction_input_coh)
            generated_predictions_multi = rsf.predict(prediction_input_multi)
            
            # Calculate variance as mean squared difference from ablated risk
            prediction_variances_gen_coh.append(np.mean((generated_predictions_coh - ablated_prediction)**2))
            prediction_variances_gen_multi.append(np.mean((generated_predictions_multi - ablated_prediction)**2))

        variance_df_gen_coh = pd.DataFrame({'variance': prediction_variances_gen_coh, 'original_index': test_data.index}).sort_values(by='variance', ascending=True)
        variance_df_gen_multi = pd.DataFrame({'variance': prediction_variances_gen_multi, 'original_index': test_data.index}).sort_values(by='variance', ascending=True)

        # --- METRIC: Calculate C-Index for all methods ---
        gen_selective_c_index_coh, gen_selective_c_index_multi, random_ablation_c_index = [], [], []

        for ratio in ABLATION_STEPS:
            n_to_ablate = int(n_test_samples_to_use * ratio)

            # Selective Ablation (Coherent)
            data_for_pred_coh = test_data.copy()
            ablation_indices = variance_df_gen_coh.head(n_to_ablate)['original_index'].values
            if len(ablation_indices) > 0:
                data_for_pred_coh.loc[ablation_indices, ablated_cols] = np.nan
            preds_coh = rsf.predict(data_for_pred_coh)
            gen_selective_c_index_coh.append(concordance_index_censored(test_labels_full["event"], test_labels_full["time"], preds_coh)[0])

            # Selective Ablation (Multi)
            data_for_pred_multi = test_data.copy()
            ablation_indices = variance_df_gen_multi.head(n_to_ablate)['original_index'].values
            if len(ablation_indices) > 0:
                data_for_pred_multi.loc[ablation_indices, ablated_cols] = np.nan
            preds_multi = rsf.predict(data_for_pred_multi)
            gen_selective_c_index_multi.append(concordance_index_censored(test_labels_full["event"], test_labels_full["time"], preds_multi)[0])

            # Random Ablation
            iter_scores = []
            for _ in range(N_ITERATIONS):
                data_for_pred_rand = test_data.copy()
                ablation_indices = np.random.choice(data_for_pred_rand.index, size=n_to_ablate, replace=False)
                if len(ablation_indices) > 0:
                    data_for_pred_rand.loc[ablation_indices, ablated_cols] = np.nan
                preds_rand = rsf.predict(data_for_pred_rand)
                iter_scores.append(concordance_index_censored(test_labels_full["event"], test_labels_full["time"], preds_rand)[0])
            random_ablation_c_index.append(np.mean(iter_scores))

        all_runs_gen_coh.append(gen_selective_c_index_coh)
        all_runs_gen_multi.append(gen_selective_c_index_multi)
        all_runs_random.append(random_ablation_c_index)

    # --- 3c. Aggregate Results ---
    print("\nAggregating results...")
    # Return raw scores from all runs
    results = {
        'random': all_runs_random,
        'coherent': all_runs_gen_coh,
        'multi': all_runs_gen_multi
    }
    return results

# =============================================================================
# 4. PLOTTING FUNCTION (Adapted for Survival)
# =============================================================================
def plot_multi_panel_prioritization(
    results_dict: Dict[str, Dict],
    ablation_steps: np.ndarray,
    figsize: Tuple[float, float] = (16, 7),
    savepath: Optional[str] = None
):
    """
    Generates a polished multi-panel line plot to compare prioritization impact.
    """
    # --- 1. Setup ---
    modalities_to_plot = list(results_dict.keys())
    n_panels = len(modalities_to_plot)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=False) # Use sharey=False
    if n_panels == 1:
        axes = [axes]

    color_map = {'random': '#d55e00', 'coherent': '#56b4e9', 'multi': '#0072b2'} # Adjusted colors
    legend_labels = {'random': 'Random', 'coherent': 'Informed (Coherent Denoising)', 'multi': 'Informed (Multi-condition)'}
    
    # X-axis is "Ratio of Samples OBSERVED", so we invert the ablation steps
    inverted_steps = 1 - ablation_steps 

    # --- 2. Plotting Loop ---
    for i, modality_key in enumerate(modalities_to_plot):
        ax = axes[i]
        results = results_dict[modality_key]
        
        for method in ['random', 'coherent', 'multi']:
            # Calculate mean and std from the raw run data
            mean = np.mean(results[method], axis=0)
            std = np.std(results[method], axis=0)
            
            ax.plot(inverted_steps, mean, marker='o', linestyle='--', markersize=5,
                    label=legend_labels[method], color=color_map[method], zorder=10)
            ax.fill_between(inverted_steps, mean - std, mean + std, alpha=0.15, color=color_map[method], zorder=5)

        # --- 3. Panel-specific Aesthetics ---
        ax.set_title(f"Impact of Observing {modality_key.upper()}", fontsize=14, weight='bold')
        ax.set_xlabel('Ratio of Samples with Observed Modality', fontsize=12)
        
        ax.yaxis.grid(True, linewidth=0.7, color="#CDCCCC", zorder=0)
        ax.xaxis.grid(True, linewidth=0.7, color="#CDCCCC", zorder=0)
        ax.set_axisbelow(True)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#B0B0B0')
        ax.tick_params(axis='x', colors='#505050', length=0)
        ax.tick_params(axis='y', colors='#505050', length=0)
        ax.xaxis.label.set_color('#303030')
        ax.yaxis.label.set_color('#303030')
        
        ax.set_xlim(1.02, -0.02) # Invert axis from 100% down to 0%
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_xticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.2)], rotation=0)
        ax.tick_params(labelsize=10)
        
        # --- METRIC: Set Y-axis label to C-Index ---
        ax.set_ylabel('Survival Analysis C-Index', fontsize=12)

    # --- 4. Global Aesthetics & Layout ---
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=11, frameon=False, title='Prioritization Strategy')
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(12)
    
    fig.suptitle("Random Prioritization vs. Counterfactual Inference (Survival Analysis)", fontsize=18, weight='bold', y=1.0)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96], w_pad=4) # Adjust layout

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {savepath}")
    plt.show()
    plt.close(fig)


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    # --- Setup ---
    config_args = SimpleNamespace(
        folder='results', metric='mse', dim='32', mask=False,
        labels_dir=labels_path,
        data_dir=train_path, # Use the normalized data path
        surv_dir=surv_dir_path
    )
    
    # Define the experiments to run
    modalities_to_test = [('rnaseq',), ('wsi',)]
    all_results = {}

    for modality_tuple in modalities_to_test:
        # Generate a simple key for the results dictionary
        key = '_'.join(modality_tuple) 
        result = run_ablation_experiment(modality_tuple)
        if result:
            all_results[key] = result
            
    # --- Save & Plot Results ---
    if all_results:
        # --- NEW: Save to new directory ---
        os.makedirs(output_dir, exist_ok=True)
        raw_data_savepath = os.path.join(output_dir, "survival_counterfactual_RAW.json")
        
        print(f"\nSaving raw experiment data to {raw_data_savepath}")
        with open(raw_data_savepath, "w") as f:
            json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
        print("Save complete.")
        
        plot_multi_panel_prioritization(
            results_dict=all_results,
            ablation_steps=ABLATION_STEPS,
            savepath=os.path.join(output_dir, "counterfactual_survival_comparison.png")
        )
    else:
        print("\nNo experiments were successfully completed. Plotting skipped.")
    
    print("\n--- Counterfactual Survival Analysis Finished ---")