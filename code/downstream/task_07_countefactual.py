import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# Define all modalities available in the dataset
ALL_MODALITIES = ['cna', 'rna', 'rppa', 'wsi']

# --- Experiment Parameters ---
N_EXPERIMENT_REPEATS = 10
N_ITERATIONS = 30
ABLATION_STEPS = np.arange(0, 1.05, 0.05)
K_NEIGHBORS = 10

# --- File Paths ---
# IMPORTANT: Make sure these paths are correct for your environment.
gen_path = '../../results/32/'
train_path = '../../datasets_TCGA/07_normalized/32/'
test_path = '../../datasets_TCGA/07_normalized/32/'
labels_path = '../../datasets_TCGA/downstream_labels/'

# Mapping from the ablated modalities tuple to the corresponding generated data file.
GENERATED_DATA_MAPPING_coh = {
    ('rna',): f'{gen_path}rnaseq_from_coherent/test/generated_samples_from_cna_rppa_wsi_best_mse.csv',
    ('wsi',): f'{gen_path}wsi_from_coherent/test/generated_samples_from_cna_rnaseq_rppa_best_mse.csv',
}

GENERATED_DATA_MAPPING_multi = {
    ('rna',): f'{gen_path}rnaseq_from_multi/test/generated_samples_from_cna_rppa_wsi_best_mse.csv',
    ('wsi',): f'{gen_path}wsi_from_multi/test/generated_samples_from_cna_rnaseq_rppa_best_mse.csv',
}


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def clean_and_unify_tables(df_dict, modalities):
    """
    Unifies multiple modality-specific dataframes into a single dataframe.
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

# =============================================================================
# 3. CORE EXPERIMENT FUNCTION
# =============================================================================
def run_ablation_experiment(ablated_modalities: Tuple[str, ...]):
    """
    Runs the full progressive ablation experiment for a given set of ablated modalities.
    
    Args:
        ablated_modalities: A tuple containing the names of modalities to treat as missing.

    Returns:
        A dictionary containing the raw F1 scores for each method across all runs.
    """
    print(f"\n{'='*30}\nRUNNING EXPERIMENT FOR ABLATED MODALITY: {str(ablated_modalities).upper()}\n{'='*30}")

    # --- 3a. Data Loading and Preparation ---
    print("Loading datasets...")
    train_raw = {
        'cna': pd.read_csv(f'{train_path}cna_train.csv', sep=','),
        'rna': pd.read_csv(f'{train_path}rnaseq_train.csv', sep=','),
        'rppa': pd.read_csv(f'{train_path}rppa_train.csv', sep=','),
        'wsi': pd.read_csv(f'{train_path}wsi_train.csv', sep=','),
    }

    test_raw = {
        'cna': pd.read_csv(f'{test_path}cna_test.csv', sep=','),
        'rna': pd.read_csv(f'{test_path}rnaseq_test.csv', sep=','),
        'rppa': pd.read_csv(f'{test_path}rppa_test.csv', sep=','),
        'wsi': pd.read_csv(f'{test_path}wsi_test.csv', sep=','),
    }

    train_stage = pd.read_csv(f'{labels_path}train_stage.csv')
    test_stage = pd.read_csv(f'{labels_path}test_stage.csv')

    print("Processing and unifying tables...")
    train_full = clean_and_unify_tables(train_raw, ALL_MODALITIES)
    test_full = clean_and_unify_tables(test_raw, ALL_MODALITIES)

    train_mask = ~train_stage['stage'].isna()
    train_data = train_full[train_mask]
    train_labels_filtered = train_stage['stage'][train_mask]

    test_mask = ~test_stage['stage'].isna()
    test_data_full = test_full[test_mask].reset_index(drop=True)
    test_labels_filtered = test_stage['stage'][test_mask].reset_index(drop=True)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels_filtered)
    test_labels = le.transform(test_labels_filtered)

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
        
        # A new model is trained in each run with a different random_state
        rf = RandomForestClassifier(n_estimators=100, random_state=42 + run, oob_score=True, n_jobs=-1)
        rf.fit(train_data.values, train_labels)

        # Progressive Ablation Analysis
        test_data = test_data_full.copy()
        ablated_cols = [c for c in test_data.columns if c.split('_')[0] in ablated_modalities]
        
        # Prepare generated candidates
        n_test_samples = len(test_full)
        generated_candidates_coh = {mod: [df.iloc[np.arange(K_NEIGHBORS) * n_test_samples + i] for i in range(n_test_samples)] for mod, df in gen_data_long_coh.items()}
        generated_candidates_multi = {mod: [df.iloc[np.arange(K_NEIGHBORS) * n_test_samples + i] for i in range(n_test_samples)] for mod, df in gen_data_long_multi.items()}
        
        original_indices = test_mask[test_mask].index
        filtered_gen_candidates_coh = {mod: [generated_candidates_coh[mod][i] for i in original_indices] for mod in ablated_modalities}
        filtered_gen_candidates_multi = {mod: [generated_candidates_multi[mod][i] for i in original_indices] for mod in ablated_modalities}
        
        # Calculate prediction variances
        prediction_variances_gen_coh, prediction_variances_gen_multi = [], []
        for i in range(len(test_data)):
            original_sample = test_data.iloc[[i]]
            ablated_sample = original_sample.copy()
            ablated_sample[ablated_cols] = 0 # Use 0 for nan to avoid sklearn warning
            ablated_prediction = rf.predict(ablated_sample.values)[0]
            
            prediction_input_coh = pd.concat([original_sample] * K_NEIGHBORS, ignore_index=True)
            prediction_input_multi = pd.concat([original_sample] * K_NEIGHBORS, ignore_index=True)
            for mod in ablated_modalities:
                mod_cols = [c for c in test_data.columns if c.startswith(f'{mod}_')]
                prediction_input_coh[mod_cols] = filtered_gen_candidates_coh[mod][i].values
                prediction_input_multi[mod_cols] = filtered_gen_candidates_multi[mod][i].values
            
            generated_predictions_coh = rf.predict(prediction_input_coh.values)
            generated_predictions_multi = rf.predict(prediction_input_multi.values)
            prediction_variances_gen_coh.append(np.mean((generated_predictions_coh != ablated_prediction)))
            prediction_variances_gen_multi.append(np.mean((generated_predictions_multi != ablated_prediction)))

        variance_df_gen_coh = pd.DataFrame({'variance': prediction_variances_gen_coh, 'original_index': test_data.index}).sort_values(by='variance', ascending=True)
        variance_df_gen_multi = pd.DataFrame({'variance': prediction_variances_gen_multi, 'original_index': test_data.index}).sort_values(by='variance', ascending=True)

        # Calculate F1 scores for all methods
        gen_selective_f1_scores_coh, gen_selective_f1_scores_multi, random_ablation_f1_scores = [], [], []

        for ratio in ABLATION_STEPS:
            # Selective Ablation (Coherent)
            data_for_pred_coh = test_data.copy()
            n_to_ablate = int(len(data_for_pred_coh) * ratio)
            ablation_indices = variance_df_gen_coh.head(n_to_ablate)['original_index'].values
            if len(ablation_indices) > 0:
                data_for_pred_coh.loc[ablation_indices, ablated_cols] = 0
            preds_coh = rf.predict(data_for_pred_coh.values)
            gen_selective_f1_scores_coh.append(f1_score(test_labels, preds_coh, average='weighted', zero_division=0))

            # Selective Ablation (Multi)
            data_for_pred_multi = test_data.copy()
            ablation_indices = variance_df_gen_multi.head(n_to_ablate)['original_index'].values
            if len(ablation_indices) > 0:
                data_for_pred_multi.loc[ablation_indices, ablated_cols] = 0
            preds_multi = rf.predict(data_for_pred_multi.values)
            gen_selective_f1_scores_multi.append(f1_score(test_labels, preds_multi, average='weighted', zero_division=0))

            # Random Ablation
            iter_scores = []
            for _ in range(N_ITERATIONS):
                data_for_pred_rand = test_data.copy()
                ablation_indices = np.random.choice(data_for_pred_rand.index, size=n_to_ablate, replace=False)
                if len(ablation_indices) > 0:
                    data_for_pred_rand.loc[ablation_indices, ablated_cols] = 0
                preds_rand = rf.predict(data_for_pred_rand.values)
                iter_scores.append(f1_score(test_labels, preds_rand, average='weighted', zero_division=0))
            random_ablation_f1_scores.append(np.mean(iter_scores))

        all_runs_gen_coh.append(gen_selective_f1_scores_coh)
        all_runs_gen_multi.append(gen_selective_f1_scores_multi)
        all_runs_random.append(random_ablation_f1_scores)

    # --- 3c. Return Raw Results ---
    # **** MODIFICATION 1: Return the raw data from all runs, not the aggregate. ****
    raw_results = {
        'random': all_runs_random,
        'coherent': all_runs_gen_coh,
        'multi': all_runs_gen_multi
    }
    return raw_results



# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    modalities_to_test = [('rna',), ('wsi',)]
    all_results = {}

    for modality_tuple in modalities_to_test:
        key = '_'.join(modality_tuple) 
        result = run_ablation_experiment(modality_tuple)
        if result:
            all_results[key] = result
    
    # **** MODIFICATION 2: Save raw data, then create summary dict for plotting ****
    if all_results:
        # Save the raw data to a new file for statistical analysis
        raw_data_savepath = "../../results/downstream/task_07_counterfactual/rna_wsi_results_RAW.json"
        print(f"\nSaving raw experiment data to {raw_data_savepath}")
        with open(raw_data_savepath, "w") as f:
            # The 'default' argument is not needed here if results are already lists
            json.dump(all_results, f, indent=2)
        print("Save complete.")

    else:
        print("\nNo experiments were successfully completed. Plotting skipped.")