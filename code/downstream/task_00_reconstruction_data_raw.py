import pandas as pd
import numpy as np
import json

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# This list defines all the experiments to run. 
# You MUST fill in the file paths for the experiments you want to process.
# - target_modality: The modality being predicted (e.g., 'cna').
# - source_label: A name for the source data used for generation.
# - real_data_path: Path to the TRUE test data for the target modality.
# - generated_data_path: Path to the CSV file containing the 10x concatenated
#                        generated samples for this experiment.


target_real_cna = '../../datasets_TCGA/07_normalized/32/cna_test.csv'
target_real_rnaseq = '../../datasets_TCGA/07_normalized/32/rnaseq_test.csv'
target_real_rppa = '../../datasets_TCGA/07_normalized/32/rppa_test.csv'
target_real_wsi = '../../datasets_TCGA/07_normalized/32/wsi_test.csv'

EXPERIMENTS = [
    # --- Target: CNA ---
    {'target_modality': 'cna', 'source_label': 'rnaseq', 'real_data_path': target_real_cna, 'generated_data_path': '../../results/32/cna_from_rnaseq/test/generated_samples_best_mse.csv'},
    {'target_modality': 'cna', 'source_label': 'rppa', 'real_data_path': target_real_cna, 'generated_data_path': '../../results/32/cna_from_rppa/test/generated_samples_best_mse.csv'},
    {'target_modality': 'cna', 'source_label': 'wsi', 'real_data_path': target_real_cna, 'generated_data_path': '../../results/32/cna_from_wsi/test/generated_samples_best_mse.csv'},
    {'target_modality': 'cna', 'source_label': 'Coherent', 'real_data_path': target_real_cna, 'generated_data_path': '../../results/32/cna_from_coherent/test/generated_samples_from_rnaseq_rppa_wsi_best_mse.csv'},
    {'target_modality': 'cna', 'source_label': 'Multi', 'real_data_path': target_real_cna, 'generated_data_path': '../../results/32/cna_from_multi/test/generated_samples_from_rnaseq_rppa_wsi_best_mse.csv'},

    # --- Target: RNAseq ---
    {'target_modality': 'rnaseq', 'source_label': 'cna', 'real_data_path': target_real_rnaseq, 'generated_data_path': '../../results/32/rnaseq_from_cna/test/generated_samples_best_mse.csv'},
    {'target_modality': 'rnaseq', 'source_label': 'rppa', 'real_data_path': target_real_rnaseq, 'generated_data_path': '../../results/32/rnaseq_from_rppa/test/generated_samples_best_mse.csv'},
    {'target_modality': 'rnaseq', 'source_label': 'wsi', 'real_data_path': target_real_rnaseq, 'generated_data_path': '../../results/32/rnaseq_from_wsi/test/generated_samples_best_mse.csv'},
    {'target_modality': 'rnaseq', 'source_label': 'Coherent', 'real_data_path': target_real_rnaseq, 'generated_data_path': '../../results/32/rnaseq_from_coherent/test/generated_samples_from_cna_rppa_wsi_best_mse.csv'},
    {'target_modality': 'rnaseq', 'source_label': 'Multi', 'real_data_path': target_real_rnaseq, 'generated_data_path': '../../results/32/rnaseq_from_multi/test/generated_samples_from_cna_rppa_wsi_best_mse.csv'},
    
    # --- Target: RPPA ---
    {'target_modality': 'rppa', 'source_label': 'cna', 'real_data_path': target_real_rppa, 'generated_data_path': '../../results/32/rppa_from_cna/test/generated_samples_best_mse.csv'},
    {'target_modality': 'rppa', 'source_label': 'rnaseq', 'real_data_path': target_real_rppa, 'generated_data_path': '../../results/32/rppa_from_rnaseq/test/generated_samples_best_mse.csv'},
    {'target_modality': 'rppa', 'source_label': 'wsi', 'real_data_path': target_real_rppa, 'generated_data_path': '../../results/32/rppa_from_wsi/test/generated_samples_best_mse.csv'},
    {'target_modality': 'rppa', 'source_label': 'Coherent', 'real_data_path': target_real_rppa, 'generated_data_path': '../../results/32/rppa_from_coherent/test/generated_samples_from_cna_rnaseq_wsi_best_mse.csv'},
    {'target_modality': 'rppa', 'source_label': 'Multi', 'real_data_path': target_real_rppa, 'generated_data_path': '../../results/32/rppa_from_multi/test/generated_samples_from_cna_rnaseq_wsi_best_mse.csv'},

    # --- Target: WSI ---
    {'target_modality': 'wsi', 'source_label': 'cna', 'real_data_path': target_real_wsi, 'generated_data_path': '../../results/32/wsi_from_cna/test/generated_samples_best_mse.csv'},
    {'target_modality': 'wsi', 'source_label': 'rnaseq', 'real_data_path': target_real_wsi, 'generated_data_path': '../../results/32/wsi_from_rnaseq/test/generated_samples_best_mse.csv'},
    {'target_modality': 'wsi', 'source_label': 'rppa', 'real_data_path': target_real_wsi, 'generated_data_path': '../../results/32/wsi_from_rppa/test/generated_samples_best_mse.csv'},
    {'target_modality': 'wsi', 'source_label': 'Coherent', 'real_data_path': target_real_wsi, 'generated_data_path': '../../results/32/wsi_from_coherent/test/generated_samples_from_cna_rnaseq_rppa_best_mse.csv'},
    {'target_modality': 'wsi', 'source_label': 'Multi', 'real_data_path': target_real_wsi, 'generated_data_path': '../../results/32/wsi_from_multi/test/generated_samples_from_cna_rnaseq_rppa_best_mse.csv'},
]

N_REPETITIONS = 10
OUTPUT_FILE_PATH = "../../results/downstream/task_00_rsquared/recalculated_r2_scores_RAW.json"


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R-squared (coefficient of determination) score.
    
    Args:
        y_true: The ground truth values.
        y_pred: The predicted values.
        
    Returns:
        The R-squared score.
    """
    # Sum of squared residuals
    ss_res = np.sum((y_true - y_pred) ** 2)
    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Avoid division by zero if the true values are constant
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
        
    r2 = 1 - (ss_res / ss_tot)
    return r2


# =============================================================================
# 3. MAIN EXECUTION SCRIPT
# =============================================================================
def main():
    """
    Main function to loop through experiments and recalculate raw R-squared scores.
    """
    all_raw_scores = {}

    print("Starting R-squared score recalculation process...")

    for experiment in EXPERIMENTS:
        target = experiment['target_modality']
        source = experiment['source_label']
        real_path = experiment['real_data_path']
        gen_path = experiment['generated_data_path']

        # Skip experiments where paths are not filled in
        if not real_path or not gen_path:
            continue

        print(f"\nProcessing: Target='{target}', Source='{source}'")

        try:
            # Load the real and generated data
            real_df = pd.read_csv(real_path)
            generated_df_all = pd.read_csv(gen_path)

            # Ensure column order matches, which is crucial for correct calculations
            if 'sample_id' in real_df.columns:
                real_df = real_df.drop(columns=['sample_id'])
            if 'sample_id' in generated_df_all.columns:
                generated_df_all = generated_df_all.drop(columns=['sample_id'])
            
            # Align columns just in case they are out of order
            generated_df_all = generated_df_all[real_df.columns]

        except FileNotFoundError as e:
            print(f"  -> WARNING: File not found for this experiment. Skipping. Details: {e}")
            continue
        except Exception as e:
            print(f"  -> ERROR: Could not process files for this experiment. Skipping. Details: {e}")
            continue

        # Get the number of samples in one repetition (the size of the real test set)
        n_samples_per_rep = len(real_df)
        
        # Verify that the generated data file is a multiple of the real data
        if len(generated_df_all) % n_samples_per_rep != 0:
            print(f"  -> WARNING: Generated data length ({len(generated_df_all)}) is not a multiple of real data length ({n_samples_per_rep}). Skipping.")
            continue
        
        # This should equal N_REPETITIONS (i.e., 10)
        num_reps_in_file = len(generated_df_all) // n_samples_per_rep
        if num_reps_in_file != N_REPETITIONS:
            print(f"  -> WARNING: Expected {N_REPETITIONS} repetitions in file, but found {num_reps_in_file}. Processing anyway.")

        
        repetition_r2_scores = []
        real_values_np = real_df.values

        # Loop through each repetition stored in the concatenated file
        for i in range(num_reps_in_file):
            start_index = i * n_samples_per_rep
            end_index = (i + 1) * n_samples_per_rep
            
            # Slice the dataframe to get the generated data for the current repetition
            generated_rep_df = generated_df_all.iloc[start_index:end_index]
            generated_rep_np = generated_rep_df.values
            
            # Calculate the R-squared score for this repetition
            r2_score = calculate_r_squared(real_values_np, generated_rep_np)
            repetition_r2_scores.append(r2_score)
        
        # Store the list of 10 raw scores
        if target not in all_raw_scores:
            all_raw_scores[target] = {}
        all_raw_scores[target][source] = repetition_r2_scores
        print(f"  -> Successfully calculated {len(repetition_r2_scores)} R-squared scores.")

    # Save the final dictionary of raw scores to a JSON file
    if not all_raw_scores:
        print("\nNo experiments were processed. No output file will be created.")
    else:
        print(f"\nSaving all raw R-squared scores to '{OUTPUT_FILE_PATH}'...")
        with open(OUTPUT_FILE_PATH, 'w') as f:
            json.dump(all_raw_scores, f, indent=4)
        print("Process complete.")


if __name__ == '__main__':
    main()