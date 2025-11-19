import pandas as pd
import numpy as np
import os
import glob

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
RESULTS_DIR = "../../results/32"
DATA_DIR = "../../datasets_TCGA/07_normalized/32"
LABELS_DIR = "../../datasets_TCGA/downstream_labels"
OUTPUT_DIR = "../../results/downstream/task_00_rsquared"

# Define the experiments to process. 
EXPERIMENTS = [
    # --- Target: CNA ---
    {'target': 'cna', 'source': 'rnaseq', 'type': 'single'},
    {'target': 'cna', 'source': 'rppa', 'type': 'single'},
    {'target': 'cna', 'source': 'wsi', 'type': 'single'},
    {'target': 'cna', 'source': 'cna_rnaseq_rppa_wsi', 'type': 'coherent'},
    {'target': 'cna', 'source': 'cna_rnaseq_rppa_wsi', 'type': 'multi'},

    # --- Target: RNAseq ---
    {'target': 'rnaseq', 'source': 'cna', 'type': 'single'},
    {'target': 'rnaseq', 'source': 'rppa', 'type': 'single'},
    {'target': 'rnaseq', 'source': 'wsi', 'type': 'single'},
    {'target': 'rnaseq', 'source': 'cna_rnaseq_rppa_wsi', 'type': 'coherent'},
    {'target': 'rnaseq', 'source': 'cna_rnaseq_rppa_wsi', 'type': 'multi'},

    # --- Target: RPPA ---
    {'target': 'rppa', 'source': 'cna', 'type': 'single'},
    {'target': 'rppa', 'source': 'rnaseq', 'type': 'single'},
    {'target': 'rppa', 'source': 'wsi', 'type': 'single'},
    {'target': 'rppa', 'source': 'cna_rnaseq_rppa_wsi', 'type': 'coherent'},
    {'target': 'rppa', 'source': 'cna_rnaseq_rppa_wsi', 'type': 'multi'},

    # --- Target: WSI ---
    {'target': 'wsi', 'source': 'cna', 'type': 'single'},
    {'target': 'wsi', 'source': 'rnaseq', 'type': 'single'},
    {'target': 'wsi', 'source': 'rppa', 'type': 'single'},
    {'target': 'wsi', 'source': 'cna_rnaseq_rppa_wsi', 'type': 'coherent'},
    {'target': 'wsi', 'source': 'cna_rnaseq_rppa_wsi', 'type': 'multi'},
]

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def calculate_r2(y_true, y_pred, global_mean=None):
    """
    Calculates R2 score. If global_mean is provided, it's used for SS_tot.
    """
    # Residual sum of squares for THIS subset
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    
    if global_mean is not None:
        # Total sum of squares using the GLOBAL mean
        # This ensures the metric is consistent with the global R2
        ss_tot = np.sum((y_true - global_mean) ** 2, axis=0)
    else:
        # Fallback to standard R2 using local mean
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)

    # Handle zero variance features to avoid division by zero
    r2_per_feature = 1 - (ss_res / np.where(ss_tot == 0, 1e-9, ss_tot))
    return np.mean(r2_per_feature)

def get_generated_data_path(base_dir, target, source, exp_type):
    if exp_type == 'single':
        folder = f"{target}_from_{source}"
        return os.path.join(base_dir, folder, "test", "generated_samples_best_mse.csv")
    else:
        folder = f"{target}_from_{exp_type}"
        conds = source.split('_')
        if target in conds: conds.remove(target)
        cond_str = '_'.join(conds)
        return os.path.join(base_dir, folder, "test", f"generated_samples_from_{cond_str}_best_mse.csv")

# =============================================================================
# 3. MAIN ANALYSIS
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading cancer type labels...")
    labels_df = pd.read_csv(os.path.join(LABELS_DIR, "test_cancer_type.csv"), index_col=0)
    cancer_types = labels_df['cancertype'].unique()
    print(f"Found {len(cancer_types)} cancer types.")

    all_results = []

    for exp in EXPERIMENTS:
        target = exp['target']
        source = exp['source']
        exp_type = exp['type']
        
        print(f"\nProcessing: Target={target}, Source={source} ({exp_type})")
        
        # 1. Load Real Data & Calculate Global Mean
        real_path = os.path.join(DATA_DIR, f"{target}_test.csv")
        try:
            real_df = pd.read_csv(real_path, index_col=0)
            current_labels = labels_df.loc[real_df.index]
            
            # --- CRITICAL CHANGE: Calculate global mean of the entire test set ---
            global_mean_vector = np.mean(real_df.values, axis=0)
            
        except FileNotFoundError:
            print(f"  [Error] Real data not found: {real_path}")
            continue

        # 2. Load Generated Data
        gen_path = get_generated_data_path(RESULTS_DIR, target, source, exp_type)
        try:
            gen_df_full = pd.read_csv(gen_path)
            if 'sample_id' in gen_df_full.columns:
                gen_df_full = gen_df_full.drop(columns=['sample_id'])
        except FileNotFoundError:
             print(f"  [Error] Generated data not found: {gen_path}")
             continue

        n_test_samples = len(real_df)
        n_repeats = len(gen_df_full) // n_test_samples
        if n_repeats < 1: continue

        # 3. Calculate R2 per Cancer Type using GLOBAL mean
        for ct in cancer_types:
            ct_samples = current_labels[current_labels['cancertype'] == ct].index
            if len(ct_samples) < 2: continue

            ct_indices = [real_df.index.get_loc(idx) for idx in ct_samples]
            real_data_ct = real_df.iloc[ct_indices].values
            
            r2_scores = []
            for i in range(n_repeats):
                start_idx = i * n_test_samples
                gen_rep_full = gen_df_full.iloc[start_idx : start_idx + n_test_samples].values
                gen_data_ct = gen_rep_full[ct_indices]
                
                # Pass the global mean vector here
                r2 = calculate_r2(real_data_ct, gen_data_ct, global_mean=global_mean_vector)
                r2_scores.append(r2)
            
            all_results.append({
                'target': target,
                'source': source,
                'type': exp_type,
                'cancer_type': ct,
                'n_samples': len(ct_samples),
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores)
            })

    # 4. Save Results
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Save with a new name to distinguish from the previous version
        output_path = os.path.join(OUTPUT_DIR, "r2_scores_by_cancertype.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("\nNo results to save.")

if __name__ == '__main__':
    main()