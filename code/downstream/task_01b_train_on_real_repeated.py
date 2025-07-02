import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import sys

sys.path.append("../")

from lib.config import modalities_list
from lib.read_data import read_data


def load_labels(data_dir, split):
    """Load tumor type and stage labels for the given split."""
    tumor_type_path = os.path.join(data_dir, f'{split}_cancer_type.csv')
    stage_path = os.path.join(data_dir, f'{split}_stage.csv')
    
    tumor_type = pd.read_csv(tumor_type_path, index_col=0) if os.path.exists(tumor_type_path) else None
    stage = pd.read_csv(stage_path, index_col=0) if os.path.exists(stage_path) else None
    
    return {'tumor_type': tumor_type, 'stage': stage}


def train_rf_classifier(X_train, y_train, label_type, random_state):
    """Train a Random Forest classifier with a specific random state."""
    print(f"Training samples before dropping missing labels for {label_type}: {len(y_train)}")
    valid_idx = ~pd.isna(y_train)
    X_train_clean = X_train[valid_idx]
    y_train_clean = y_train[valid_idx]
    print(f"Training samples after dropping missing labels for {label_type}: {len(y_train_clean)}")
    
    if len(y_train_clean) == 0:
        print(f"Warning: No valid labels for {label_type}")
        return None, None
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_clean)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train_clean, y_train_encoded)
    
    return rf, le


def evaluate_classifier(rf, le, X_test, y_test, label_type):
    """Evaluate the trained classifier on test data."""
    if rf is None:
        return None
    
    valid_idx = ~pd.isna(y_test)
    X_test_clean = X_test[valid_idx]
    y_test_clean = y_test[valid_idx]

    if len(y_test_clean) == 0:
        print(f"Warning: No valid test labels for {label_type}")
        return None
    
    try:
        y_test_encoded = le.transform(y_test_clean)
    except ValueError as e:
        print(f"Warning: Unknown labels in test set for {label_type}: {e}")
        return None

    y_pred = rf.predict(X_test_clean)
    
    return {
        'balanced_accuracy': balanced_accuracy_score(y_test_encoded, y_pred),
        'f1_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        'n_samples': len(y_test_encoded),
        'le': le
    }


def evaluate_modality_single_run(modality, modalities_map, labels_train, labels_test, results_base_path, dim, random_state, repetition_idx):
    """Evaluate one modality for a single repetition."""
    print(f"\n{'='*50}")
    print(f"Repetition {repetition_idx+1}/10 | Modality: {modality} | Seed: {random_state}")
    print(f"{'='*50}")

    X_train_real = modalities_map[modality]['train']
    X_test_real = modalities_map[modality]['test']
    train_mask = modalities_map[modality]['mask_train']
    
    X_train_real = X_train_real[train_mask == 1]
    
    y_train_tumor = labels_train['tumor_type'][train_mask == 1].values.ravel()
    y_train_stage = labels_train['stage'][train_mask == 1].values.ravel()
    y_test_tumor = labels_test['tumor_type'].values.ravel()
    y_test_stage = labels_test['stage'].values.ravel()

    results = {'modality': modality, 'tumor_type': {}, 'stage': {}}
    
    # Train classifiers
    rf_tumor, le_tumor = train_rf_classifier(X_train_real, y_train_tumor, 'tumor_type', random_state)
    rf_stage, le_stage = train_rf_classifier(X_train_real, y_train_stage, 'stage', random_state)
    
    # Test on real data
    results['tumor_type']['real'] = evaluate_classifier(rf_tumor, le_tumor, X_test_real, y_test_tumor, 'tumor_type')
    results['stage']['real'] = evaluate_classifier(rf_stage, le_stage, X_test_real, y_test_stage, 'stage')
    
    # Test on synthetic data
    methods = ['coherent', 'multi']
    conditioning_string = '_'.join([m for m in modalities_map.keys() if m != modality])
    
    n_test_samples = len(X_test_real)
    slice_start = repetition_idx * n_test_samples
    slice_end = slice_start + n_test_samples

    for method in methods:
        synth_path = os.path.join(results_base_path, dim, f'{modality}_from_{method}/test/generated_samples_from_{conditioning_string}_best_mse.csv')
        if not os.path.exists(synth_path):
            print(f"Warning: Synthetic data not found at {synth_path}")
            continue
        
        synthetic_data_full = pd.read_csv(synth_path)
        
        # Use the specific slice for this repetition
        synthetic_data_slice = synthetic_data_full.iloc[slice_start:slice_end]

        if len(synthetic_data_slice) != n_test_samples:
            print(f"Warning: Mismatch in synthetic data slice size for {method}. Expected {n_test_samples}, got {len(synthetic_data_slice)}")
            continue

        # Evaluate on the slice using the original test labels
        results['tumor_type'][f'synthetic_from_{method}'] = evaluate_classifier(rf_tumor, le_tumor, synthetic_data_slice, y_test_tumor, f'tumor_type_synth_{method}')
        results['stage'][f'synthetic_from_{method}'] = evaluate_classifier(rf_stage, le_stage, synthetic_data_slice, y_test_stage, f'stage_synth_{method}')
        
    return results


def aggregate_results(all_run_results):
    """Aggregate results from all repetitions to compute mean and std dev."""
    if not all_run_results:
        return {}
        
    aggregated = {}
    
    # Get all keys from the first result
    first_result = all_run_results[0][0]
    modalities = [res['modality'] for res in all_run_results[0]]
    tasks = ['tumor_type', 'stage']
    data_types = list(first_result['tumor_type'].keys())

    for modality in modalities:
        aggregated[modality] = {'tumor_type': {}, 'stage': {}}
        for task in tasks:
            for data_type in data_types:
                # Collect scores from all runs
                b_acc_scores = [run_results[modalities.index(modality)][task][data_type]['balanced_accuracy'] for run_results in all_run_results if run_results[modalities.index(modality)][task].get(data_type)]
                f1_scores = [run_results[modalities.index(modality)][task][data_type]['f1_macro'] for run_results in all_run_results if run_results[modalities.index(modality)][task].get(data_type)]
                
                if not b_acc_scores:
                    continue

                # Calculate mean and std dev
                agg_data = {
                    'balanced_accuracy_mean': np.mean(b_acc_scores),
                    'balanced_accuracy_std': np.std(b_acc_scores),
                    'f1_macro_mean': np.mean(f1_scores),
                    'f1_macro_std': np.std(f1_scores),
                    'n_samples': all_run_results[0][modalities.index(modality)][task][data_type]['n_samples'],
                    'n_repetitions': len(b_acc_scores)
                }
                aggregated[modality][task][data_type] = agg_data

    return aggregated


def print_summary_results(aggregated_results):
    """Print a summary of the aggregated results."""
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS (Mean ± Std Dev over 10 repetitions)")
    print(f"{'='*80}")
    
    for modality, mod_results in aggregated_results.items():
        print(f"\n{modality.upper()}:")
        for task in ['tumor_type', 'stage']:
            print(f"  {task.replace('_', ' ').title()} Classification:")
            if task in mod_results:
                for data_type, results in mod_results[task].items():
                    print(f"    {data_type:<25}: "
                          f"Bal Acc = {results['balanced_accuracy_mean']:.3f} ± {results['balanced_accuracy_std']:.3f}, "
                          f"Macro F1 = {results['f1_macro_mean']:.3f} ± {results['f1_macro_std']:.3f} "
                          f"(n={results['n_samples']}, reps={results['n_repetitions']})")


def save_summary_csv(aggregated_results, output_dir):
    """Save the aggregated results to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    for modality, mod_results in aggregated_results.items():
        for task in ['tumor_type', 'stage']:
            if task in mod_results:
                for data_type, res in mod_results[task].items():
                    summary_rows.append({
                        'modality': modality,
                        'task': task,
                        'data_type': data_type,
                        'balanced_accuracy_mean': res['balanced_accuracy_mean'],
                        'balanced_accuracy_std': res['balanced_accuracy_std'],
                        'f1_macro_mean': res['f1_macro_mean'],
                        'f1_macro_std': res['f1_macro_std'],
                        'n_samples': res['n_samples'],
                        'n_repetitions': res['n_repetitions']
                    })

    summary_df = pd.DataFrame(summary_rows)
    output_path = os.path.join(output_dir, 'summary_results_10_runs.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"\nAggregated results saved to {output_path}")


if __name__ == '__main__':
    # Parameters
    dim = '32'
    results_path = '../../results'
    labels_dir = "../../datasets_TCGA/downstream_labels/"
    data_dir = '../../datasets_TCGA/07_normalized/'
    output_dir = '../../results/downstream/task_01_train_on_real'
    
    N_REPETITIONS = 10
    random_seeds = [i for i in range(N_REPETITIONS)]
    
    print("Loading data...")
    modalities_map = read_data(
        modalities=modalities_list,
        splits=['train', 'test'],
        data_dir=data_dir,
        dim=dim,
        mask_train_path=f'../../datasets_TCGA/06_masked/{dim}/masks_train.csv'
    )
    labels_train = load_labels(labels_dir, 'train')
    labels_test = load_labels(labels_dir, 'test')
    
    print(f"Loaded data for modalities: {list(modalities_map.keys())}")
    
    all_run_results = []
    
    # Run evaluations for N repetitions
    for i in range(N_REPETITIONS):
        repetition_results = []
        for modality in modalities_map.keys():
            result = evaluate_modality_single_run(
                modality, modalities_map, labels_train, labels_test,
                results_path, dim, random_seeds[i], i
            )
            repetition_results.append(result)
        all_run_results.append(repetition_results)
    
    # Aggregate and save results
    aggregated_results = aggregate_results(all_run_results)
    
    # Save raw results from all runs to JSON for detailed inspection
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'evaluation_results_10_runs.json'), 'w') as f:
        json.dump(all_run_results, f, indent=2, default=str)

    # Print and save the final aggregated summary
    print_summary_results(aggregated_results)
    save_summary_csv(aggregated_results, output_dir)