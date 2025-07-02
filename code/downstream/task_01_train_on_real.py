# for each modality
    # train RF classifier on real train data to predict the tumortype
    # train RF classifier on real train data to predict the stage

    # test on real test data
    # test on synthetic test data (generated with other test modalities)

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append("../")  

from lib.config import modalities_list
from lib.read_data import read_data

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def load_labels(data_dir, split):
    """Load tumor type and stage labels for the given split."""

    # If labels are in separate files
    tumor_type_path = os.path.join(data_dir, f'{split}_cancer_type.csv')
    stage_path = os.path.join(data_dir, f'{split}_stage.csv')
    
    tumor_type = pd.read_csv(tumor_type_path, index_col=0) if os.path.exists(tumor_type_path) else None
    stage = pd.read_csv(stage_path, index_col=0) if os.path.exists(stage_path) else None
    
    return {'tumor_type': tumor_type, 'stage': stage}


def train_rf_classifier(X_train, y_train, label_type, random_state=42):
    """Train a Random Forest classifier."""
    # Print number of samples before and after dropping missing labels
    print(f"Training samples before dropping missing labels for {label_type}: {len(y_train)}")
    # Handle missing labels
    valid_idx = ~pd.isna(y_train)
    X_train_clean = X_train[valid_idx]
    y_train_clean = y_train[valid_idx]
    print(f"Training samples after dropping missing labels for {label_type}: {len(y_train_clean)}")
    
    if len(y_train_clean) == 0:
        print(f"Warning: No valid labels for {label_type}")
        return None, None
    
    # Encode labels if they're strings
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_clean)

    
    # Train RF classifier
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
    
    # Handle missing labels
    valid_idx = ~pd.isna(y_test)

    
    X_test_clean = X_test[valid_idx]
    y_test_clean = y_test[valid_idx]

    if len(y_test_clean) == 0:
        print(f"Warning: No valid test labels for {label_type}")
        return None
    
    # Encode test labels
    if le is not None:
        try:
            y_test_encoded = le.transform(y_test_clean)
        except ValueError as e:
            print(f"Warning: Unknown labels in test set for {label_type}: {e}")
            return None
    else:
        y_test_encoded = y_test_clean

    
    # Predict
    y_pred = rf.predict(X_test_clean)
    y_pred_proba = rf.predict_proba(X_test_clean) if hasattr(rf, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test_encoded, y_pred) 
    f1_macro    = f1_score(y_test_encoded, y_pred, average='macro')
    f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')
        
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'n_samples': len(y_test_encoded),
        'y_pred_proba': y_pred_proba,
        'y_test': y_test_clean,
        'y_test_encoded': y_test_encoded,
    }





def evaluate_modality(modality, modalities_map, labels_train, labels_test, results_base_path, dim):
    """Evaluate one modality: train classifiers and test on real + synthetic data."""
    print(f"\n{'='*50}")
    print(f"Evaluating modality: {modality}")
    print(f"{'='*50}")
    
    # Get real training and test data
    X_train_real = modalities_map[modality]['train']
    X_test_real = modalities_map[modality]['test']
    
    # Get masks for filtering
    train_mask = modalities_map[modality]['mask_train']
    
    # Filter samples where modality is present
    X_train_real = X_train_real[train_mask == 1]
    
    # Get corresponding labels
    y_train_tumor = labels_train['tumor_type'][train_mask == 1].values.ravel()
    y_train_stage = labels_train['stage'][train_mask == 1].values.ravel()
    y_test_tumor = labels_test['tumor_type'].values.ravel()
    y_test_stage = labels_test['stage'].values.ravel()
    
    results = {
        'modality': modality,
        'n_train_samples': len(X_train_real),
        'n_test_samples': len(X_test_real),
        'tumor_type': {},
        'stage': {}
    }
    
    # Train classifiers
    print(f"Training classifiers for {modality}...")
    rf_tumor, le_tumor = train_rf_classifier(X_train_real, y_train_tumor, 'tumor_type')
    rf_stage, le_stage = train_rf_classifier(X_train_real, y_train_stage, 'stage')

    results['labels_maps_tumor_type'] = le_tumor.classes_.tolist() 
    results['labels_maps_stage'] = le_stage.classes_.tolist() 
    
    # Test on real data
    print(f"Testing on real data...")
    real_tumor_results = evaluate_classifier(rf_tumor, le_tumor, X_test_real, y_test_tumor, 'tumor_type')
    real_stage_results = evaluate_classifier(rf_stage, le_stage, X_test_real, y_test_stage, 'stage')


    
    results['tumor_type']['real'] = real_tumor_results
    results['stage']['real'] = real_stage_results
    
    methods = ['coherent', 'multi']

    condtioning_modalities = [m for m in modalities_map.keys() if m != modality]
    conditioning_string = '_'.join(condtioning_modalities)
    
    for method in methods:
        print(f"Testing synthetic {modality} generated from {method}...")
        
        # Load synthetic data
        synth_path = os.path.join(results_base_path, dim, f'{modality}_from_{method}/test/generated_samples_from_{conditioning_string}_best_mse.csv')
        synthetic_data = pd.read_csv(synth_path)


        # Tile the test labels
        reps = int(len(synthetic_data) / len(y_test_tumor))
        y_synth_tumor = np.tile(y_test_tumor, reps)
        y_synth_stage = np.tile(y_test_stage, reps)


        # overall synthetic eval
        synthetic_tumor_results = evaluate_classifier(
            rf_tumor, le_tumor,
            synthetic_data,
            y_synth_tumor,
            f'tumor_type_synthetic_{modality}_from_{method}'
        )
        synthetic_stage_results = evaluate_classifier(
            rf_stage, le_stage,
            synthetic_data,
            y_synth_stage,
            f'stage_synthetic_{modality}_from_{method}'
        )
        
        results['tumor_type'][f'synthetic_from_{method}'] = synthetic_tumor_results
        results['stage'][f'synthetic_from_{method}'] = synthetic_stage_results

    return results


def print_summary_results(all_results):
    """Print a summary of all results."""
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    
    for modality_results in all_results:
        modality = modality_results['modality']
        print(f"\n{modality.upper()}:")
        print(f"  Training samples: {modality_results['n_train_samples']}")
        print(f"  Test samples: {modality_results['n_test_samples']}")
        
        # Tumor type results
        print(f"  Tumor Type Classification:")  
        for data_type, results in modality_results['tumor_type'].items():
            if results:
                print(f"      {data_type}: acc={results['accuracy']:.3f}, bal_acc={results['balanced_accuracy']:.3f}, F1={results['f1_macro']:.3f}    (n={results['n_samples']})")



        # Stage results
        print(f"  Stage Classification:")
        for data_type, results in modality_results['stage'].items():
            if data_type in ['real','synthetic_from_coherent','synthetic_from_multi'] and results:
                print(f"      {data_type}: acc={results['accuracy']:.3f}, bal_acc={results['balanced_accuracy']:.3f}, F1={results['f1_macro']:.3f}    (n={results['n_samples']})")




def save_results(all_results, output_dir):
    """Save detailed results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")


def save_summary_csv(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # --- 1) build overall summary (no by_tumor rows) ---
    overall_rows = []
    for mod_result in all_results:
        modality = mod_result['modality']
        for task in ['tumor_type', 'stage']:
            for data_type, res in mod_result[task].items():
                # skip any of the new by_tumor dicts
                if task == 'stage' and data_type.startswith('by_tumor'):
                    continue
                if not res:
                    continue
                overall_rows.append({
                    'modality':          modality,
                    'task':              task,
                    'data_type':         data_type,
                    'accuracy':          res['accuracy'],
                    'balanced_accuracy': res['balanced_accuracy'],
                    'f1_macro':          res['f1_macro'],
                    'f1_weighted':       res['f1_weighted'],
                    'n_samples':         res['n_samples']
                })

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(os.path.join(output_dir, 'summary_results.csv'), index=False)


    # --- 2) build by-tumor summary (only the breakdown rows) ---
    by_tumor_rows = []
    for mod_result in all_results:
        modality = mod_result['modality']
        # only the stage task has these breakdowns
        for data_type, breakdown in mod_result['stage'].items():
            if not data_type.startswith('by_tumor'):
                continue
            # breakdown is a dict: tumor_label -> metrics
            for tumor_label, subres in breakdown.items():
                if not subres:
                    continue
                by_tumor_rows.append({
                    'modality':          modality,
                    'data_type':         data_type,       # e.g. "by_tumor" or "by_tumor_synth_coherent"
                    'tumor_type_label':  tumor_label,
                    'accuracy':          subres['accuracy'],
                    'balanced_accuracy': subres['balanced_accuracy'],
                    'f1_macro':          subres['f1_macro'],
                    'f1_weighted':       subres['f1_weighted'],
                    'n_samples':         subres['n_samples']
                })

    by_tumor_df = pd.DataFrame(by_tumor_rows)
    by_tumor_df.to_csv(os.path.join(output_dir, 'summary_by_tumor.csv'), index=False)

    print(f"Written:\n • summary_results.csv ({len(overall_df)} rows)\n • summary_by_tumor.csv ({len(by_tumor_df)} rows)")





def darker_color(color, factor=0.6):
    """Return a darker shade of the given matplotlib color."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb * factor)

def get_split_probs(y_true_enc, y_pred_proba):
    """Return arrays of correct_frac, incorrect_frac per class."""
    y_pred = np.argmax(y_pred_proba, axis=1)
    correct = (y_pred == y_true_enc)
    n = len(y_true_enc)
    n_classes = y_pred_proba.shape[1]

    correct_counts = np.zeros(n_classes)
    incorrect_counts = np.zeros(n_classes)
    for cls in range(n_classes):
        mask = (y_pred == cls)
        correct_counts[cls] = np.sum(mask & correct)
        incorrect_counts[cls] = np.sum(mask & ~correct)
    return correct_counts / n, incorrect_counts / n


def save_plots(all_results, output_dir):
    """Generate and save bar plots of class distributions per modality and label type."""
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        'true': "#e6c700",
        'real': "#e66000",
        'synthetic_from_coherent': "#56b4e9",
        'synthetic_from_multi': "#0072b2"
    }

    for result in all_results:
        for label_type in ['tumor_type', 'stage']:
            class_names = result[f'labels_maps_{label_type}']
            n_classes = len(class_names)

            # True distribution from the real test set
            y_real_enc = np.array(result[label_type]['real']['y_test_encoded'])
            true_dist = np.bincount(y_real_enc, minlength=n_classes) / len(y_real_enc)

            groups = [
                ('True', None, color_map['true']),
                ('Real Pred', 'real', color_map['real']),
                ('Coherent Pred', 'synthetic_from_coherent', color_map['synthetic_from_coherent']),
                ('Multi Pred', 'synthetic_from_multi', color_map['synthetic_from_multi']),
            ]

            # Bar geometry
            width = 0.25
            cluster_width = width * len(groups)
            gap = width
            x = np.arange(n_classes) * (cluster_width + gap)
            offsets = np.linspace(-cluster_width/2 + width/2,
                                  cluster_width/2 - width/2,
                                  len(groups))

            plt.figure(figsize=(16, 6))

            for (label, key, color), dx in zip(groups, offsets):
                if key is None:
                    plt.bar(x + dx, true_dist, width,
                            label=label,
                            color=color,
                            edgecolor='black',
                            linewidth=1.2)
                else:
                    pred_result = result[label_type].get(key)
                    if pred_result is None:
                        continue

                    y_enc = np.array(pred_result['y_test_encoded'])
                    y_proba = np.array(pred_result['y_pred_proba'])

                    corr, incorr = get_split_probs(y_enc, y_proba)

                    dark = darker_color(color)
                    plt.bar(x + dx, corr, width,
                            label=None,
                            color=color,
                            edgecolor='black',
                            linewidth=0.8)
                    plt.bar(x + dx, incorr, width,
                            bottom=corr,
                            label=None,
                            color=dark,
                            edgecolor='black',
                            linewidth=0.8)

            # Labels and legend
            plt.xticks(x, class_names, rotation=45, ha='right')
            plt.xlabel(f'{label_type.capitalize()} Class')
            plt.ylabel('Proportion / Probability')
            plt.title(f"{result['modality']} | True vs Predicted Distribution for {label_type.replace('_', ' ').title()}")

            # Main legend
            main_legend = [
                Patch(facecolor=color_map['true'], edgecolor='black', label='Ground Truth'),
                Patch(facecolor=color_map['real'], edgecolor='black', label='Real Test Set'),
                Patch(facecolor=color_map['synthetic_from_coherent'], edgecolor='black', label='Coherent Generated Test Set'),
                Patch(facecolor=color_map['synthetic_from_multi'], edgecolor='black', label='Multi-Condition Generated Test Set'),
            ]

            # Tone legend
            tone_legend = [
                Patch(facecolor='#cccccc', edgecolor='black', label='Correct (lighter tone)'),
                Patch(facecolor='#555555', edgecolor='black', label='Incorrect (darker tone)'),
            ]

            # Get current axis
            ax = plt.gca()

            # Add second legend (tone explanation)
            legend2 = ax.legend(handles=tone_legend, ncol=1, fontsize='small',
                                bbox_to_anchor=(1.02, 0.7), loc='upper left', title="Correctness of Prediction")

            # Add first legend (prediction source)
            legend1 = ax.legend(handles=main_legend, ncol=1, fontsize='small',
                                bbox_to_anchor=(1.02, 1), loc='upper left', title="Prediction Source")



            # Manually add the first legend back to the axes (so it's not overwritten)
            ax.add_artist(legend2)

            plt.tight_layout()

            # Save figure
            fname = f"{result['modality']}_{label_type}_distribution_plot.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close()

    print(f"Plots saved to {output_dir}")




if __name__ == '__main__':

    # parameters
    dim = '32'    

    results_path = '../../results'    
    labels_dir = "../../datasets_TCGA/downstream_labels/"
    data_dir = '../../datasets_TCGA/07_normalized/'

    output_dir = '../../results/downstream/task_01_train_on_real'
    output_dir_images = '../../results/downstream/task_01_train_on_real/images'
    
    print("Loading data...")
    # Load modality data
    modalities_map = read_data(
        modalities=modalities_list,
        splits=['train', 'test'],
        data_dir=data_dir,
        dim=dim,
        mask_train_path=f'../../datasets_TCGA/06_masked/{dim}/masks_train.csv')
    
    # Load labels
    labels_train = load_labels(labels_dir, 'train')
    labels_test = load_labels(labels_dir, 'test')
    
    print(f"Loaded data for modalities: {list(modalities_map.keys())}")
    
    # Run evaluations sequentially
    print("Running evaluation...")
    all_results = []
    for modality in modalities_map.keys():
        result = evaluate_modality(
            modality, modalities_map, labels_train, labels_test, 
            results_path, dim
        )
        all_results.append(result)
    
    # Print and save results
    print_summary_results(all_results)
    save_results(all_results, output_dir)
    save_summary_csv(all_results, output_dir)
    save_plots(all_results, output_dir_images)  