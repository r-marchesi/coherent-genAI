import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from itertools import combinations
import warnings
import torch
import json
import pathlib
from types import SimpleNamespace
import sys

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Add your custom library path ---
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) 
from lib.test import coherent_test_cos_rejection, test_model
from lib.config import modalities_list
from lib.get_models import get_diffusion_model
from lib.diffusion_models import GaussianDiffusion

# --- SECTION 1: HELPER FUNCTIONS (UNCHANGED) ---
def load_and_prepare_pancancer_data(labels_dir: str, data_dir: str):
    print("--- Loading and preparing all pan-cancer data... ---")
    X_train_orig = pd.read_csv(os.path.join(data_dir, "real_data_train.csv"), index_col=0)
    X_test_orig = pd.read_csv(os.path.join(data_dir, "test_data.csv"), index_col=0)
    train_stage_df = pd.read_csv(os.path.join(labels_dir, "train_stage.csv"), index_col=0)
    train_type_df = pd.read_csv(os.path.join(labels_dir, "train_cancer_type.csv"), index_col=0)
    test_stage_df = pd.read_csv(os.path.join(labels_dir, "test_stage.csv"), index_col=0)
    test_type_df = pd.read_csv(os.path.join(labels_dir, "test_cancer_type.csv"), index_col=0)
    train_labels_combined = train_stage_df.join(train_type_df).dropna(subset=['stage', 'cancertype'])
    test_labels_combined = test_stage_df.join(test_type_df).dropna(subset=['stage', 'cancertype'])
    train_common_idx = train_labels_combined.index.intersection(X_train_orig.index)
    test_common_idx = test_labels_combined.index.intersection(X_test_orig.index)
    X_train = X_train_orig.loc[train_common_idx].sort_index()
    y_train = train_labels_combined.loc[train_common_idx, 'stage'].sort_index()
    train_types = train_labels_combined.loc[train_common_idx, 'cancertype'].sort_index()
    X_test = X_test_orig.loc[test_common_idx].sort_index()
    y_test = test_labels_combined.loc[test_common_idx, 'stage'].sort_index()
    test_types = test_labels_combined.loc[test_common_idx, 'cancertype'].sort_index()
    print(f"  Found {len(X_train)} training samples and {len(X_test)} test samples across all cancers.")
    train_cancer_dummies = pd.get_dummies(train_types, prefix='cancer')
    test_cancer_dummies = pd.get_dummies(test_types, prefix='cancer')
    train_cancer_dummies, test_cancer_dummies = train_cancer_dummies.align(test_cancer_dummies, join='outer', axis=1, fill_value=0)
    X_train_final = pd.concat([X_train, train_cancer_dummies], axis=1)
    X_test_final = pd.concat([X_test, test_cancer_dummies], axis=1)
    return X_train_final, y_train, X_test_final, y_test

def load_single_model(target_mod, cond_mod, diffusion, config_args, device):
    path = pathlib.Path(f'../../{config_args.folder}/{config_args.dim}/{target_mod}_from_{cond_mod}')
    ckpt_path = path / f'train/best_by_{config_args.metric}.pth'
    if not ckpt_path.exists(): raise FileNotFoundError(f"Checkpoint not found for single model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    config = SimpleNamespace(**ckpt["config"])
    x_dim = config_args.modality_dims[target_mod]
    cond_dim = config_args.modality_dims[cond_mod]
    model = get_diffusion_model(config.architecture, diffusion, config, x_dim=x_dim, cond_dims=cond_dim).to(device)
    model.load_state_dict(ckpt[f"best_model_{config_args.metric}"])
    model.eval()
    return model, config, ckpt['best_loss']

def load_multi_model(target_mod, diffusion, config_args, device):
    base_dir = pathlib.Path(f"../../{config_args.folder}/{config_args.dim}/{target_mod}_from_multi{'_masked' if config_args.mask else ''}")
    ckpt_path = base_dir / 'train' / f'best_by_{config_args.metric}.pth'
    if not ckpt_path.exists(): raise FileNotFoundError(f"Checkpoint not found for multi model: {ckpt_path}")
    with open(base_dir / 'cond_order.json', 'r') as f: cond_order = json.load(f)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    config = SimpleNamespace(**ckpt['config'])
    x_dim = config_args.modality_dims[target_mod]
    cond_dim_list = [config_args.modality_dims[c] for c in cond_order]
    model = get_diffusion_model(config.architecture, diffusion, config, x_dim=x_dim, cond_dims=cond_dim_list).to(device)
    model.load_state_dict(ckpt[f'best_model_{config_args.metric}'])
    model.eval()
    return model, config, cond_order

def impute_missing_modalities(X_test_with_nan, modalities_to_impute, available_modalities, gen_mode, config_args, diffusion, device):
    X_imputed = X_test_with_nan.copy()
    generation_order = sorted(modalities_to_impute)
    conditioning_modalities = [m for m in available_modalities if m not in modalities_to_impute]
    for i, target_mod in enumerate(generation_order):
        print(f"    Imputing '{target_mod}' (step {i+1}/{len(generation_order)}) with '{gen_mode}' model...")
        current_conds = conditioning_modalities + generation_order[:i]
        cond_data_list = []
        for cond_mod in current_conds:
            cond_cols = [c for c in X_imputed.columns if c.startswith(cond_mod + '_')]
            cond_data_list.append(X_imputed[cond_cols])
        if gen_mode == 'coherent':
            models = [load_single_model(target_mod, c, diffusion, config_args, device)[0] for c in current_conds]
            weights = [load_single_model(target_mod, c, diffusion, config_args, device)[2] for c in current_conds]
            _, generated_df, _ = coherent_test_cos_rejection(
                pd.DataFrame(np.zeros((X_imputed.shape[0], config_args.modality_dims[target_mod]))), 
                cond_data_list, models, diffusion, test_iterations=1, max_retries=10, 
                device=device, weights_list=weights
            )
        elif gen_mode == 'multi':
            model, _, cond_order = load_multi_model(target_mod, diffusion, config_args, device)
            final_cond_list = []
            for c_name in cond_order:
                if c_name in current_conds:
                    cond_cols = [c for c in X_imputed.columns if c.startswith(c_name + '_')]
                    final_cond_list.append(X_imputed[cond_cols])
                else: 
                    shape = (X_imputed.shape[0], config_args.modality_dims[c_name])
                    final_cond_list.append(pd.DataFrame(np.zeros(shape)))
            _, generated_df = test_model(
                pd.DataFrame(np.zeros((X_imputed.shape[0], config_args.modality_dims[target_mod]))),
                final_cond_list, model, diffusion, test_iterations=1, device=device
            )
        target_cols = [c for c in X_imputed.columns if c.startswith(target_mod + '_')]
        generated_df.columns = target_cols
        generated_df.index = X_imputed.index
        X_imputed[target_cols] = generated_df
    return X_imputed

def get_metrics(y_true, y_pred):
    """Helper function to calculate and return all desired metrics."""
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro_f1 = report['macro avg']['f1-score']
    return balanced_acc, macro_f1

# --- SECTION 2: MAIN ANALYSIS PIPELINE  ---

def run_pancancer_analysis_with_imputation(random_seed: int, config_args, diffusion, device):
    X_train, y_train, X_test, y_test = load_and_prepare_pancancer_data(
        config_args.labels_dir, config_args.data_dir
    )
    if X_train is None: return None

    print(f"\n--- Training pan-cancer model with random_state={random_seed}... ---")
    classifier = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)
    classifier.fit(X_train, y_train)
    
    all_prefixes = {col.split('_')[0] for col in X_test.columns if '_' in col}
    possible_modalities = ['cna', 'rnaseq', 'rppa', 'wsi'] 
    available_modalities = sorted([m for m in possible_modalities if m in all_prefixes])
    print(f"  Available modalities for ablation/imputation: {available_modalities}")

    all_results = []
    
    # --- Step 1 - Evaluate on the full, unmodified test set first ---
    print("\n--- Processing Test Condition: full_data ---")
    y_pred_full = classifier.predict(X_test)
    b_acc, f1 = get_metrics(y_test, y_pred_full)
    all_results.append({
        'test_condition': 'full_data',
        'test_type': 'full_data',
        'balanced_accuracy': b_acc,
        'macro_f1_score': f1
    })

    # --- Step 2 - Loop through combinations of modalities to remove/impute ---
    for r in range(1, len(available_modalities) + 1):
        for combo in combinations(available_modalities, r):
            
            # NEW: Rename the final condition for clarity
            if len(combo) == len(available_modalities):
                condition_name = "cancer_label_only"
            else:
                condition_name = f"no_{'_'.join(combo)}"
                
            modalities_to_process = list(combo)
            print(f"\n--- Processing Test Condition: {condition_name} ---")

            X_test_ablated = X_test.copy()
            cols_to_nullify = [col for mod in modalities_to_process for col in X_test.columns if col.startswith(mod + '_')]
            X_test_ablated[cols_to_nullify] = np.nan
            
            y_pred_ablated = classifier.predict(X_test_ablated)
            b_acc, f1 = get_metrics(y_test, y_pred_ablated)
            all_results.append({
                'test_condition': condition_name,
                'test_type': 'ablation',
                'balanced_accuracy': b_acc,
                'macro_f1_score': f1
            })

            if len(modalities_to_process) < len(available_modalities):
                for gen_mode in ['multi', 'coherent']:
                    X_test_imputed = impute_missing_modalities(
                        X_test_ablated, modalities_to_process, available_modalities, 
                        gen_mode, config_args, diffusion, device
                    )
                    y_pred_imputed = classifier.predict(X_test_imputed)
                    b_acc, f1 = get_metrics(y_test, y_pred_imputed)
                    all_results.append({
                        'test_condition': condition_name,
                        'test_type': f'imputed_{gen_mode}',
                        'balanced_accuracy': b_acc,
                        'macro_f1_score': f1
                    })
            else:
                print(f"  Skipping generative imputation for {condition_name}.")
                for gen_mode in ['multi', 'coherent']:
                    all_results.append({
                        'test_condition': condition_name,
                        'test_type': f'imputed_{gen_mode}',
                        'balanced_accuracy': np.nan,
                        'macro_f1_score': np.nan
                    })

    return pd.DataFrame(all_results)



if __name__ == '__main__':
    config_args = SimpleNamespace(
        folder='results', metric='mse', dim='32', mask=False,
        labels_dir="../../datasets_TCGA/downstream_labels",
        data_dir="./data_task_02",
        modality_dims={'cna': 32, 'rnaseq': 32, 'rppa': 32, 'wsi': 32}
    )
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    diffusion = GaussianDiffusion(num_timesteps=1000).to(device)
    N_RUNS = 10
    all_run_dfs = []

    for i in range(N_RUNS):
        print(f"\n{'='*25} Starting Run {i+1}/{N_RUNS} {'='*25}")
        results_df = run_pancancer_analysis_with_imputation(
            random_seed=i, config_args=config_args, diffusion=diffusion, device=device
        )
        if results_df is not None:
            results_df['run'] = i + 1
            all_run_dfs.append(results_df)

    if all_run_dfs:
        final_results = pd.concat(all_run_dfs, ignore_index=True)
        final_results.dropna(subset=['balanced_accuracy', 'macro_f1_score'], inplace=True)

        results_path = '../../results/downstream/task_05_imputing_test_set'
        # Ensure the results directory exists
        os.makedirs(results_path, exist_ok=True)

        # Save the final results to a CSV file
        final_results.to_csv(os.path.join(results_path, f'results_{str(N_RUNS)}_runs.csv'), index=False)


        print("\n\n===== SUMMARY STATISTICS ACROSS ALL RUNS =====")
        summary_stats = final_results.groupby(['test_condition', 'test_type'])[['balanced_accuracy', 'macro_f1_score']].agg(
            ['mean', 'std', 'median']
        )
        # Define a logical sort order for the summary table
        summary_stats['n_removed'] = summary_stats.index.get_level_values('test_condition').map(
            lambda x: 0 if x == 'full_data' else (x.count('_') + 2 if x == 'cancer_label_only' else x.count('_') + 1)
        )
        print(summary_stats.sort_values(by=['n_removed', ('balanced_accuracy', 'mean')], ascending=[True, False]).drop(columns='n_removed').to_string())

