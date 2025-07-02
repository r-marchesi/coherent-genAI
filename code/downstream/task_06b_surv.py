import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import warnings
import torch
import json
import pathlib
from types import SimpleNamespace
import sys

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
from sklearn.impute import SimpleImputer, KNNImputer

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Add your custom library path ---
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from lib.test import coherent_test_cos_rejection, test_model
from lib.config import modalities_list
from lib.get_models import get_diffusion_model
from lib.diffusion_models import GaussianDiffusion

# ====================================================================================
# SECTION 1: HELPER FUNCTIONS (UNCHANGED)
# ====================================================================================
def to_survival_structured_array(df: pd.DataFrame) -> np.ndarray:
    event_indicator = df['OS'].astype(bool)
    event_time = df['OS.time'].astype(float)
    return np.array(list(zip(event_indicator, event_time)), dtype=[('event', bool), ('time', float)])

def load_and_prepare_pancancer_survival_data(labels_dir: str, data_dir: str, surv_dir: str):
    print("--- Loading and preparing all pan-cancer data for survival analysis... ---")
    X_train_orig = pd.read_csv(os.path.join(data_dir, "real_data_train.csv"), index_col=0)
    X_test_orig = pd.read_csv(os.path.join(data_dir, "test_data.csv"), index_col=0)
    train_type_df = pd.read_csv(os.path.join(labels_dir, "train_cancer_type.csv"), index_col=0)
    test_type_df = pd.read_csv(os.path.join(labels_dir, "test_cancer_type.csv"), index_col=0)
    all_surv_files = [f for f in os.listdir(surv_dir) if f.endswith('.survival.tsv')]
    surv_dfs = [pd.read_csv(os.path.join(surv_dir, fn), sep="\t")
                .rename(columns={"sample": "sample_id"})
                .assign(sample_id=lambda df: df.sample_id.str[:-1])
                .set_index("sample_id").drop(columns=["_PATIENT"], errors='ignore')
                .pipe(lambda df: df[~df.index.duplicated(keep="first")])
                for fn in all_surv_files]
    pan_surv = pd.concat(surv_dfs).dropna(subset=['OS', 'OS.time'])
    train_common_idx = train_type_df.index.intersection(X_train_orig.index).intersection(pan_surv.index)
    test_common_idx = test_type_df.index.intersection(X_test_orig.index).intersection(pan_surv.index)
    X_train = X_train_orig.loc[train_common_idx].sort_index()
    train_surv = pan_surv.loc[train_common_idx].sort_index()
    train_types = train_type_df.loc[train_common_idx, 'cancertype'].sort_index()
    X_test = X_test_orig.loc[test_common_idx].sort_index()
    test_surv = pan_surv.loc[test_common_idx].sort_index()
    test_types = test_type_df.loc[test_common_idx, 'cancertype'].sort_index()
    y_train = to_survival_structured_array(train_surv)
    y_test = to_survival_structured_array(test_surv)
    print(f"  Found {len(X_train)} training samples and {len(X_test)} test samples with survival data.")
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
    x_dim = config_args.modality_dims[target_mod]; cond_dim = config_args.modality_dims[cond_mod]
    model = get_diffusion_model(config.architecture, diffusion, config, x_dim=x_dim, cond_dims=cond_dim).to(device)
    model.load_state_dict(ckpt[f"best_model_{config_args.metric}"]); model.eval()
    return model, config, ckpt['best_loss']

def load_multi_model(target_mod, diffusion, config_args, device):
    base_dir = pathlib.Path(f"../../{config_args.folder}/{config_args.dim}/{target_mod}_from_multi{'_masked' if config_args.mask else ''}")
    ckpt_path = base_dir / 'train' / f'best_by_{config_args.metric}.pth'
    if not ckpt_path.exists(): raise FileNotFoundError(f"Checkpoint not found for multi model: {ckpt_path}")
    with open(base_dir / 'cond_order.json', 'r') as f: cond_order = json.load(f)
    ckpt = torch.load(ckpt_path, map_location='cpu'); config = SimpleNamespace(**ckpt['config'])
    x_dim = config_args.modality_dims[target_mod]
    cond_dim_list = [config_args.modality_dims[c] for c in cond_order]
    model = get_diffusion_model(config.architecture, diffusion, config, x_dim=x_dim, cond_dims=cond_dim_list).to(device)
    model.load_state_dict(ckpt[f'best_model_{config_args.metric}']); model.eval()
    return model, config, cond_order

def impute_missing_modalities_generative(X_test_with_nan, modalities_to_impute, available_modalities, gen_mode, config_args, diffusion, device):
    X_imputed = X_test_with_nan.copy()
    generation_order = sorted(modalities_to_impute)
    conditioning_modalities = [m for m in available_modalities if m not in modalities_to_impute]
    for i, target_mod in enumerate(generation_order):
        print(f"    Imputing '{target_mod}' (step {i+1}/{len(generation_order)}) with '{gen_mode}' model...")
        current_conds = conditioning_modalities + generation_order[:i]
        cond_data_list = [X_imputed[[c for c in X_imputed.columns if c.startswith(cond_mod + '_')]] for cond_mod in current_conds]
        if gen_mode == 'coherent':
            models = [load_single_model(target_mod, c, diffusion, config_args, device)[0] for c in current_conds]
            weights = [load_single_model(target_mod, c, diffusion, config_args, device)[2] for c in current_conds]
            _, generated_df, _ = coherent_test_cos_rejection(pd.DataFrame(np.zeros((X_imputed.shape[0], config_args.modality_dims[target_mod]))), cond_data_list, models, diffusion, test_iterations=1, max_retries=10, device=device, weights_list=weights)
        elif gen_mode == 'multi':
            model, _, cond_order = load_multi_model(target_mod, diffusion, config_args, device)
            final_cond_list = []
            for c_name in cond_order:
                if c_name in current_conds:
                    final_cond_list.append(X_imputed[[c for c in X_imputed.columns if c.startswith(c_name + '_')]])
                else:
                    final_cond_list.append(pd.DataFrame(np.zeros((X_imputed.shape[0], config_args.modality_dims[c_name]))))
            _, generated_df = test_model(pd.DataFrame(np.zeros((X_imputed.shape[0], config_args.modality_dims[target_mod]))), final_cond_list, model, diffusion, test_iterations=1, device=device)
        target_cols = [c for c in X_imputed.columns if c.startswith(target_mod + '_')]; generated_df.columns = target_cols
        generated_df.index = X_imputed.index; X_imputed[target_cols] = generated_df
    return X_imputed

# ====================================================================================
# SECTION 2: MAIN ANALYSIS PIPELINE (UNCHANGED)
# ====================================================================================
def run_pancancer_survival_analysis_with_all_imputations(random_seed: int, rsf_params: dict, config_args, diffusion, device):
    X_train, y_train, X_test, y_test = load_and_prepare_pancancer_survival_data(
        config_args.labels_dir, config_args.data_dir, config_args.surv_dir
    )
    current_rsf_params = rsf_params.copy()
    current_rsf_params['random_state'] = random_seed
    print(f"\n--- Training pan-cancer Random Survival Forest with random_state={random_seed}... ---")
    print(f"  Using parameters: { {k:v for k,v in current_rsf_params.items() if k != 'random_state'} }")
    rsf = RandomSurvivalForest(**current_rsf_params)
    rsf.fit(X_train, y_train)
    all_prefixes = {col.split('_')[0] for col in X_test.columns if '_' in col}
    available_modalities = sorted([m for m in ['cna', 'rnaseq', 'rppa', 'wsi'] if m in all_prefixes])
    all_results = []
    risk_scores_full = rsf.predict(X_test)
    c_index_full = concordance_index_censored(y_test["event"], y_test["time"], risk_scores_full)[0]
    all_results.append({'test_condition': 'full_data', 'test_type': 'full_data', 'c_index': c_index_full})
    for r in range(1, len(available_modalities) + 1):
        for combo in combinations(available_modalities, r):
            condition_name = "cancer_label_only" if len(combo) == len(available_modalities) else f"no_{'_'.join(combo)}"
            modalities_to_process = list(combo)
            print(f"\n--- Processing Test Condition: {condition_name} ---")
            X_test_ablated = X_test.copy()
            cols_to_nullify = [col for mod in modalities_to_process for col in X_test.columns if col.startswith(mod + '_')]
            X_test_ablated[cols_to_nullify] = np.nan
            imputation_strategies = {'ablation': X_test_ablated}
            print("    Imputing with 'mean' and 'knn'...")
            mean_imputer = SimpleImputer(strategy='mean').fit(X_train)
            imputation_strategies['imputed_mean'] = pd.DataFrame(mean_imputer.transform(X_test_ablated), index=X_test.index, columns=X_test.columns)
            knn_imputer = KNNImputer(n_neighbors=5).fit(X_train)
            imputation_strategies['imputed_knn'] = pd.DataFrame(knn_imputer.transform(X_test_ablated), index=X_test.index, columns=X_test.columns)
            if len(modalities_to_process) < len(available_modalities):
                imputation_strategies['imputed_multi'] = impute_missing_modalities_generative(X_test_ablated, modalities_to_process, available_modalities, 'multi', config_args, diffusion, device)
                imputation_strategies['imputed_coherent'] = impute_missing_modalities_generative(X_test_ablated, modalities_to_process, available_modalities, 'coherent', config_args, diffusion, device)
            else:
                print("    Skipping generative imputation: no conditioning data available.")
            for test_type, X_test_current in imputation_strategies.items():
                risk_scores = rsf.predict(X_test_current)
                c_index = concordance_index_censored(y_test["event"], y_test["time"], risk_scores)[0]
                all_results.append({'test_condition': condition_name, 'test_type': test_type, 'c_index': c_index})
    return pd.DataFrame(all_results)

# ====================================================================================
# SECTION 3: ORCHESTRATION AND VISUALIZATION (MODIFIED)
# ====================================================================================

# --- MODIFIED: Plotting function now saves the figure instead of showing it ---
def create_survival_summary_plot(data: pd.DataFrame, metric: str, title: str, save_path: str):
    """Creates the final comparison plot for a given metric and saves it to a file."""
    print(f"\n--- Generating plot for: {metric} ---")
    
    condition_to_exclude = (data['test_condition'] == 'cancer_label_only') & (data['test_type'] != 'ablation')
    plot_data = data[~condition_to_exclude].copy()
    plot_data['n_removed'] = plot_data['test_condition'].apply(lambda x: 0 if x == 'full_data' else (99 if x == 'cancer_label_only' else x.count('_') + 1))
    plot_order = plot_data.sort_values(by=['n_removed', 'test_condition']).test_condition.unique()
    palette = {'full_data': '#4C72B0', 'ablation': '#A9A9A9', 'imputed_mean': '#8DEEEE', 'imputed_knn': '#00CED1', 'imputed_multi': '#FFB6C1', 'imputed_coherent': '#DC143C'}
    hue_order = ['full_data', 'ablation', 'imputed_mean', 'imputed_knn', 'imputed_multi', 'imputed_coherent']
    plot_data_hue_order = [h for h in hue_order if h in plot_data['test_type'].unique()]
    
    g = sns.catplot(data=plot_data, x='test_condition', y=metric, hue='test_type', order=plot_order, hue_order=plot_data_hue_order, kind='bar', height=7, aspect=2.2, palette=palette, errorbar='sd')
    sns.move_legend(g, "center right", bbox_to_anchor=(1.1, 0.5), frameon=True, title='Test Type')
    g.fig.suptitle(title, y=1.03, fontsize=18)
    g.set_axis_labels("Test Condition (Modalities Removed)", "Mean Concordance Index (C-Index)", fontsize=14)
    g.set_xticklabels(rotation=45, ha='right')
    
    # --- THIS IS THE CHANGE: Save the figure and close it ---
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig) # Close the figure to free up memory
    print(f"Plot saved to: {save_path}")


def run_full_experiment(experiment_name: str, rsf_params: dict, config_args, diffusion, device):
    """
    Runs the entire N_RUNS experiment for a given set of RSF parameters and saves the results.
    """
    N_RUNS = 10
    all_run_dfs = []
    
    print(f"\n{'='*20} STARTING EXPERIMENT: {experiment_name.upper()} {'='*20}")

    for i in range(N_RUNS):
        print(f"\n--- Starting Run {i+1}/{N_RUNS} for experiment '{experiment_name}' ---")
        results_df = run_pancancer_survival_analysis_with_all_imputations(
            random_seed=i, rsf_params=rsf_params, config_args=config_args, diffusion=diffusion, device=device
        )
        if results_df is not None:
            results_df['run'] = i + 1
            all_run_dfs.append(results_df)

    if all_run_dfs:
        final_results = pd.concat(all_run_dfs, ignore_index=True)
        
        # --- MODIFIED: Ensure results directory exists and define paths ---
        results_dir = '../../results/downstream/task_06_imputing_test_set_surv'
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f'all_imputations_results_{experiment_name}.csv')
        final_results.to_csv(results_path, index=False)
        print(f"\nResults for '{experiment_name}' saved successfully to '{results_path}'")

        summary_stats = final_results.groupby(['test_condition', 'test_type'])['c_index'].agg(['mean', 'std', 'median'])
        summary_stats['n_removed'] = summary_stats.index.get_level_values('test_condition').map(lambda x: 0 if x == 'full_data' else (99 if x == 'cancer_label_only' else x.count('_') + 1))
        print(f"\n===== SUMMARY STATISTICS (C-INDEX) FOR EXPERIMENT: {experiment_name.upper()} =====")
        print(summary_stats.sort_values(by=['n_removed', 'mean'], ascending=[True, False]).drop(columns='n_removed').to_string())

        # --- MODIFIED: Call the plotting function with a save path ---
        plot_path = os.path.join(results_dir, f'survival_comparison_{experiment_name}.png')
        create_survival_summary_plot(final_results, 'c_index', f'Comparison of Imputation Strategies ({experiment_name} RSF) - C-Index', save_path=plot_path)
        
        return final_results
    else:
        print(f"No results were generated for experiment '{experiment_name}'.")
        return None


if __name__ == '__main__':
    config_args = SimpleNamespace(
        folder='results', metric='mse', dim='32', mask=False,
        labels_dir="../../datasets_TCGA/downstream_labels",
        data_dir="./data_task_02",
        surv_dir="../../datasets_TCGA/downstream_labels/survival",
        modality_dims={'cna': 32, 'rnaseq': 32, 'rppa': 32, 'wsi': 32}
    )
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    diffusion = GaussianDiffusion(num_timesteps=1000).to(device)

    RSF_PARAMS_QUICK = {'n_estimators': 30, 'max_depth': 12, 'max_features': 'sqrt', 'n_jobs': -1}
    RSF_PARAMS_LONG = {'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt', 'n_jobs': -1}

    results_quick = run_full_experiment(
        experiment_name='quick_rf', rsf_params=RSF_PARAMS_QUICK,
        config_args=config_args, diffusion=diffusion, device=device
    )
    results_long = run_full_experiment(
        experiment_name='long_rf', rsf_params=RSF_PARAMS_LONG,
        config_args=config_args, diffusion=diffusion, device=device
    )

    if results_quick is not None and results_long is not None:
        print("\n\n" + "="*20 + " FINAL COMPARISON: QUICK vs. LONG RF " + "="*20)
        results_quick['model_type'] = 'Quick RF (30 trees)'
        results_long['model_type'] = 'Long RF (100 trees)'
        
        comparison_df = pd.concat([results_quick, results_long], ignore_index=True)
        
        # --- MODIFIED: Save the final comparison plot ---
        results_dir = '../../results/downstream/task_06_imputing_test_set_surv'
        final_plot_path = os.path.join(results_dir, 'final_comparison_quick_vs_long_rf.png')
        
        create_survival_summary_plot(
            comparison_df, 
            metric='c_index',
            title='Final Comparison: Quick RF vs. Long RF Models',
            save_path=final_plot_path
        )