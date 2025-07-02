import itertools
import json
import pathlib
import random
from types import SimpleNamespace

import sys
sys.path.append('../') 

from lib.test import coherent_test_cos_rejection, test_model
from lib.config import modalities_list
from lib.get_models import get_diffusion_model
from lib.diffusion_models import GaussianDiffusion

from lib.read_data import read_data
import argparse

import numpy as np
import pandas as pd
import torch

# --- Settings ---
device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Cascade generation of modalities")
parser.add_argument('--mode', type=str, choices=['multi', 'coherent'], required=True,
                    help="Specify the generation mode: 'multi' or 'coherent'")
parser.add_argument('--folder', type=str, default='results', help="Root folder for trained models and results")
parser.add_argument('--metric', type=str, choices=['mse', 'cosine', 'timestep'], default='mse',
                    help="Specify the metric to determine the best trained model")
parser.add_argument('--dim', type=str, default='32', help="Input dimension of the data")
parser.add_argument('--test_repeats', type=int, default=10, help="Number of repetitions for the test set")
parser.add_argument('--mask', action='store_true', help="Use the multi conditioning models trained with extra masks (for 'multi' mode)")

args = parser.parse_args()

mask_train_path = '../../datasets_TCGA/06_masked/32/masks_train.csv'
data_dir = '../../datasets_TCGA/07_normalized'

# --- Modality and Data Setup ---
# Define which modalities are inputs and which are targets for the cascade
input_modalities = ['cna', 'rppa']
target_modalities = ['wsi', 'rnaseq']

# Validate that the defined input/target modalities match the master list from config
all_defined_modalities = set(input_modalities + target_modalities)
if not all_defined_modalities.issubset(set(modalities_list)):
    raise ValueError(f"Defined modalities {all_defined_modalities} are not a subset of modalities_list from config: {modalities_list}")
if len(all_defined_modalities) != len(modalities_list):
     raise ValueError(f"The number of defined modalities ({len(all_defined_modalities)}) does not match the number in modalities_list ({len(modalities_list)})")


# Load data for all modalities specified in the config's modalities_list
print(f"Loading data for all modalities in config: {modalities_list}")
modalities_map = read_data(
    modalities=modalities_list,
    splits=['test'],
    data_dir=data_dir,
    dim=args.dim,
    mask_train_path=mask_train_path
)
print("Data loaded successfully.")

# --- Helper Functions for Model Loading ---

def load_single_model(target_mod, cond_mod, diffusion, config_args):
    """
    Loads a single-condition diffusion model for the 'coherent' method.
    Returns the model, its config, and the best loss value (for weighting).
    """
    path = pathlib.Path(f'../../{config_args.folder}/{config_args.dim}/{target_mod}_from_{cond_mod}')
    ckpt_path = path / f'train/best_by_{config_args.metric}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for single model: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw_cfg = ckpt["config"]
    config = SimpleNamespace(**raw_cfg)
    state_dict = ckpt[f"best_model_{config_args.metric}"]
    best_loss = ckpt['best_loss']

    x_dim = modalities_map[target_mod]['test'].shape[1]
    cond_dim = modalities_map[cond_mod]['test'].shape[1]

    model = get_diffusion_model(
        config.architecture,
        diffusion,
        config,
        x_dim=x_dim,
        cond_dims=cond_dim
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, best_loss


def load_multi_model(target_mod, diffusion, config_args):
    """
    Loads a multi-condition diffusion model for the 'multi' method.
    Returns the model, its config, and the conditioning order from training.
    """
    base_dir = pathlib.Path(f"../../{config_args.folder}/{config_args.dim}/{target_mod}_from_multi{'_masked' if config_args.mask else ''}")
    ckpt_path = base_dir / 'train' / f'best_by_{config_args.metric}.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for multi model: {ckpt_path}")

    # This file stores the exact order of conditioning modalities used during training.
    cond_order_path = base_dir / 'cond_order.json'
    with open(cond_order_path, 'r') as f:
        cond_order = json.load(f)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw_cfg = ckpt['config']
    config = SimpleNamespace(**raw_cfg)
    state_dict = ckpt[f'best_model_{config_args.metric}']

    x_dim = modalities_map[target_mod]['test'].shape[1]
    cond_dim_list = [modalities_map[c]['test'].shape[1] for c in cond_order]

    model = get_diffusion_model(
        config.architecture,
        diffusion,
        config,
        x_dim=x_dim,
        cond_dims=cond_dim_list
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, cond_order


# --- Main Cascade Generation Logic ---

# Define the two possible generation orders for the target modalities
possible_orders = list(itertools.permutations(target_modalities))

# Lists to store results from all iterations
all_run_results = []
all_run_metrics = []


for i in range(args.test_repeats):
    # For each iteration, alternate the generation order
    first_target, second_target = possible_orders[i % len(possible_orders)]
    
    print(f"\n{'='*20} RUNNING ITERATION {i + 1}/{args.test_repeats} | ORDER: {first_target} -> {second_target} {'='*20}")
    
    diffusion = GaussianDiffusion(num_timesteps=1000).to(device)
    num_samples = modalities_map[first_target]['test'].shape[0]

    # ---
    # STEP 1: Generate `first_target` from `input_modalities`
    # ---
    print(f"\n--- Step 1: Generating '{first_target}' from {input_modalities} using '{args.mode}' mode ---")
    
    x_test_step1 = modalities_map[first_target]['test']
    
    if args.mode == 'coherent':
        cond_test_list_s1 = [modalities_map[c]['test'] for c in input_modalities]
        models_list_s1 = []
        weights_list_s1 = []
        for cond_mod in input_modalities:
            model, _, weight = load_single_model(first_target, cond_mod, diffusion, args)
            models_list_s1.append(model)
            weights_list_s1.append(weight)

        metrics_s1, generated_first_target_df, _ = coherent_test_cos_rejection(
            x_test_step1, cond_test_list_s1, models_list_s1, diffusion,
            test_iterations=1, max_retries=10, device=device, weights_list=weights_list_s1
        )

    elif args.mode == 'multi':
        model_s1, _, cond_order_s1 = load_multi_model(first_target, diffusion, args)
        
        cond_test_list_s1 = []
        masks_s1 = []
        for cond_name in cond_order_s1:
            if cond_name in input_modalities:
                cond_test_list_s1.append(modalities_map[cond_name]['test'])
                if args.mask:
                    masks_s1.append(np.ones(num_samples))
            else:
                shape = modalities_map[cond_name]['test'].shape
                cond_test_list_s1.append(pd.DataFrame(np.zeros(shape), columns=modalities_map[cond_name]['test'].columns))
                if args.mask:
                    masks_s1.append(np.zeros(num_samples))
        
        metrics_s1, generated_first_target_df = test_model(
            x_test_step1, cond_test_list_s1, model_s1, diffusion,
            test_iterations=1, device=device, masks=masks_s1 if args.mask else None
        )

    # ---
    # STEP 2: Generate `second_target` from `input_modalities` + generated `first_target`
    # ---
    conds_for_step2 = input_modalities + [first_target]
    print(f"\n--- Step 2: Generating '{second_target}' from {conds_for_step2} using '{args.mode}' mode ---")

    x_test_step2 = modalities_map[second_target]['test']

    if args.mode == 'coherent':
        cond_test_list_s2 = [modalities_map[c]['test'] for c in input_modalities] + [generated_first_target_df]
        models_list_s2 = []
        weights_list_s2 = []
        for cond_mod in conds_for_step2:
            model, _, weight = load_single_model(second_target, cond_mod, diffusion, args)
            models_list_s2.append(model)
            weights_list_s2.append(weight)

        metrics_s2, generated_second_target_df, _ = coherent_test_cos_rejection(
            x_test_step2, cond_test_list_s2, models_list_s2, diffusion,
            test_iterations=1, max_retries=10, device=device, weights_list=weights_list_s2
        )

    elif args.mode == 'multi':
        model_s2, _, cond_order_s2 = load_multi_model(second_target, diffusion, args)
        
        cond_test_list_s2 = []
        masks_s2 = []
        for cond_name in cond_order_s2:
            if cond_name in input_modalities:
                cond_test_list_s2.append(modalities_map[cond_name]['test'])
                if args.mask:
                    masks_s2.append(np.ones(num_samples))
            elif cond_name == first_target:
                cond_test_list_s2.append(generated_first_target_df)
                if args.mask:
                    masks_s2.append(np.ones(num_samples))
            else:
                shape = modalities_map[cond_name]['test'].shape
                cond_test_list_s2.append(pd.DataFrame(np.zeros(shape), columns=modalities_map[cond_name]['test'].columns))
                if args.mask:
                    masks_s2.append(np.zeros(num_samples))
        
        metrics_s2, generated_second_target_df = test_model(
            x_test_step2, cond_test_list_s2, model_s2, diffusion,
            test_iterations=1, device=device, masks=masks_s2 if args.mask else None
        )

    # --- Store results for this iteration ---
    run_data = {
        'iteration': i + 1,
        'generation_order': f"{first_target}->{second_target}",
        first_target: generated_first_target_df,
        second_target: generated_second_target_df
    }
    all_run_results.append(run_data)

    run_metrics = {
        'iteration': i + 1,
        'generation_order': f"{first_target}->{second_target}",
        f'metrics_step1_gen_{first_target}': metrics_s1,
        f'metrics_step2_gen_{second_target}': metrics_s2
    }
    all_run_metrics.append(run_metrics)
    print(f"--- Iteration {i+1} completed and results stored. ---")


# --- Consolidate and Save All Results ---
print("\nConsolidating all generated data...")

final_results_list = []
for run_data in all_run_results:
    # Ensure dataframes are in the canonical order (wsi, then rnaseq) for consistent combining
    # and add prefixes to the column names for clarity.
    df_wsi = run_data['wsi'].add_prefix('wsi_')
    df_rnaseq = run_data['rnaseq'].add_prefix('rnaseq_')
    
    # Combine the two dataframes side-by-side
    combined_df = pd.concat([df_wsi, df_rnaseq], axis=1)
    
    # Add metadata
    combined_df['iteration'] = run_data['iteration']
    combined_df['generation_order'] = run_data['generation_order']
    
    final_results_list.append(combined_df)

# Concatenate all iterations into a single dataframe
final_df = pd.concat(final_results_list, ignore_index=True)

# Reorder columns to have metadata first
meta_cols = ['iteration', 'generation_order']
data_cols = [col for col in final_df.columns if col not in meta_cols]
final_df = final_df[meta_cols + data_cols]

# Create a directory for the cascade results
output_dir = pathlib.Path(f'./data_cascade')
output_dir.mkdir(parents=True, exist_ok=True)

# Save the consolidated data
output_data_path = output_dir / f'cascade_{args.mode}.csv'
final_df.to_csv(output_data_path, index=False)
print(f"All generated data saved to: {output_data_path}")

# Save the consolidated metrics
output_metrics_path = output_dir / f'cascade_{args.mode}_all_metrics.json'
with open(output_metrics_path, 'w') as f:
    json.dump(all_run_metrics, f, indent=4)
print(f"All metrics saved to: {output_metrics_path}")


print("\nCascade testing completed successfully.")