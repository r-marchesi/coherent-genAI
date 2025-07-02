
''' Model training functions '''
import copy
import json
import os
import pathlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from lib.sampling import coherent_sample, sample
from lib.get_models import get_diffusion_model
from lib import datasets
from lib.diffusion_models import MultiConditioningDiffusionModelUnet, GaussianDiffusion
import torch.optim as optim
import torch.nn.functional as F
import itertools

def r_squared(y_pred, y_true):
    # Convert to PyTorch tensors if they aren't already
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    # Calculate the mean of y_true
    y_mean = torch.mean(y_true)
    # Calculate total sum of squares (proportional to variance)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    # Calculate residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)
    # Calculate R^2
    r2 = 1 - (ss_res / ss_tot)
    return r2


def test_model(x_test, cond_test, model, diffusion, test_iterations, device='cuda', masks=None):
    x_dim = x_test.shape[1]
    
    col_names = x_test.columns

    # transform test sets into tensors
    x_test = torch.tensor(x_test.values).float().to(device)

    if isinstance(cond_test, list):
        cond_test = [torch.tensor(cond.values).float().to(device) for cond in cond_test]
    else:
        cond_test = torch.tensor(cond_test.values).float().to(device)

    # Handle masks similarly
    if masks is not None:
        masks = [torch.tensor(mask).float().to(device) for mask in masks]
 
    
    mse_list = list()
    r2_list = list()
    cos_list = list()
    generated_samples_list = list()


    model.eval()
    with torch.no_grad():

        for _ in range(test_iterations):
            # generation
            generated_samples = sample(model=model, diffusion=diffusion, cond=cond_test, num_features=x_dim, mask=masks, device=device)
            # update lists
            mse_list.append(F.mse_loss(generated_samples, x_test).item())
            r2_list.append(r_squared(generated_samples, x_test).item())
            cos_list.append(1 - F.cosine_similarity(generated_samples, x_test, dim=1).mean().item())
            generated_samples_list.append(generated_samples.detach().cpu().numpy())

    all_generated = np.concatenate(generated_samples_list, axis=0)
    generated_df = pd.DataFrame(all_generated, columns=col_names)

    metrics = {
        "mse_mean": float(np.mean(mse_list)),
        "mse_std":  float(np.std(mse_list)),
        "r2_mean":  float(np.mean(r2_list)),
        "r2_std":   float(np.std(r2_list)),
        "cos_mean": float(np.mean(cos_list)),
        "cos_std":  float(np.std(cos_list))
    }
    
    return metrics, generated_df





def coherent_test(x_test, cond_test_list, models_list, diffusion, test_iterations, device,  weights_list=None):
    x_dim = x_test.shape[1]
    num_samples = x_test.shape[0]

    col_names = x_test.columns

 
    x_test = torch.tensor(x_test.values).float().to(device)

    cond_test_list = [torch.tensor(cond_test.values).float().to(device) for cond_test in cond_test_list]



    # model testing
    mse_list = list()
    r2_list = list()
    cos_list = list()
    generated_samples_list = list()


    with torch.no_grad(): # assuming that the models are already .to(device).eval()
        for _ in range(test_iterations):
            # generation                        
            generated_samples = coherent_sample(models=models_list, diffusion=diffusion, num_samples=num_samples, 
                                                num_features=x_dim, conds=cond_test_list, device=device, weights=weights_list)
            # update lists
            mse_list.append(F.mse_loss(generated_samples, x_test).item())
            r2_list.append(r_squared(generated_samples, x_test).item())
            cos_list.append(1 - F.cosine_similarity(generated_samples, x_test, dim=1).mean().item())
            generated_samples_list.append(generated_samples.detach().cpu().numpy())

    all_generated = np.concatenate(generated_samples_list, axis=0)
    generated_df = pd.DataFrame(all_generated, columns=col_names)
    
    metrics = {
        "mse_mean": float(np.mean(mse_list)),
        "mse_std":  float(np.std(mse_list)),
        "r2_mean":  float(np.mean(r2_list)),
        "r2_std":   float(np.std(r2_list)),
        "cos_mean": float(np.mean(cos_list)),
        "cos_std":  float(np.std(cos_list))
    }
    
    return metrics, generated_df






def coherent_test_cos_rejection(
    x_test: pd.DataFrame,
    cond_test_list: list[pd.DataFrame],
    models_list: list,
    diffusion,
    test_iterations: int,
    max_retries: int,
    cos_threshold: float = 1.0,   
    prop_threshold: float = 0.1,   
    device: str = 'cuda',
    weights_list: list[float] = None,
):
    """
    Runs coherent_sample + retry logic over the entire test set, repeated `test_iterations` times.
    Aggregates:
     • per-iteration metrics (MSE, R2, cos) → reports mean+std across iterations
     • all generated samples concatenated into one DataFrame
     • detailed cosine-rejection stats per iteration & per sample

    Returns:
      metrics_summary: dict with keys
        - mse_mean, mse_std, r2_mean, r2_std, cos_mean, cos_std
      generated_df:  DataFrame of shape (num_samples * test_iterations, x_dim)
      cos_stats: dict with:
        - trajectories: list of np.ndarray, each (num_samples, timesteps)
        - fracs:        list of np.ndarray, each (num_samples,)
        - retries:      list of np.ndarray, each (num_samples,)  # number of attempts used
    """
    # Prep ground truth & conditioning
    x_dim       = x_test.shape[1]
    num_samples = x_test.shape[0]
    col_names   = x_test.columns

    x_target = torch.tensor(x_test.values, dtype=torch.float32, device=device)
    conds_list = [
        torch.tensor(c.values, dtype=torch.float32, device=device)
        for c in cond_test_list
    ]

    # Hold per-iteration metrics
    all_mse  = []
    all_r2   = []
    all_cos  = []

    # Hold concatenated samples
    all_samples = []

    # Hold detailed cos stats
    trajectories = []  # list of shape (num_samples, T)
    fracs        = []  # list of shape (num_samples,)
    retries_used = []  # list of shape (num_samples,)

    for it in range(test_iterations):
        final, cos_traj, cos_frac, used_retries = _coherent_once(
            x_target, conds_list,
            models_list, diffusion,
            max_retries, cos_threshold, prop_threshold, 
            device, weights_list
        )
        # final: (num_samples, x_dim)
        # cos_traj: (num_samples, T)
        # cos_frac: (num_samples,)
        # used_retries: (num_samples,)

        # compute metrics this iteration
        with torch.no_grad():
            # MSE
            mse_per = F.mse_loss(final, x_target, reduction='none').mean(dim=1)
            all_mse.append(mse_per.mean().item())

            # R2
            res_sq = ((final - x_target)**2).sum(dim=1)
            tot_sq = ((x_target - x_target.mean(dim=1, keepdim=True))**2).sum(dim=1)
            r2_per = 1 - res_sq / (tot_sq + 1e-8)
            all_r2.append(r2_per.mean().item())

            # Cosine-distance
            cos_sim = F.cosine_similarity(final, x_target, dim=1)
            cos_dist = 1 - cos_sim
            all_cos.append(cos_dist.mean().item())

        # collect samples & cos stats
        all_samples.append(final.cpu().numpy())
        trajectories.append(cos_traj.cpu().numpy())
        fracs.append(cos_frac.cpu().numpy())
        retries_used.append(used_retries.cpu().numpy())

    # Build summary metrics
    metrics_summary = {
        "mse_mean": np.mean(all_mse),
        "mse_std":  np.std(all_mse),
        "r2_mean":  np.mean(all_r2),
        "r2_std":   np.std(all_r2),
        "cos_mean": np.mean(all_cos),
        "cos_std":  np.std(all_cos),
    }

    # Concatenate samples into DataFrame
    concatenated = np.concatenate(all_samples, axis=0)
    generated_df = pd.DataFrame(concatenated, columns=col_names)

    # Package cos_stats
    cos_stats = {
        "trajectories": trajectories,    # list length=test_iterations of (num_samples, T)
        "fracs":        fracs,           # list of (num_samples,)
        "retries":      retries_used,    # list of (num_samples,)
    }

    return metrics_summary, generated_df, cos_stats


def _coherent_once(
    x_target: torch.Tensor,
    conds_list: list[torch.Tensor],
    models_list: list,
    diffusion,
    max_retries: int,
    cos_threshold: float,   
    prop_threshold: float,   
    device: str,
    weights_list: list[float] = None
):
    """
    Single‐pass over the dataset with per‐sample retry:
      • returns final_samples,
                cos_traj (num_samples×T),
                cos_frac (num_samples,),
                retries_used (num_samples,)
    """
    num_samples, x_dim = x_target.shape
    # placeholders
    final_samples = torch.zeros_like(x_target)
    best_frac     = torch.ones((num_samples,), device=device)
    best_sample   = torch.zeros_like(x_target)
    retries_used  = torch.zeros((num_samples,), dtype=torch.int, device=device)

    # pending sample indices
    pending = torch.arange(num_samples, device=device)

    # we'll record the *last* trajectory for each slot as well
    last_traj = torch.zeros((num_samples, diffusion.num_timesteps), device=device)

    for attempt in range(1, max_retries + 1):
        if len(pending) == 0:
            break

        # Generate only pending slots
        subsamps, cos_traj = coherent_sample(
            models=models_list,
            diffusion=diffusion,
            num_samples=len(pending),
            num_features=x_dim,
            conds=[c[pending] for c in conds_list],
            device=device,
            return_cos=True,
            weights=weights_list
        )
        # cos_traj: (batch, T)

        # compute fraction of “bad” timesteps using your threshold
        frac_bad = (cos_traj > cos_threshold).float().mean(dim=1) 

        # accept only those with proportion of bad ≤ your prop_threshold
        accept = frac_bad <= prop_threshold 
        acc_idx = pending[accept]
        final_samples[acc_idx] = subsamps[accept]
        last_traj[acc_idx]     = cos_traj[accept]

        # for rejected, see if this is 'best so far'
        rej_idx_local = (~accept).nonzero(as_tuple=False).squeeze(1)
        rej_global    = pending[~accept]
        rej_fracs     = frac_bad[~accept]
        rej_samps     = subsamps[~accept]
        rej_trajs     = cos_traj[~accept]

        for li, gidx in enumerate(rej_global):
            if rej_fracs[li] < best_frac[gidx]:
                best_frac[gidx]   = rej_fracs[li]
                best_sample[gidx] = rej_samps[li]
                last_traj[gidx]   = rej_trajs[li]
        # mark a retry
        retries_used[rej_global] = attempt

        # update pending
        pending = rej_global

    # for any still pending, fill with best_sample
    if len(pending) > 0:
        for gidx in pending:
            final_samples[gidx] = best_sample[gidx]
            # last_traj already set to best
            # retries_used already set to max_retries

    return final_samples, last_traj, (best_frac.cpu()), retries_used.cpu()