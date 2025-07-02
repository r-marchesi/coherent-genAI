
''' Model training functions '''
import copy
import json
import os
import pathlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from lib.sampling import sample
from lib import datasets
from lib.diffusion_models import  GaussianDiffusion
from lib.get_models import get_diffusion_model
from lib.metrics import r_squared, compute_pca_metrics
import torch.optim as optim
import torch.nn.functional as F

import itertools




''' Training Function'''

def train_diffusion_model(
    dataloader, 
    x_dim,
    cond_dim,
    x_val,
    cond_val,
    config, #SimpleNamespace object 
    mask_val=None,
    device='cuda',
    
):
    """
    Train the diffusion model with conditioning
    
    Args:
        dataloader (DataLoader): DataLoader for training data.
        x_dim (int): Dimension of the input data.
        cond_dim (int): Dimension(s) of the conditioning data.
        x_val (torch.Tensor): Validation data.
        cond_val (torch.Tensor): Validation conditioning data.
        config (SimpleNamespace): Configuration object with training parameters.
        architecture (str): Architecture type (eg: 'unet', ...).
        mask_val (torch.Tensor, optional): Mask for validation data. Defaults to None.
        pca (PCA, optional): PCA object for computing metrics. Defaults to None.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
    """

    # Initialize diffusion process
    diffusion = GaussianDiffusion(num_timesteps=1000).to(device)

    model = get_diffusion_model(
        config.architecture, 
        diffusion,
        config,
        x_dim=x_dim,
        cond_dims=cond_dim,
        fusion_method='concat'	
    ).to(device)

    print(model)


    model.to(device)

    #print(model)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    val_mse_list = list()
    val_r2_list = list()
    val_cosine_list = list()

    val_mse_timestep_list = list()
    val_r2_timestep_list = list()
    val_cosine_timestep_list = list()

    train_errors_list = list()
    best_val_mse = np.inf
    best_val_mse_timestep = np.inf
    best_val_cosine = np.inf


    # Training loop
    for epoch in range(config.num_epochs):
        total_loss = 0
        model.train()
        
        for x_batch, cond_batch, *mask_batch in dataloader:

            
            x_batch = x_batch.to(device)
            
            #  check if it's a list
            if isinstance(cond_batch, list):
                cond_batch = [c.to(device) for c in cond_batch]
            else:
                cond_batch = cond_batch.to(device)

            # Handle mask_batch similarly
            if mask_batch:  # Check if mask_batch exists
                if isinstance(mask_batch[0], list):
                    mask_batch = [m.to(device) for m in mask_batch[0]]
                else:
                    mask_batch = mask_batch[0].to(device)
            else:
                mask_batch = None
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Sample random timesteps
            batch_size = x_batch.size(0)
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,)).to(device)
            
            # Generate noise
            noise = torch.randn_like(x_batch).to(device)
            
            # Add noise to the input
            x_noisy = diffusion.q_sample(x_batch, t, noise)

            
            # Predict noise
            predicted_noise = model(x_noisy, t, cond_batch, mask_batch)

            
            # Compute loss (MSE between predicted and actual noise)
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        if (epoch+1) % config.validation_epochs == 0:
            model.eval()

            with torch.no_grad():
                
                # Generate samples from the model using all timesteps
                generated_samples = sample(model=model, diffusion=diffusion, cond=cond_val, num_features=x_dim, mask=mask_val ,device=device)
                
                val_r2 = r_squared(generated_samples, x_val).item()
                val_r2_list.append(val_r2)
                
                val_mse = F.mse_loss(generated_samples, x_val).item()
                val_mse_list.append(val_mse)
                
                val_cosine = 1 - F.cosine_similarity(generated_samples, x_val, dim=1).mean().item()
                val_cosine_list.append(val_cosine)


                train_errors_list.append(total_loss / len(dataloader))
                print(f'Validation Loss at Epoch {epoch+1}: {val_mse:.4f}') 
                
                #  Generate samples from the model using the current timesteps
                t = torch.randint(0, diffusion.num_timesteps, (x_val.shape[0],)).to(device)
                
                # Generate noise
                noise = torch.randn_like(x_val).to(device)
                
                # Add noise to the input
                x_noisy = diffusion.q_sample(x_val, t, noise)
                
                # Predict noise
                predicted_noise = model(x_noisy, t, cond_val, mask_val)
                
                # Compute loss (MSE between predicted and actual noise)
                val_mse_timestep = F.mse_loss(predicted_noise, noise).item()
                val_mse_timestep_list.append(val_mse_timestep)
                val_r2_timestep = r_squared(predicted_noise, noise).item()
                val_r2_timestep_list.append(val_r2_timestep)
                val_cosine_timestep = 1- F.cosine_similarity(predicted_noise, noise, dim=1).mean().item()
                val_cosine_timestep_list.append(val_cosine_timestep)

                print(f"Epoch {epoch+1} | Val MSE: {val_mse:.4f} | Val Cosine: {val_cosine:.4f} | Val timestep MSE: {val_mse_timestep:.4f}")

            if val_mse < best_val_mse:
                best_model_state_mse = copy.deepcopy(model.state_dict())
                best_val_mse = val_mse
            
            if val_cosine < best_val_cosine:
                best_model_state_cosine = copy.deepcopy(model.state_dict())
                best_val_cosine = val_cosine

            if val_mse_timestep < best_val_mse_timestep:
                best_model_state_timestep = copy.deepcopy(model.state_dict())
                best_val_mse_timestep = val_mse_timestep



            model.train()


    models_dict = {
        'best_model_mse': best_model_state_mse,
        'best_model_cosine': best_model_state_cosine,
        'best_model_timestep': best_model_state_timestep
    }
    losses_dict = {
        'train_loss': train_errors_list,
        'val_mse': val_mse_list,
        'val_r2': val_r2_list,
        'val_cosine': val_cosine_list,
        'val_mse_timestep': val_mse_timestep_list,
        'val_r2_timestep': val_r2_timestep_list,
        'val_cosine_timestep': val_cosine_timestep_list
    }
    best_val_losses = {
        'best_val_mse': best_val_mse,
        'best_val_cosine': best_val_cosine,
        'best_val_mse_timestep': best_val_mse_timestep
    }
    return models_dict, best_val_losses, losses_dict #TODO handle returning multiple models, for now we return the best model for mse, cosine and timestep separately




def GridSearch(x_train, cond_train, x_val, cond_val, grid_params, device, res_path, grid_type='random', val_repeats=1):
    print(f'grid running on device: {device}')

    # prepare dims, datasets, etc. (unchanged)
    x_dim, cond_dim = x_train.shape[1], cond_train.shape[1]
    x_val = pd.concat([x_val]*val_repeats, ignore_index=True)
    cond_val = pd.concat([cond_val]*val_repeats, ignore_index=True)
    x_val = torch.tensor(x_val.values).float().to(device)
    cond_val = torch.tensor(cond_val.values).float().to(device)

    res_path = pathlib.Path(f'{res_path}/train')
    res_path.mkdir(parents=True, exist_ok=True)

    # rolling list of all experiments
    all_experiments = []
    summary_file = res_path / 'grid_search_history.json'

    # build experiment list
    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if grid_type == 'random':
        np.random.shuffle(experiments)
        experiments = experiments[:grid_params['max_iter'][0]]

    # initialize three global best records
    best_overall = {
        'mse':      {'loss': np.inf, 'config': None, 'models': None, 'losses_dict': None},
        'cosine':   {'loss': np.inf, 'config': None, 'models': None, 'losses_dict': None},
        'timestep': {'loss': np.inf, 'config': None, 'models': None, 'losses_dict': None},
    }



    for i, exp in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] Params: {exp}")

        # train and get back: {best_model_mse,…}, {best_val_mse,…}, losses_dict
        dataloader = datasets.create_dataloader(x_train, cond_train,
                                                batch_size=exp['batch_size'])
        models_dict, best_val_losses, losses_dict = train_diffusion_model(
            dataloader, x_dim, cond_dim,
            x_val, cond_val, SimpleNamespace(**exp), device=device
        )

        # loop over each primary metric
        for metric, key_name in [
            ('mse',      'best_val_mse'),
            ('cosine',   'best_val_cosine'),
            ('timestep', 'best_val_mse_timestep')
        ]:
            this_loss = best_val_losses[key_name]
            record = best_overall[metric]

            if this_loss < record['loss']:
                # update in-memory
                record.update({
                    'loss':        this_loss,
                    'config':      exp,
                    'models':      models_dict,
                    'losses_dict': losses_dict,
                })
                print(f" ➤ New best “{metric}” = {this_loss:.4f}")

                # immediately persist to disk
                ckpt_file = res_path / f'best_by_{metric}.pth'
                torch.save({
                    'best_model_mse':       models_dict['best_model_mse'],
                    'best_model_cosine':    models_dict['best_model_cosine'],
                    'best_model_timestep':  models_dict['best_model_timestep'],
                    'config':               exp,
                    'best_loss':            this_loss
                }, ckpt_file)

                # save its losses history
                losses_file = res_path / f'best_by_{metric}_losses.json'
                with open(losses_file, 'w') as f:
                    json.dump(losses_dict, f, indent=2)

                print(f"    • checkpoint → {ckpt_file}")
                print(f"    • losses     → {losses_file}")

        
        # append this experiment to the history

        result = {
            'experiment_index': i+1,
            'params': exp,
            'best_val_losses': best_val_losses,
            'losses_history': losses_dict
        }
        all_experiments.append(result)
    
        # overwrite full history after each run
        with open(summary_file, 'w') as f:
            json.dump(all_experiments, f, indent=2)
        print(f" ← saved experiment #{i+1} to {summary_file}")




    # after all experiments, dump the grid-search table
    all_results = []
    for metric, rec in best_overall.items():
        row = {
            'which_best': metric,
            'best_loss':  rec['loss'],
            **rec['config']
        }
        all_results.append(row)
    df = pd.DataFrame(all_results)
    df.to_csv(res_path / 'grid_search_bst_results.csv', index=False)

    print("\nGridSearch complete. Summary saved to:")
    print(f"   • {res_path / 'grid_search_summary.csv'}")
    for m in best_overall:
        print(f"   • best_by_{m}.pth (+ *_losses.json)")


def GridSearchMulti(
    x_train,
    cond_train_list,
    x_val,
    cond_val_list,       
    grid_params,    
    res_path,
    mask_train_list=None,      
    mask_val_list=None,  
    device='cuda',
    grid_type='random',
    val_repeats=10,
    extra_masking=False
):
    
    '''
        x_train, cond_train,: x and conditioning 
        x_val, cond_val: validation sets
        grid_params: dictionary for the gridsearch (from file grid_params.py)
        res_path: path to results folder
        grid_type: 'random' or 'full'
        max_iter: n of training if grid_type = random
        val_repeats : how many times the validation set is repeated (for robustness), like repeating multiple times sampling from the original validation set
        extra_masking: if True, apply additional random masking each epoch

        Return 0 -> store metrics, params and best model.
    '''

    print(f'grid running on device: {device}')


    #extract columns number
    x_dim = x_train.shape[1]
    cond_dim_list = [cond_train.shape[1] for cond_train in cond_train_list]
    
    # repeat validation set and concatenate into a single df
    x_val = pd.concat([x_val]*val_repeats, ignore_index= True)
    cond_val_list = [pd.concat([cond_val]*val_repeats, ignore_index= True) for cond_val in cond_val_list]

    # transform validation sets into tensors
    x_val = torch.tensor(x_val.values).float().to(device)
    cond_val_list = [torch.tensor(cond_val.values).float().to(device) for cond_val in cond_val_list]


    res_path = pathlib.Path(f'{res_path}/train')
    res_path.mkdir(parents=True, exist_ok=True)
    
    # rolling list of all experiments
    all_experiments = []
    summary_file = res_path / 'grid_search_history.json'

    keys, values = zip(*grid_params.items())
    experiments = [dict(zip(keys,v)) for v in itertools.product(*values)] # list of all possible combinations of parameters

    if grid_type == 'random':
        np.random.shuffle(experiments)
        experiments = experiments[:grid_params['max_iter'][0]]


    # initialize three global best records
    best_overall = {
        'mse':      {'loss': np.inf, 'config': None, 'models': None, 'losses_dict': None},
        'cosine':   {'loss': np.inf, 'config': None, 'models': None, 'losses_dict': None},
        'timestep': {'loss': np.inf, 'config': None, 'models': None, 'losses_dict': None},
    }

    for i, exp in enumerate(experiments):
        
        print(f"Training n.{i}")
        dataloader = datasets.create_dataloader(x_df = x_train, cond_df_list = cond_train_list, mask_list = mask_train_list, batch_size=exp['batch_size'], extra_masking=extra_masking)
        models_dict, best_val_losses, losses_dict = train_diffusion_model(dataloader, 
                                                             x_dim, 
                                                             cond_dim_list, 
                                                             x_val, 
                                                             cond_val_list, 
                                                             SimpleNamespace(**exp),
                                                             device=device)
        
        # loop over each primary metric
        for metric, key_name in [
            ('mse',      'best_val_mse'),
            ('cosine',   'best_val_cosine'),
            ('timestep', 'best_val_mse_timestep')
        ]:
            this_loss = best_val_losses[key_name]
            record = best_overall[metric]

            if this_loss < record['loss']:
                # update in-memory
                record.update({
                    'loss':        this_loss,
                    'config':      exp,
                    'models':      models_dict,
                    'losses_dict': losses_dict,
                })
                print(f" ➤ New best “{metric}” = {this_loss:.4f}")

                # immediately persist to disk
                ckpt_file = res_path / f'best_by_{metric}.pth'
                torch.save({
                    'best_model_mse':       models_dict['best_model_mse'],
                    'best_model_cosine':    models_dict['best_model_cosine'],
                    'best_model_timestep':  models_dict['best_model_timestep'],
                    'config':               exp,
                    'best_loss':            this_loss
                }, ckpt_file)

                # save its losses history
                losses_file = res_path / f'best_by_{metric}_losses.json'
                with open(losses_file, 'w') as f:
                    json.dump(losses_dict, f, indent=2)

                print(f"    • checkpoint → {ckpt_file}")
                print(f"    • losses     → {losses_file}")

        # append this experiment to the history

        result = {
            'experiment_index': i+1,
            'params': exp,
            'best_val_losses': best_val_losses,
            'losses_history': losses_dict
        }
        all_experiments.append(result)
    
        # overwrite full history after each run
        with open(summary_file, 'w') as f:
            json.dump(all_experiments, f, indent=2)
        print(f" ← saved experiment #{i+1} to {summary_file}")




    # after all experiments, dump the grid-search table
    all_results = []
    for metric, rec in best_overall.items():
        row = {
            'which_best': metric,
            'best_loss':  rec['loss'],
            **rec['config']
        }
        all_results.append(row)
    df = pd.DataFrame(all_results)
    df.to_csv(res_path / 'grid_search_bst_results.csv', index=False)

    print("\nGridSearch complete. Summary saved to:")
    print(f"   • {res_path / 'grid_search_summary.csv'}")
    for m in best_overall:
        print(f"   • best_by_{m}.pth (+ *_losses.json)")