import itertools
import json
import pathlib
from types import SimpleNamespace

from lib.test import coherent_test_cos_rejection, test_model
from lib.config import modalities_list, data_dir
from lib.get_models import get_diffusion_model
from lib.diffusion_models import GaussianDiffusion

from lib.read_data import read_data
import argparse

import numpy as np
import pandas as pd
import torch

# settings
device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")




# Parse arguments
parser = argparse.ArgumentParser(description="Choose testing parameters")
parser.add_argument('--mode', type=str, choices=['simple', 'multi', 'coherent', 'full'],  default='full', 
                    help="Specify the training mode: 'simple', 'multi', 'coherent', or 'full'")
parser.add_argument('--folder', type=str, default='results')
parser.add_argument('--metric', type=str, choices=['mse', 'cosine', 'timestep'], default='mse',
                    help="Specify the metric to determine the best trained model: 'mse', 'cosine', 'timestep'")
parser.add_argument('--dim', type=str,  default='32', help="Input dimension")
parser.add_argument('--test_repeats', type=int, default=10, help="Number of repetitions of the test set")
parser.add_argument('--mask', action='store_true', help="Use the multi conditioning models traind with extra masks")


args = parser.parse_args()



# input data
modalities_map = read_data(
    modalities=modalities_list,
    splits=['test'],
    data_dir=data_dir,
    dim=args.dim,
)

# list of all combinations between the modalities in the input data
exp_list = list(itertools.permutations(modalities_map.keys(),2)) 




''' Single Sampling '''
if args.mode in ['simple', 'full']:
    for exp in exp_list:
        print(f'Test the model to generate  {exp[0]} from {exp[1]}')
        x_test = modalities_map[exp[0]]['test'] 
        cond_test = modalities_map[exp[1]]['test'] 
        
        path = pathlib.Path(f'../{args.folder}/{args.dim}/{exp[0]}_from_{exp[1]}') 

        # read checkpoint dictionary
        ckpt_path = path / f'train/best_by_{args.metric}.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # load from checkpoint
        raw_cfg = ckpt["config"]
        config = SimpleNamespace(**raw_cfg)
        state_dict = ckpt[f"best_model_{args.metric}"]        

        # load model 
        x_dim    = x_test.shape[1]
        cond_dim = cond_test.shape[1]
        diffusion = GaussianDiffusion(num_timesteps=1000).to(device)
        model     = get_diffusion_model(
            config.architecture,
            diffusion,
            config,
            x_dim=x_dim,
            cond_dims=cond_dim
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        test_metrics, generated_samples = test_model(x_test, cond_test, model, diffusion,
                                                    test_iterations=args.test_repeats, device=device)

        # store generated samples
        test_path = path / 'test'
        test_path.mkdir(parents=True, exist_ok=True)

        generated_samples.to_csv(test_path / f'generated_samples_best_{args.metric}.csv', index=False)

        #store test metrics
        with open(test_path / f'test_metrics_best_{args.metric}.json', 'w') as f:
            json.dump(test_metrics, f, indent=4)

    


''' Coherent Sampling ''' 
if args.mode in ['coherent', 'full']:
    for datatype in modalities_map.keys():
        print(f'Test the models to generate {datatype} with Coherent')

        x_test = modalities_map[datatype]['test']

        cond_datatypes = list(modalities_map.keys())
        cond_datatypes.remove(datatype)

        test_path = pathlib.Path(f'../{args.folder}/{args.dim}/{datatype}_from_coherent/test')
        test_path.mkdir(parents=True, exist_ok=True)
 
        for r in range(2, len(cond_datatypes) + 1): #For each subset‐size r = 2…|cond_datatypes|
            for cond_combo in itertools.combinations(cond_datatypes, r):
                combo_name = "_".join(cond_combo)
                print(f'Generating {datatype} from {combo_name}')

                cond_test_list = [modalities_map[c]['test'] for c in cond_combo]
                
                models_list = []    

                weights_list = [] 

                diffusion = GaussianDiffusion(num_timesteps=1000).to(device)

                for c in cond_combo:
                    # Path to “datatype_from_c/train/best_by_{metric}.pth”
                    single_ckpt_dir = pathlib.Path(f'../{args.folder}/{args.dim}/{datatype}_from_{c}/train')
                    ckpt_path = single_ckpt_dir / f'best_by_{args.metric}.pth'
                    if not ckpt_path.exists():
                        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

                    # Load the checkpoint dict
                    ckpt       = torch.load(ckpt_path, map_location='cpu')
                    raw_cfg    = ckpt['config']
                    config_c   = SimpleNamespace(**raw_cfg)
                    state_dict = ckpt[f'best_model_{args.metric}']
                    
                    best_loss = ckpt['best_loss']
                    weights_list.append(best_loss)


                    x_dim = x_test.shape[1]
                    cond_dim = modalities_map[c]['test'].shape[1]
                    
                    model_c = get_diffusion_model(
                        config_c.architecture,
                        diffusion,
                        config_c,
                        x_dim=x_dim,
                        cond_dims=cond_dim
                    )
                    model_c.load_state_dict(state_dict)
                    model_c.to(device).eval()

                    models_list.append(model_c)


                # Run coherent sampling
                test_metrics,  generated_samples, _ = coherent_test_cos_rejection(
                    x_test, cond_test_list, models_list, diffusion,
                    test_iterations=args.test_repeats, max_retries=10, device=device, weights_list=weights_list
                )

                # Save generated samples
                generated_samples.to_csv(test_path / f'generated_samples_from_{combo_name}_best_{args.metric}.csv', index=False)

                #store test metrics
                with open(test_path / f'test_metrics_from_{combo_name}_best_{args.metric}.json', 'w') as f:
                    json.dump(test_metrics, f, indent=4)




 
''' Multi Sampling '''
if args.mode in ['multi', 'full']:
    for datatype in modalities_map.keys():
        print(f'Test the model to generate {datatype} with Multi')

        x_test = modalities_map[datatype]['test']

        cond_datatypes = list(modalities_map.keys())
        cond_datatypes.remove(datatype)

        test_path = pathlib.Path(f"../{args.folder}/{args.dim}/{datatype}_from_multi{'_masked' if args.mask else ''}/test")
        test_path.mkdir(parents=True, exist_ok=True)
 
        # paths handling
        base_dir = pathlib.Path(f"../{args.folder}/{args.dim}/{datatype}_from_multi{'_masked' if args.mask else ''}") 
        ckpt_path = base_dir / 'train' / f'best_by_{args.metric}.pth'

        cond_order_path = base_dir / 'cond_order.json'
        with open(cond_order_path, 'r') as f:
            cond_order = json.load(f)
        assert cond_order == cond_datatypes, (
            f'Conditioning order mismatch: {cond_order} != {cond_datatypes}'
        )

        # Load the checkpoint dict
        ckpt = torch.load(ckpt_path, map_location='cpu')
        raw_cfg = ckpt['config']
        config = SimpleNamespace(**raw_cfg)
        state_dict = ckpt[f'best_model_{args.metric}']

        x_dim = x_test.shape[1]
        cond_dim_list = [modalities_map[c]['test'].shape[1] for c in cond_order]

        diffusion = GaussianDiffusion(num_timesteps=1000).to(device)
        
        # Load the model
        model = get_diffusion_model(
            config.architecture,
            diffusion,
            config,
            x_dim=x_dim,
            cond_dims=cond_dim_list   # list of conditioning dims
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()


        for r in range(1, len(cond_datatypes) + 1):
            for cond_combo in itertools.combinations(cond_datatypes, r):

                combo_name = "_".join(cond_combo)
                print(f'Generating {datatype} from {combo_name}')

                # Build cond_test_list with zero replacement
                cond_test_list = []
                for cond_name in cond_datatypes:
                    if cond_name in cond_combo:
                        cond_test_list.append(modalities_map[cond_name]['test'])
                    else:
                        shape = modalities_map[cond_name]['test'].shape
                        cond_test_list.append(pd.DataFrame(np.zeros(shape), columns=modalities_map[cond_name]['test'].columns))


                # Create masks for each conditioning: 1 if the condition is in the combo, 0 otherwise
                masks = []
                num_samples = x_test.shape[0]
                for cond_name in cond_datatypes:
                    if cond_name in cond_combo:
                        masks.append(np.ones(num_samples))  
                    else:
                        masks.append(np.zeros(num_samples))

           
                test_metrics, generated_df = test_model(
                    x_test, cond_test_list, model, diffusion,
                    test_iterations=args.test_repeats, device=device, masks=masks
                )

                out_csv = test_path / f'generated_samples_from_{combo_name}_best_{args.metric}.csv'
                generated_df.to_csv(out_csv, index=False)

                out_json = test_path / f'test_metrics_from_{combo_name}_best_{args.metric}.json'
                with open(out_json, 'w') as f:
                    json.dump(test_metrics, f, indent=4)



print("Testing completed successfully.")