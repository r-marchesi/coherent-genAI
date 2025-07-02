import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from lib.masking import generate_masks
from lib.config import params, data_dir, modalities_list, simple_exps_1, simple_exps_2
from lib.train import GridSearch, GridSearchMulti
from lib.read_data import read_data


def run_simple(exp, modalities_map, val_repeats, dim, gpu_id):
    """
    Run a simple one-to-one grid search on a specified GPU.
    """
    x_train = modalities_map[exp[0]]['train']
    x_val = modalities_map[exp[0]]['val']
    cond_train = modalities_map[exp[1]]['train']
    cond_val = modalities_map[exp[1]]['val']

    # get masks
    x_train_mask = modalities_map[exp[0]]['mask_train']
    cond_train_mask = modalities_map[exp[1]]['mask_train']

    # filter samples where both modalities present
    keep_ids = (x_train_mask == 1) & (cond_train_mask == 1)
    x_train = x_train[keep_ids]
    cond_train = cond_train[keep_ids]

    print(f"[GPU {gpu_id}] Training {exp[0]} from {exp[1]} with {keep_ids.sum()} samples")

    results_path = f"../results/{dim}/{exp[0]}_from_{exp[1]}"
    os.makedirs(results_path, exist_ok=True)

    GridSearch(
        x_train=x_train,
        cond_train=cond_train,
        x_val=x_val,
        cond_val=cond_val,
        grid_params=params,
        device=f"cuda:{gpu_id}",
        res_path=results_path,
        grid_type='random',
        val_repeats=val_repeats
    )


def run_multi(datatype, modalities_map, val_repeats, dim, gpu_id, extra_masking=False):
    """
    Run a multi-to-one grid search on a specified GPU.
    """
    x_train = modalities_map[datatype]['train']
    x_val = modalities_map[datatype]['val']
    x_train_mask = modalities_map[datatype]['mask_train']

    # prepare conditioning lists
    cond_train_list = []
    cond_val_list = []
    cond_mask_list = []

    cond_datatypes = [m for m in modalities_map.keys() if m != datatype]
    for dt in cond_datatypes:
        cond_train_list.append(modalities_map[dt]['train'])
        cond_val_list.append(modalities_map[dt]['val'])
        cond_mask_list.append(modalities_map[dt]['mask_train'])

    # filter samples where target present
    x_train = x_train[x_train_mask == 1]
    keep_ids = (x_train_mask == 1)
    cond_train_list = [ct[keep_ids] for ct in cond_train_list]

    # replace NaNs in conditioning with zeros
    cond_train_list = [ct.fillna(0) for ct in cond_train_list]

    # save cond order
    results_path = f"../results/{dim}/{datatype}_from_multi{'_masked' if extra_masking else ''}"
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, 'cond_order.json'), 'w') as f:
        import json
        json.dump(cond_datatypes, f, indent=4)

    print(f'[GPU {gpu_id}] Training multi {" with extra masking" if extra_masking else ""} -> {datatype} with {keep_ids.sum()} samples')

    GridSearchMulti(
        x_train=x_train,
        cond_train_list=cond_train_list,
        x_val=x_val,
        cond_val_list=cond_val_list,
        grid_params=params,
        res_path=results_path,
        mask_train_list=cond_mask_list,
        mask_val_list=None,
        device=f"cuda:{gpu_id}",
        grid_type='random',
        val_repeats=val_repeats,
        extra_masking=extra_masking
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parallel grid search across multiple GPUs'
    )
    parser.add_argument(
        '--mode', choices=['simple', 'simple1', 'simple2', 'multi', 'full'], default='full',
        help="Which experiments to run: simple, multi, simple1 (first half), simple2 (second half), or full (both)"
    )
    parser.add_argument(
        '--val_repeats', type=int, default=5,
        help="Number of validation repeats for grid search"
    )
    parser.add_argument(
        '--dim', type=str, default='32',
        help="Dimensionality of the input data (default: 32)"
    )
    parser.add_argument(
        '--extra_masking', action='store_true',
        help="Apply additional random masking during training"
    )
    args = parser.parse_args()

    # Auto-detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs detected. Cannot run parallel grid search.")
    gpu_ids = list(range(num_gpus))
    print(f"Detected {num_gpus} GPUs: {gpu_ids}")

    # Load data
    modalities_map = read_data(
        modalities=modalities_list,
        splits=['train','val'],
        data_dir=data_dir,
        dim=args.dim,
    )

    # Build task list
    tasks = []  # list of tuples: (kind, payload)
    simple_exps = list(itertools.permutations(modalities_map.keys(), 2))


    if args.mode in ('simple', 'simple1', 'simple2', 'full'):
        if args.mode == 'simple1':
            selected_exps = simple_exps_1
        elif args.mode == 'simple2':
            selected_exps = simple_exps_2
        else:  # simple or full
            selected_exps = simple_exps
        tasks += [('simple', exp) for exp in selected_exps]

    if args.mode in ('multi', 'full'):
        tasks += [('multi', dt) for dt in modalities_map.keys()]
    # Dispatch tasks in round-robin over GPUs
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for idx, (kind, payload) in enumerate(tasks):
            gpu = gpu_ids[idx % num_gpus]
            if kind == 'simple':
                futures.append(
                    executor.submit(run_simple, payload, modalities_map, args.val_repeats, args.dim, gpu)
                )
            else:
                futures.append(
                    executor.submit(run_multi, payload, modalities_map, args.val_repeats, args.dim, gpu, args.extra_masking)
                )

        # Wait for all to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("Task error:", e)


