#!/usr/bin/env python
import os
import copy
import json
import argparse
import itertools

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd

from config import MODALITY_CONFIG
from autoencoder import FlexibleAutoencoder


def train_one_epoch(model, loader, device, optimizer=None) -> float:
    if optimizer:
        model.train()
    else:
        model.eval()
    total, n = 0.0, 0
    for (x,) in loader:
        x = x.to(device)
        if optimizer:
            optimizer.zero_grad()
        x_recon = model(x)
        loss = model.loss_function(x_recon, x)
        if optimizer:
            loss.backward()
            optimizer.step()
        batch_size = x.size(0)
        total += loss.item() * batch_size
        n += batch_size
    return total / n


def train_autoencoder(model, tr_loader, optimizer, device,
                      num_epochs, vl_loader, patience) -> dict:
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'best_epoch': None}
    best_val, no_imp = float('inf'), 0
    best_state = None

    for ep in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, tr_loader, device, optimizer)
        history['train_loss'].append(tr_loss)
        print(f"Epoch {ep:3d} | train_loss={tr_loss:.6f}", end='')

        vl_loss = train_one_epoch(model, vl_loader, device) if vl_loader else None
        history['val_loss'].append(vl_loss)
        if vl_loss is not None:
            print(f" | val_loss={vl_loss:.6f}", end='')
            if vl_loss < best_val:
                best_val, no_imp = vl_loss, 0
                best_state = copy.deepcopy(model.state_dict())
                history['best_epoch'] = ep
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f"  ▶ early stopping (best ep {history['best_epoch']})")
                    model.load_state_dict(best_state)
                    break
        print()

    return history


def load_data(mod, cfg):
    df_tr = pd.read_csv(os.path.join(cfg['data_dir'], f"05_imputed/{mod}_train.csv"))
    df_vl = pd.read_csv(os.path.join(cfg['data_dir'], f"05_imputed/{mod}_val.csv"))
    for df in (df_tr, df_vl):
        if 'sample_id' in df.columns:
            df.drop(columns='sample_id', inplace=True)
    return (
        torch.tensor(df_tr.values, dtype=torch.float32),
        torch.tensor(df_vl.values, dtype=torch.float32),
    )


def parameter_search(mod, cfg):
    grid    = cfg['param_grid']
    out_dir = cfg['output_dir']
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tr, x_vl = load_data(mod, cfg)
    input_dim  = x_tr.shape[1]

    best_val, best_cfg = float('inf'), None
    best_hist, best_st  = None, None

    keys, vals = zip(*grid.items())
    combos = list(itertools.product(*vals))
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n--- Trial {i}/{len(combos)}: {params}")

        tr_loader = DataLoader(TensorDataset(x_tr),
                               batch_size=params['batch_size'], shuffle=True)
        vl_loader = DataLoader(TensorDataset(x_vl),
                               batch_size=params['batch_size'], shuffle=False)

        model = FlexibleAutoencoder(input_dim, params['bottleneck_dim'], params['n_layers'], params['shrink_exponent'])
        opt   = optim.AdamW(model.parameters(),
                            lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])

        hist = train_autoencoder(model, tr_loader, opt, device,
                                 params['num_epochs'], vl_loader, params['patience'])

        # evaluate
        be = hist['best_epoch'] - 1 if hist['best_epoch'] else -1
        vl = hist['val_loss'][be]
        if vl < best_val:
            best_val, best_cfg  = vl, params
            best_hist, best_st = hist, copy.deepcopy(model.state_dict())

    # save best
    bottleneck = best_cfg['bottleneck_dim']
    out_dir = os.path.join(out_dir, str(bottleneck))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_st,  os.path.join(out_dir, f"model_{mod}.pth"))
    torch.save(best_hist, os.path.join(out_dir, f"history_{mod}.pth"))
    with open(os.path.join(out_dir, f"best_params_{mod}.json"), 'w') as f:
        json.dump(best_cfg, f, indent=2)

    print(f"\n>>> Best for {mod}: {best_cfg} (val_loss={best_val:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", required=True, choices=MODALITY_CONFIG.keys())
    args = parser.parse_args()

    cfg = MODALITY_CONFIG[args.mod]
    parameter_search(args.mod, cfg)
