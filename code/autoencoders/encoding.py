import os
import json
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from config import MODALITY_CONFIG
from autoencoder import FlexibleAutoencoder


def load_split_df(mod: str, split: str, cfg: dict) -> pd.DataFrame:
    """Load CSV for a given modality and split (train/val/test), keeping sample_id."""
    path = os.path.join(cfg['data_dir'], f"05_imputed/{mod}_{split}.csv")
    df = pd.read_csv(path)
    return df


def encode_split(model: FlexibleAutoencoder, df: pd.DataFrame, device: torch.device, batch_size: int = 256):
    """Run encoder on the entire DataFrame and return a numpy array of embeddings."""
    # Separate sample_ids and feature matrix
    if 'sample_id' in df.columns:
        ids = df['sample_id'].values
        features = df.drop(columns=['sample_id']).values
    else:
        ids = None
        features = df.values

    x = torch.tensor(features, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    reps = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            z = model.encode(batch)
            reps.append(z.cpu())
    reps = torch.cat(reps, dim=0).numpy()

    # Reassemble DataFrame
    rep_df = pd.DataFrame(reps)
    if ids is not None:
        rep_df.insert(0, 'sample_id', ids)
    return rep_df


def encode_modality(mod: str, cfg: dict, dim: int):
    out_dir = os.path.join(cfg['data_dir'], '05b_embeddings', str(dim))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best params
    params_path = os.path.join(cfg['output_dir'], str(dim), f"best_params_{mod}.json")
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Load one split to infer input_dim
    sample_df = load_split_df(mod, 'train', cfg)
    # Determine input_dim from number of features
    feature_cols = sample_df.columns.drop('sample_id') if 'sample_id' in sample_df.columns else sample_df.columns
    input_dim = len(feature_cols)

    # Instantiate model with detected input_dim
    model = FlexibleAutoencoder(
        input_dim=input_dim,
        bottleneck_dim=params['bottleneck_dim'],
        n_layers=params['n_layers'],
        shrink_exponent=params['shrink_exponent'],
    )

    # Load checkpoint
    ckpt_path = os.path.join(cfg['output_dir'], str(dim), f"model_{mod}.pth")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    # Encode splits
    for split in ['train', 'val', 'test']:
        print(f"Encoding {mod} {split} set...")
        df = load_split_df(mod, split, cfg)
        encoded_df = encode_split(
            model, df, device,
            batch_size=params.get('batch_size', 256)
        )
        out_csv = os.path.join(out_dir, f"{mod}_{split}.csv")
        encoded_df.to_csv(out_csv, index=False)
        print(f"  → Saved {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode splits using best-trained autoencoder(s)."
    )
    parser.add_argument(
        "--mod",
        choices=list(MODALITY_CONFIG.keys()) + ['all'],
        default='all',
        help="Which modality to encode, or 'all' for every modality."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="Encoding dimension."
    )
    args = parser.parse_args()

    targets = MODALITY_CONFIG.keys() if args.mod == 'all' else [args.mod]
    for mod in targets:
        print(f"\n>>> Processing modality: {mod}")
        cfg = MODALITY_CONFIG[mod]
        encode_modality(mod, cfg, args.dim)
