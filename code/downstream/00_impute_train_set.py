import pathlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer

import copy
import sys
sys.path.append("../")

from lib.read_data import read_data
from lib.get_models import get_diffusion_model
from lib.diffusion_models import GaussianDiffusion
from lib.sampling import coherent_sample, sample
from lib.config import modalities_list

# --- Configuration ---
dim = '32'
data_dir = pathlib.Path('../../datasets_TCGA/07_normalized/')
mask_path = pathlib.Path(f'../../datasets_TCGA/06_masked/{dim}/masks_train.csv')
results_path = pathlib.Path('../../results/')
out_path = pathlib.Path(f'./data_task_02/')
out_path.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading data...")
modalities_map = read_data(
    modalities=modalities_list,
    splits=['train'],
    data_dir=str(data_dir),
    dim=dim,
    mask_train_path=str(mask_path)
)

# --- 1) FILTER OUT SAMPLES WITH ONLY ONE MODALITY --------------------------------
# find for each idx how many modalities are present
idxs = modalities_map[next(iter(modalities_map))]['train'].index
present_counts = {
    idx: sum(modalities_map[m]['mask_train'].loc[idx] for m in modalities_map)
    for idx in idxs
}
to_drop = [idx for idx, cnt in present_counts.items() if cnt == 1]
print(f"Dropping {len(to_drop)} samples with only one modality present.")

# drop from each modality’s train and mask_train
for m in modalities_map:
    modalities_map[m]['train'].drop(index=to_drop, inplace=True)
    modalities_map[m]['mask_train'].drop(index=to_drop, inplace=True)

# --- 2) SAVE FILTERED “ORIGINAL” DATA --------------------------------------------
# prefix columns, concat side-by-side
orig_dfs = []
for m, data in modalities_map.items():
    df = data['train'].copy()
    df.columns = [f"{m}_{c}" for c in df.columns]
    orig_dfs.append(df)
orig_combined = pd.concat(orig_dfs, axis=1).sort_index()
orig_out = out_path / f"real_data_train.csv"
orig_combined.to_csv(orig_out)
print(f"Saved filtered original data to {orig_out}")

# --- 2.1) FILL MISSING VALUES IN ORIGINAL DATA -----------------------------------
knn = KNNImputer(n_neighbors=5, weights='uniform')
knn_arr = knn.fit_transform(orig_combined)
knn_df = pd.DataFrame(knn_arr, index=orig_combined.index, columns=orig_combined.columns)
knn_out = out_path / f"imputed_knn.csv"
knn_df.to_csv(knn_out)

print(f"Saved KNN imputed data to {knn_out}")

# --- 3) DEFINE GENERATION ROUTINE -----------------------------------------------
def load_models(mode):
    models_dict = {}
    diffusion = GaussianDiffusion(num_timesteps=1000).to(device)
    for target in modalities_map:
        conds = [m for m in modalities_map if m != target]
        if mode == 'multi':
            ckpt = torch.load(results_path / dim / f"{target}_from_multi/train/best_by_mse.pth",
                              map_location='cpu')
            cfg, state = ckpt['config'], ckpt['best_model_mse']
            model = get_diffusion_model(cfg['architecture'], diffusion, SimpleNamespace(**cfg),
                                        x_dim=modalities_map[target]['train'].shape[1],
                                        cond_dims=[modalities_map[c]['train'].shape[1] for c in conds]
            ).to(device)
            model.load_state_dict(state); model.eval()
            models_dict[target] = {'models':[model], 'conds':conds, 'diffusion':diffusion}
        else:  # coherent
            mdls = []
            for c in conds:
                ckpt = torch.load(results_path / dim / f"{target}_from_{c}/train/best_by_mse.pth",
                                  map_location='cpu')
                cfg, state = ckpt['config'], ckpt['best_model_mse']
                mdl = get_diffusion_model(cfg['architecture'], diffusion, SimpleNamespace(**cfg),
                                         x_dim=modalities_map[target]['train'].shape[1],
                                         cond_dims=modalities_map[c]['train'].shape[1]
                ).to(device)
                mdl.load_state_dict(state); mdl.eval()
                mdls.append(mdl)
            models_dict[target] = {'models':mdls, 'conds':conds, 'diffusion':diffusion}
    return models_dict

def generate_batch(models_dict, mode, target, cond_batches, present):
    entry = models_dict[target]
    if mode == 'multi':
        # build cond list + mask tensor
        cond_ts, masks = [], []
        for cm in entry['conds']:
            if cm in present:
                arr = cond_batches[present.index(cm)]
                masks.append(np.ones(len(arr)))
            else:
                arr = np.zeros((len(cond_batches[0]), modalities_map[cm]['train'].shape[1]), dtype=np.float32)
                masks.append(np.zeros(len(cond_batches[0])))
            cond_ts.append(torch.tensor(arr, dtype=torch.float32, device=device))
        mask_tensor = torch.tensor(np.stack(masks), dtype=torch.float32, device=device)
        out = sample(entry['models'][0], entry['diffusion'], cond_ts,
                     num_features=modalities_map[target]['train'].shape[1],
                     mask=mask_tensor, device=device)
    else:
        cond_ts, mdls = [], []
        for mdl, cm in zip(entry['models'], entry['conds']):
            if cm in present:
                cond_ts.append(torch.tensor(cond_batches[present.index(cm)], dtype=torch.float32, device=device))
                mdls.append(mdl)
        out = coherent_sample(mdls, entry['diffusion'],
                              num_samples=cond_ts[0].shape[0],
                              num_features=modalities_map[target]['train'].shape[1],
                              conds=cond_ts, device=device)
    return out.cpu().numpy()

def impute_all(mode):
    print(f"\n=== Running imputation in {mode} mode ===")
    models_dict = load_models(mode)

    # build groups (present, missing) → list of idx
    groups = {}
    for idx in modalities_map[next(iter(modalities_map))]['train'].index:
        mask = {m: modalities_map[m]['mask_train'].loc[idx] for m in modalities_map}
        missing = tuple(sorted([m for m,v in mask.items() if v==0]))
        present = tuple(sorted([m for m in modalities_map if m not in missing]))
        if missing and len(present)>1:
            groups.setdefault((present, missing), []).append(idx)

    imputed = {}
    # one‑missing
    for (pres, miss), idxs in groups.items():
        if len(miss)==1:
            tgt = miss[0]
            conds = [np.vstack([modalities_map[c]['train'].loc[i].values for i in idxs]) for c in pres]
            gen = generate_batch(models_dict, mode, tgt, conds, pres)
            for i, idx in enumerate(idxs):
                rec = {c: modalities_map[c]['train'].loc[idx].values for c in pres}
                rec[tgt] = gen[i]
                imputed[idx] = rec
    # two‑missing
    for (pres, miss), idxs in groups.items():
        if len(miss)==2:
            a,b = miss
            stacks = {c: np.vstack([modalities_map[c]['train'].loc[i].values for i in idxs]) for c in pres}
            gen_a = generate_batch(models_dict, mode, a, [stacks[c] for c in pres], pres)
            gen_b = generate_batch(models_dict, mode, b, [stacks[c] for c in pres], pres)
            for i, idx in enumerate(idxs):
                rec = {c: modalities_map[c]['train'].loc[idx].values for c in pres}
                rec[a], rec[b] = gen_a[i], gen_b[i]
                imputed[idx] = rec

    # append imputed back into modalities_map (in‑place)
    for m in modalities_map:
        df = modalities_map[m]['train']
        # for each sample we generated, write its imputed vector into the same row
        for idx, rec in imputed.items():
            df.loc[idx] = rec[m]
        # store it back
        modalities_map[m]['train'] = df

    print(f" Imputation ({mode}) generated {len(imputed)} records.")

    # SAVE COMBINED
    dfs = []
    for m, data in modalities_map.items():
        df = data['train'].copy()
        df.columns = [f"{m}_{c}" for c in df.columns]
        dfs.append(df)
    combined = pd.concat(dfs, axis=1).sort_index()
    out = out_path / f"imputed_{mode}.csv"
    combined.to_csv(out)
    print(f" Saved {mode} output to {out}")

# --- 4) RUN BOTH MODES -----------------------------------------------------------
for m in ['multi', 'coherent']:
    # work on a fresh copy of modalities_map each time:
    
    modalities_map_backup = copy.deepcopy(modalities_map)
    impute_all(m)
    modalities_map = copy.deepcopy(modalities_map_backup)


# --- SAVE TEST SET IN "wide" FORMAT ------------------------------------------
print("Loading & saving TEST set...")
modalities_test = read_data(
    modalities=modalities_list,
    splits=['test'],
    data_dir=str(data_dir),
    dim=dim,
    mask_train_path=str(mask_path)
)

# collect each modality’s test DataFrame, prefix its columns
test_dfs = []
for m, data in modalities_test.items():
    df = data['test'].copy()
    df.columns = [f"{m}_{c}" for c in df.columns]
    test_dfs.append(df)

# concatenate side‑by‑side and write
test_combined = pd.concat(test_dfs, axis=1).sort_index()
test_out = out_path / "test_data.csv"
test_combined.to_csv(test_out)
print(f"Saved test data to {test_out}")