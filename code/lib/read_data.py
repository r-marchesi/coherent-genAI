import itertools
import pandas as pd


import pandas as pd
import pandas as pd
from pathlib import Path

def read_data(modalities, splits, data_dir, dim=32, 
              mask_train_path='../datasets_TCGA/06_masked/32/masks_train.csv'):
    
    modalities_map = {}

    mask_train = pd.read_csv(mask_train_path, index_col=0)
   

    for modality in modalities:
        modalities_map[modality] = {}
        for split in splits:
            file_path = Path(data_dir) / f"{dim}" / f"{modality}_{split}.csv"
            modalities_map[modality][split] = pd.read_csv(file_path, index_col=0)

        modalities_map[modality]['mask_train'] = mask_train[modality].astype('float32')


    # Check for index consistency across modalities per split
    for split in splits:
        base_index = modalities_map[modalities[0]][split].index
        for modality in modalities[1:]:
            current_index = modalities_map[modality][split].index
            if current_index.equals(base_index):
                print(f"Index of {modality} {split} matches with {modalities[0]} {split}")
            else:
                print(f"Index of {modality} {split} does NOT match with {modalities[0]} {split}")

    return modalities_map