import os
import argparse
import pandas as pd


def prepare_multimodal(data_dir, output_dir, modalities):
    """
    For each split (train, val, test):
      - Load each modality_{split}.csv
      - For val/test: ensure all modalities have identical sample_id sets; else error
      - Compute sample_ids: union for train, intersection for val/test
      - Build a mask dataframe indicating presence (1) or absence (0)
      - Reindex each modality df to the common sample_ids (missing rows become NaN)
      - Write out:
          masks_{split}.csv
          <modality>_{split}_full.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    splits = ["train", "val", "test"]

    for split in splits:
        # Load all modality DataFrames
        dfs = {}
        sets = {}
        for mod in modalities:
            path = os.path.join(data_dir, f"{mod}_{split}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
            df = pd.read_csv(path)
            if 'sample_id' not in df.columns:
                raise ValueError(f"CSV {path} must contain 'sample_id' column.")
            dfs[mod] = df
            sets[mod] = set(df['sample_id'])

        # For val/test, ensure all sets are equal
        if split in ('val', 'test'):
            all_sets = list(sets.values())
            if not all_sets or any(s != all_sets[0] for s in all_sets[1:]):
                raise ValueError(
                    f"In split '{split}', not all modalities have the same samples: "
                    + ", ".join(f"{mod}({len(s)})" for mod, s in sets.items())
                )
            sample_ids = sorted(all_sets[0])
        else:
            # train: union
            sample_ids = sorted(set().union(*sets.values()))

        # Build mask DataFrame
        mask_df = pd.DataFrame({'sample_id': sample_ids})
        for mod, s in sets.items():
            mask_df[f"{mod}"] = mask_df['sample_id'].isin(s).astype(int)

        # Save masks
        mask_path = os.path.join(output_dir, f"masks_{split}.csv")
        mask_df.to_csv(mask_path, index=False)
        print(f"Wrote masks to {mask_path}")

        # Reindex and save each modality
        for mod, df in dfs.items():
            full = (
                df.set_index('sample_id')
                  .reindex(sample_ids)
                  .reset_index()
            )
            out_path = os.path.join(output_dir, f"{mod}_{split}.csv")
            full.to_csv(out_path, index=False)
            print(f"Wrote reindexed {mod} ({split}) to {out_path}")


if __name__ == '__main__':
    
    embedding_dim = 32
    data_dir = f'../datasets_TCGA/05b_embeddings/{embedding_dim}'
    output_dir = f'../datasets_TCGA/06_masked/{embedding_dim}'
    modalities = ['cna', 'rnaseq', 'rppa', 'wsi']

    prepare_multimodal(data_dir, output_dir, modalities)
