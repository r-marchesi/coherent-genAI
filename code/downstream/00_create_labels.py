import pandas as pd

import pathlib

dir = "../../datasets_TCGA/downstream_labels/"

pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

# Read the data into pandas DataFrames
split = pd.read_csv("/storage/DSH/projects/iaso/coherent_diffusion/datasets_TCGA/train_test_split_ids.csv")
clinical = pd.read_csv("/storage/DSH/projects/iaso/coherent_diffusion/datasets_TCGA/merged/reduced_clinical.csv")

# Rename 'original_id' to 'merge_id' in df1 and 'sample_id' to 'merge_id' in df2 for merging
df1_selected = split[['sample_id', 'original_id', 'split', 'cancertype']].copy()
df1_selected.rename(columns={'original_id': 'merge_id'}, inplace=True)

df2_selected = clinical[['sample_id', 'ajcc_pathologic_stage']].copy()
df2_selected.rename(columns={'sample_id': 'merge_id'}, inplace=True)

# Merge the two DataFrames on 'merge_id'
merged_df = pd.merge(df1_selected, df2_selected, on='merge_id', how='left', suffixes=('_df1', '_df2'))

# set 'sample_id' as the index
merged_df.set_index('sample_id', inplace=True)

merged_df.sort_index(inplace=True)

# Get unique splits
splits = merged_df['split'].unique()

# Process each split
for split in splits:
    split_df = merged_df[merged_df['split'] == split].copy() # Make a copy to avoid SettingWithCopyWarning

    # Create and save {split}_cancer_type.csv
    cancer_type_df = split_df[['cancertype']].copy()
    cancer_type_df.to_csv(f'../../datasets_TCGA/downstream_labels/{split}_cancer_type.csv', index=True)

    # Create and save {split}_stage.csv
    stage_df = split_df[['ajcc_pathologic_stage']].copy()
    stage_df.rename(columns={'ajcc_pathologic_stage': 'stage'}, inplace=True)
    stage_df.to_csv(f'../../datasets_TCGA/downstream_labels/{split}_stage.csv', index=True)

print("CSV files created successfully.")