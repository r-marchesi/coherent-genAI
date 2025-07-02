import pandas as pd

# --- 1) Load & clean summary_omics ---
omics = pd.read_csv(
    "../datasets_TCGA/summary_omics_>1_TCGA.tsv",
    sep="\t",
    dtype=str
)

# Preserve full original ID
omics['original_id'] = omics['sample_id']

# Remember original omics columns order (excluding sample_id, we'll reinsert trimmed + original)
omics_cols = [c for c in omics.columns if c not in ('sample_id',)]

# Trim last character, then strip it
omics['last_letter'] = omics['sample_id'].str[-1]
omics['sample_id']   = omics['sample_id'].str[:-1]

# Sort & dedupe
omics_clean = (
    omics
    .sort_values(by=['sample_id', 'last_letter'])
    .drop_duplicates(subset='sample_id', keep='first')
    .drop(columns=['last_letter'])
)

# --- 2) Load & clean wsi_embeddings ---
wsi = pd.read_csv(
    "../datasets_TCGA/wsi_embeddings/wsi_embeddings.csv",
    dtype=str
)

wsi['last_letter'] = wsi['sample_id'].str[-1]
wsi['sample_id']   = wsi['sample_id'].str[:-1]

wsi_clean = (
    wsi
    .sort_values(by=['sample_id', 'last_letter'])
    .drop_duplicates(subset='sample_id', keep='first')
    .drop(columns=['last_letter'])
)

# --- 3) Merge & flag ---
merged = omics_clean.merge(
    wsi_clean[['sample_id']],
    on='sample_id',
    how='left',
    indicator=True
)

merged['wsi'] = (merged['_merge'] == 'both').astype(int)
merged = merged.drop(columns=['_merge'])

# --- 4) Reorder & save enriched IDs file ---
# New order: trimmed sample_id, original_id, then the rest of omics columns (cancertype, genomics, ...), then wsi
final_cols = ['sample_id', 'original_id'] + [c for c in omics_cols if c != 'original_id'] + ['wsi']

merged[final_cols].to_csv(
    "../datasets_TCGA/IDs_file.csv",
    index=False
)

# --- 5) Save filtered WSI embeddings ---
kept_ids = merged.loc[merged['wsi'] == 1, 'sample_id']
wsi_filtered = wsi_clean[wsi_clean['sample_id'].isin(kept_ids)]

wsi_filtered.to_csv(
    "../datasets_TCGA/wsi_embeddings/wsi_embeddings_filtered.csv",
    index=False
)
