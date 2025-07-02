import h5py
import pandas as pd
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────
# Directory containing all .h5 embedding files 
dest_root = Path("../datasets_TCGA/wsi_embeddings/individual_embeddings")
# Output CSV path
output_csv = Path("../datasets_TCGA/wsi_embeddings/wsi_embeddings.csv")
# ────────────────────────────────────────────────────────────────────────────────


def extract_sample_id(stem: str) -> str:
    """
    Extract the sample_id from a filename stem by taking the first four
    hyphen-delimited segments.
    e.g., 'TCGA-CV-A6JZ-01Z-00-DX1.XXX' -> 'TCGA-CV-A6JZ-01Z'
    """
    parts = stem.split('-')
    # Ensure there are at least 4 parts
    if len(parts) >= 4:
        return '-'.join(parts[:4])
    return stem  # fallback to full stem if unexpected format


def aggregate_embeddings(dest_root: Path, output_csv: Path):
    """
    Reads each .h5 file in dest_root, extracts the 'features' dataset,
    builds a DataFrame indexed by filename stem, adds 'sample_id',
    and saves the table to output_csv.
    """
    records = {}
    for h5_file in dest_root.glob("*.h5"):
        stem = h5_file.stem  # filename without extension
        sample_id = extract_sample_id(stem)

        with h5py.File(h5_file, 'r') as f:
            data = f.get('features')
            if data is None:
                print(f"Warning: 'features' dataset not found in {h5_file.name}. Skipping.")
                continue
            arr = data[:]
            if arr.ndim != 1:
                print(f"Warning: unexpected features shape {arr.shape} in {h5_file.name}. Skipping.")
                continue
            # Build record for this slide
            record = {'sample_id': sample_id}
            record.update({f"{i}": val for i, val in enumerate(arr)})
            records[stem] = record

    if not records:
        print("No valid feature vectors found. Exiting.")
        return

    # Build DataFrame: index is full stem, columns include 'sample_id' and feat_*
    df = pd.DataFrame.from_dict(records, orient='index')
    df.index.name = 'slide_id'

    # Save to CSV
    df.to_csv(output_csv)
    print(f"Saved embeddings table with shape {df.shape} to {output_csv}")

if __name__ == '__main__':
    aggregate_embeddings(dest_root, output_csv)
