import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Parameters
n_dim = 32 
wsi_csv = "../datasets_TCGA/wsi_embeddings/wsi_embeddings_filtered.csv"
split_csv  = "../datasets_TCGA/train_test_split_ids.csv"
output_dir = f"../datasets_TCGA/05b_embeddings/{n_dim}"

# ── Prepare output directory ─────────────────────────────────────────────────
os.makedirs(output_dir, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
wsi_df   = pd.read_csv(wsi_csv)
split_df = pd.read_csv(split_csv)

# ── Merge and split ──────────────────────────────────────────────────────────
# Drop slide_id, keep sample_id + embeddings
wsi_df = wsi_df.drop(columns=["slide_id"])

# Only keep sample_id + split
merged = wsi_df.merge(split_df[["sample_id", "split"]], on="sample_id", how="inner")

train_df = merged[merged["split"] == "train"].drop(columns=["split"])
val_df   = merged[merged["split"] == "val"].drop(columns=["split"])
test_df  = merged[merged["split"] == "test"].drop(columns=["split"])

# Features = all columns except sample_id
feat_cols = [c for c in train_df.columns if c != "sample_id"]

X_train = train_df[feat_cols].values
X_val   = val_df[feat_cols].values
X_test  = test_df[feat_cols].values

# ── Scale ────────────────────────────────────────────────────────────────────
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ── PCA ──────────────────────────────────────────────────────────────────────
pca = PCA(n_components=n_dim, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# Print explained variance
cumvar = pca.explained_variance_ratio_.cumsum()
print(f"Cumulative explained variance by {n_dim} components: {cumvar[-1]:.3f}")

# ── Save ─────────────────────────────────────────────────────────────────────
def save_split(X_pca, df_split, name):
    out_df = pd.DataFrame(
        X_pca,
        columns=[f"pc{i+1}" for i in range(n_dim)],
        index=df_split["sample_id"]
    ).reset_index().rename(columns={"index": "sample_id"})
    out_path = os.path.join(output_dir, f"wsi_{name}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  • wrote {out_path}")

save_split(X_train_pca, train_df, "train")
save_split(X_val_pca,   val_df,   "val")
save_split(X_test_pca,  test_df,  "test")
