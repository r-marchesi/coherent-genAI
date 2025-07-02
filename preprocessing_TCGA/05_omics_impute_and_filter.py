import os
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


def get_imputer(strategy):
    if strategy == "knn":
        return KNNImputer(n_neighbors=5)
    elif strategy == "simple":
        return SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Unknown imputer strategy: {strategy}")


def preprocess_modality(mod, config):
    print(f"Processing modality: {mod}")
    file_name = config["modality_file_map"][mod]
    data_path = os.path.join(config["data_dir"], "merged", file_name)
    split_path = os.path.join(config["data_dir"], "train_test_split_ids.csv")

    # load data
    data_df = pd.read_csv(data_path)
    split_df = (
        pd.read_csv(split_path)[["sample_id", "original_id", "split"]]
          # rename so that split_df.sample_id->new_id, split_df.original_id->sample_id
          .rename(columns={
              "sample_id": "new_id",
              "original_id": "sample_id"
          })
    )

    # merge on the renamed sample_id (which now matches data_df.sample_id)
    merged_df = pd.merge(split_df, data_df, on="sample_id")

    # split into train/val/test
    train_df = merged_df[merged_df["split"] == "train"]
    val_df   = merged_df[merged_df["split"] == "val"]
    test_df  = merged_df[merged_df["split"] == "test"]

    # stash the new IDs for reinsertion later
    train_ids = train_df["new_id"].values
    val_ids   = val_df["new_id"].values
    test_ids  = test_df["new_id"].values

    # drop identifier columns before imputation
    drop_cols = ["sample_id", "new_id", "split"]
    train_df = train_df.drop(columns=drop_cols)
    val_df   = val_df.drop(columns=drop_cols)
    test_df  = test_df.drop(columns=drop_cols)

    print(f"Original number of features: {train_df.shape[1]}")

    # Drop features with >10% missing in train only
    missing_threshold = 0.1
    keep_mask = train_df.isnull().mean() < missing_threshold
    train_df = train_df.loc[:, keep_mask]
    val_df   = val_df.loc[:, keep_mask]
    test_df  = test_df.loc[:, keep_mask]

    print(f"Features after dropping >{missing_threshold*100}% missing: {train_df.shape[1]}")

    # Fit the chosen imputer on train
    imputer = get_imputer(config["imputer"][mod])
    imputer.fit(train_df)

    # Transform all splits
    train_imputed = imputer.transform(train_df)
    val_imputed   = imputer.transform(val_df)
    test_imputed  = imputer.transform(test_df)

    # Rebuild DataFrames
    train_out = pd.DataFrame(train_imputed, columns=train_df.columns)
    val_out   = pd.DataFrame(val_imputed,   columns=val_df.columns)
    test_out  = pd.DataFrame(test_imputed,  columns=test_df.columns)

    # Re-insert the new sample IDs as 'sample_id'
    train_out.insert(0, "sample_id", train_ids)
    val_out.insert(0,   "sample_id", val_ids)
    test_out.insert(0,  "sample_id", test_ids)

    # Write out
    output_dir = os.path.join(config["data_dir"], "05_imputed")
    os.makedirs(output_dir, exist_ok=True)

    train_out.to_csv(os.path.join(output_dir, f"{mod}_train.csv"), index=False)
    val_out.to_csv(os.path.join(output_dir, f"{mod}_val.csv"),   index=False)
    test_out.to_csv(os.path.join(output_dir, f"{mod}_test.csv"), index=False)

    print(f"Saved imputed data for {mod} to {output_dir}/\n")


if __name__ == "__main__":
    config = {
        "data_dir": "../datasets_TCGA",
        "modality_file_map": {
            "rnaseq": "merged_rnaseq_QC.csv",
            "rppa":   "merged_rppa.csv",
            "cna":    "merged_cna.csv",
        },
        "imputer": {
            "rnaseq": "knn",
            "rppa":   "knn",
            "cna":    "simple",
        },
    }

    for mod in config["modality_file_map"]:
        preprocess_modality(mod, config)
