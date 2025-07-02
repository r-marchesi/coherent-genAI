import os
import pandas as pd
from sklearn import preprocessing


embedding_dim = 32

input_dir = f'../datasets_TCGA/06_masked/{embedding_dim}'
output_dir = f'../datasets_TCGA/07_normalized/{embedding_dim}'

modalities = ['cna', 'rnaseq', 'rppa', 'wsi']


for modality in modalities:
    train_path = os.path.join(input_dir, f"{modality}_train.csv")
    val_path = os.path.join(input_dir, f"{modality}_val.csv")
    test_path = os.path.join(input_dir, f"{modality}_test.csv")

    # Read the CSV files into DataFrames
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Drop 'sample_id' to get features
    train_features = train_df.drop(columns=['sample_id'])
    val_features = val_df.drop(columns=['sample_id'])
    test_features = test_df.drop(columns=['sample_id'])

    # Fit scaler on training data
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_features)

    # Transform train, val, and test features
    train_norm = scaler.transform(train_features)
    val_norm = scaler.transform(val_features)
    test_norm = scaler.transform(test_features)

    # store fitted scaler in output directory
    scaler_output_path = os.path.join(output_dir, f"{modality}_scaler.pkl")
    os.makedirs(output_dir, exist_ok=True)
    pd.to_pickle(scaler, scaler_output_path)
    
    # Create new DataFrames with normalized features and add 'sample_id' back
    train_norm_df = pd.DataFrame(train_norm, columns=train_features.columns)
    train_norm_df.insert(0, 'sample_id', train_df['sample_id'])

    val_norm_df = pd.DataFrame(val_norm, columns=val_features.columns)
    val_norm_df.insert(0, 'sample_id', val_df['sample_id'])

    test_norm_df = pd.DataFrame(test_norm, columns=test_features.columns)
    test_norm_df.insert(0, 'sample_id', test_df['sample_id'])



    # save normalized DataFrames to CSV files in 08_normalized directory
    os.makedirs(output_dir, exist_ok=True)
    train_norm_df.to_csv(os.path.join(output_dir, f"{modality}_train.csv"), index=False)
    val_norm_df.to_csv(os.path.join(output_dir, f"{modality}_val.csv"), index=False)
    test_norm_df.to_csv(os.path.join(output_dir, f"{modality}_test.csv"), index=False)
    print(f"Normalized {modality} data saved to {output_dir}")



