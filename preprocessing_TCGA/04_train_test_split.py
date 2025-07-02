import pandas as pd
from tabulate import tabulate
from math import floor
 
# Load data
df = pd.read_csv("../datasets_TCGA/IDs_file.csv")
 
# Flag complete samples
complete_mask = (
    (df["genomics"] == 1) &
    (df["transcriptomics"] == 1) &
    (df["proteomics"] == 1) &
    (df["clinical"] == 1) &
    (df["wsi"] == 1) 
)
df_complete   = df[complete_mask].copy()
df_incomplete = df[~complete_mask].copy()
 
# Compute per-class targets (70% train, 10% val, 20% test)
class_counts = df["cancertype"].value_counts().to_dict()
n_val_per_class = {}
n_test_per_class = {}
n_train_per_class = {}
 
for cls, total in class_counts.items():
    val  = floor(total * 0.05)
    test = floor(total * 0.15)
    train = total - val - test  # make sure total counts match
    n_val_per_class[cls] = val
    n_test_per_class[cls] = test
    n_train_per_class[cls] = train


# Pre‐flight check
insufficient = []
for cls, total in class_counts.items():
    need = n_val_per_class[cls] + n_test_per_class[cls]
    have = int((df_complete["cancertype"] == cls).sum())
    if have < need:
        insufficient.append((cls, have, need, need - have))

if insufficient:
    print("The following cancertypes don't have enough *complete* samples for the desired val+test split:")
    print("  cancertype   have   need   shortage")
    for cls, have, need, shortage in insufficient:
        print(f"  {cls:<12} {have:>4}  {need:>4}    {shortage:>4}")
else:
    print("All classes have enough complete samples for the val+test split.")
 
val_list = []
test_list = []
train_list = []
 
used_indices = set()
 
# Sampling per class
for cls, total_cnt in class_counts.items():
    complete_pool = df_complete[df_complete["cancertype"] == cls]
    incomplete_pool = df_incomplete[df_incomplete["cancertype"] == cls]
 
    want_val  = n_val_per_class[cls]
    want_test = n_test_per_class[cls]
    want_train = n_train_per_class[cls]
 
    total_want_valtest = want_val + want_test
    avail_complete = len(complete_pool)
 
    # --- Validation/Test Split ---
    if avail_complete >= total_want_valtest:
        val_i = complete_pool.sample(n=want_val, random_state=42)
        pool2 = complete_pool.drop(val_i.index)
        test_i = pool2.sample(n=want_test, random_state=42)
    else:
        # Not enough complete: use what we can, split proportionally
        val_n = floor(avail_complete * (want_val / total_want_valtest))
        test_n = avail_complete - val_n
        val_i = complete_pool.sample(n=val_n, random_state=42)
        test_i = complete_pool.drop(val_i.index)
 
    val_list.append(val_i)
    test_list.append(test_i)
    used_indices.update(val_i.index)
    used_indices.update(test_i.index)
 
    # --- Training Split ---
    # Use remaining complete + incomplete to fill training
    remaining_pool = df[(df["cancertype"] == cls) & (~df.index.isin(used_indices))]
    train_i = remaining_pool.sample(n=want_train, random_state=42)
    train_list.append(train_i)
    used_indices.update(train_i.index)
 
# Combine splits
val_df  = pd.concat(val_list, ignore_index=True)
test_df = pd.concat(test_list, ignore_index=True)
train_df = pd.concat(train_list, ignore_index=True)
 
# Tag splits
train_df["split"] = "train"
val_df["split"]   = "val"
test_df["split"]  = "test"
 
# Combine all
df_split = pd.concat([train_df, val_df, test_df], ignore_index=True)
df_split.to_csv("../datasets_TCGA/train_test_split_ids.csv", index=False)
 
# Report
print(df_split["split"].value_counts())
print("\nPer-class cancer-type proportions in each split (in %):")

# compute normalized proportions and convert to percent
prop = df_split.groupby("split")["cancertype"] \
               .value_counts(normalize=True) \
               .unstack() * 100

# print with two decimal places
print(tabulate(
    prop,
    headers="keys",
    tablefmt="psql",
    floatfmt=".2f"
))