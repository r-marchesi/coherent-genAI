import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class MultiConditionDataset(Dataset):
    def __init__(self,
                 x_df,
                 cond_df_list=None,
                 mask_list=None,
                 extra_masking=False):
        """
        x_df           : pd.DataFrame of shape [N, x_dim]
        cond_df_list   : list of DataFrames, each [N, cond_dim_i]
        mask_list      : list of length‐N iterables of 0/1, one per cond, or None
        extra_masking  : if True, apply additional random masking each epoch
        """
        self.tensor_x = torch.tensor(x_df.values).float()
        self.extra_masking = extra_masking

        if extra_masking:
            print("Creating dataset with random extra masking at each epoch.")

        # Normalize cond_df_list into a list (possibly empty)
        self.conds = []
        if cond_df_list is not None:
            for df in cond_df_list:
                self.conds.append(torch.tensor(df.values).float())

        self.has_masks = mask_list is not None

        if self.has_masks:
            assert len(mask_list) == len(cond_df_list), \
                   "Need one mask-array per cond-DataFrame"
            # Each mask is length-N
            self.masks = [
                torch.tensor(np.asarray(m)).float()
                for m in mask_list
            ]

        self.length = len(self.tensor_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.tensor_x[idx]

        # Gather raw cond vectors (no masking applied yet)
        conds = [c[idx].clone() for c in self.conds]

        if not self.has_masks:
            # No masks at all: simply return x and conds (unwrapped if single)
            if len(conds) == 1:
                return x, conds[0]
            return x, conds

        #  apply the *original* masks
        raw_masks = []
        for i, (c_tensor, m_tensor) in enumerate(zip(self.conds, self.masks)):
            mi = m_tensor[idx].item()  # 0.0 or 1.0 (float)
            raw_masks.append(int(mi))
            if mi == 0:
                # masked: zero out this condition
                conds[i] = torch.zeros_like(conds[i])

        # If extra_masking is disabled, just return the already‐masked result
        if not self.extra_masking:

            return x, conds, raw_masks
        


        ### EXTRA_MASKING

        # Decide how many total conditions to mask this epoch
        num_conds = len(conds)
        # Ensure at least one condition remains unmasked
        num_to_mask = random.randint(0, num_conds - 1)

        # Count how many are already masked
        existing_masks = raw_masks.count(0)

        # Compute how many additional masks are needed
        add_masks = max(0, num_to_mask - existing_masks)

        if add_masks > 0:
            # Identify currently unmasked conditions
            unmasked_indices = [i for i in range(num_conds) if raw_masks[i] == 1]

            # Only sample if enough unmasked indices remain
            if len(unmasked_indices) >= add_masks:
                chosen = random.sample(unmasked_indices, add_masks)
            else:
                # Best-effort fallback: mask as many as we can
                chosen = random.sample(unmasked_indices, len(unmasked_indices))

            for i in chosen:
                raw_masks[i] = 0
                conds[i] = torch.zeros_like(conds[i])


        return x, conds, raw_masks





def create_dataloader(x_df,
                      cond_df=None,
                      cond_df_list=None,
                      mask_list=None,
                      batch_size=64,
                      extra_masking=False,
                      shuffle=True):
    """
    Conditioning:
    - If it's `cond_df`, it's wrapped into a single-element list behind the scenes.
    - If it's `cond_df_list`, it's a list in and out.
    - No conditioning: by omitting both, it only returns X.

    Masking:
    - With `mask_list` (must match `cond_df_list` length).
    - If `extra_masking=True`, additional random masking is applied at each __getitem__ call.
    """
    # Normalize cond-dfs
    if cond_df is not None:
        assert cond_df_list is None, "Pass either cond_df or cond_df_list, not both"
        cond_df_list = [cond_df]

    ds = MultiConditionDataset(x_df,
                               cond_df_list=cond_df_list,
                               mask_list=mask_list,
                               extra_masking=extra_masking)

    # Compute whether the *last* batch would be size 1
    n_samples = len(ds)
    remainder = n_samples % batch_size
    drop_last = (remainder == 1)

    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last)


def conditioning_tensor(df):
    # Convert DataFrame to tensor
    cond_tensor = torch.tensor(df.values).float()
    return cond_tensor
