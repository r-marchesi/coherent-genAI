import numpy as np
import itertools
from math import floor

def generate_masks(n_datasets: int, total_samples: int, seed: int = 23):
    """
    Generates integer masks (0 = masked out, 1 = keep) for n_datasets over total_samples such that:
    - Each sample is assigned to a combination of 1 to n_datasets.
    - Each combination group has the same number of samples (as close as possible).
    - Leftover samples are assigned to all datasets (no masking).
    
    Returns:
        A list of n_datasets integer NumPy arrays of shape (total_samples,)
    """
    rng = np.random.default_rng(seed)

    # Generate all combinations of dataset indices, sizes 1 to n
    combinations = []
    for i in range(1, n_datasets + 1):
        combos = list(itertools.combinations(range(n_datasets), i))
        combinations.extend(combos)

    num_combinations = len(combinations)
    samples_per_combination = floor(total_samples / num_combinations)

    # Shuffle the indices
    indices = np.arange(total_samples)
    rng.shuffle(indices)

    # Assign samples to each combination
    assignment = {}
    idx_pointer = 0
    for combo in combinations:
        for _ in range(samples_per_combination):
            if idx_pointer >= total_samples:
                break
            index = indices[idx_pointer]
            assignment[index] = combo
            idx_pointer += 1

    # Leftover samples go to the full-combo (all datasets)
    full_combo = tuple(range(n_datasets))
    for i in range(idx_pointer, total_samples):
        index = indices[i]
        assignment[index] = full_combo

    # Create masks with 0 (masked) and 1 (keep)
    masks = [np.zeros(total_samples, dtype=int) for _ in range(n_datasets)]
    for index, combo in assignment.items():
        for dataset_id in combo:
            masks[dataset_id][index] = 1

    return masks