# ─────────────── Data & output directories ───────────────
DATA_DIR   = "../../datasets_TCGA"
OUTPUT_DIR = "./checkpoints"

# ─────────────── Default hyperparameter‐search grid ───────────────
default_grid = {
    "bottleneck_dim": [32],
    "n_layers":       [2],
    "batch_size":     [64],
    "num_epochs":     [10000],
    "learning_rate":  [1e-3],
    "weight_decay":   [1e-5],
    "patience":       [10],
    "shrink_exponent": [1.0],
}

# ─────────────── Per‑modality config (merging defaults + overrides) ───────────────
MODALITY_CONFIG = {
    "rnaseq": {
        "data_dir":   DATA_DIR,
        "output_dir": OUTPUT_DIR,
        
        "param_grid": {
            **default_grid,
            "n_layers": [3],
            "shrink_exponent": [1.0],
        },
    },
    "rppa": {
        "data_dir":   DATA_DIR,
        "output_dir": OUTPUT_DIR,
        
        "param_grid": {
            **default_grid,
            "n_layers": [2],
            "shrink_exponent": [1.0],
        },
    },
    "cna": {
        "data_dir":   DATA_DIR,
        "output_dir": OUTPUT_DIR,
        
        "param_grid": {
            **default_grid,
            "n_layers": [2],
            "shrink_exponent": [3.0],
        },
    },
}
