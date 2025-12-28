import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Set to True if input sizes are consistent for speed
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_directories(config):
    """Ensures that the necessary output directories exist."""
    os.makedirs(config.paths.model_save_dir, exist_ok=True)
    os.makedirs(config.paths.figures_dir, exist_ok=True)
    os.makedirs(config.paths.results_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)

# Add other utility functions here if needed