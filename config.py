"""
Configuration and utility functions for HPO script.
"""

import random
import numpy as np
import torch

# Hyperparameter search space
HYPER_SPACE = {
    "lr":   (1e-6, 5e-5),     # continuous range for learning rate
    "bs":   [8, 16, 32],      # categorical options for batch size
    "epoch": (3, 5)           # integer range (inclusive) for number of epochs
}

# Set random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
