import torch
import numpy as np


def set_seed(seed_value: int = 42) -> None:
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_seed()
