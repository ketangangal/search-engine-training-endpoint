import torch
import numpy as np
import time


def set_seed(seed_value: int = 42) -> None:
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_unique_filename(filename, ext):
    return time.strftime(f"{filename}_%Y_%m_%d_%H_%M.{ext}")


print(get_unique_filename("model", "pth"))
