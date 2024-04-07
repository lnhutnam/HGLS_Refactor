import os
import random

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch import distributed as dist


def init_torch_seeds(seed=0):
    """Initialization seed for random, os, numpy, torch with cuda backend.

    Args:
        seed (int, optional): input integer for initlizing seed for re-procedure 
        results of machine leanring models. Defaults to 0.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # Disable hash randomization
    np.random.seed(seed)
    
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def get_rank():
    """Function for getting rank in distributed training (multiple GPU).

    Returns:
        int: ranking of current round.
    """
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    """Function for getting world size in distributed training (multiple GPU).

    Returns:
        int: the value of environment variable `WORLD_SIZE`
    """
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    """
    Function for synchronizing in distributed training (multiple GPU).
    """
    if get_world_size() > 1:
        dist.barrier()


def get_device(args):
    """Function for getting device in distributed training (multiple GPU).

    Args:
        args (argument object): input argument object.

    Returns:
        torch.device: torch device object.
    """
    if args.gpus:
        device = torch.device(args.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device
