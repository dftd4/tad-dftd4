"""
Collection of utility functions for testing.
"""

import torch

from tad_dftd4.typing import Tensor


def reshape_fortran(x: Tensor, shape: torch.Size | tuple):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def get_device_from_str(s: str) -> torch.device:
    """
    Convert device name to `torch.device`. Critically, this also sets the index
    for CUDA devices to `torch.cuda.current_device()`.
    Parameters
    ----------
    s : str
        Name of the device as string.
    Returns
    -------
    torch.device
        Device as torch class.
    Raises
    ------
    KeyError
        Unknown device name is given.
    """
    d = {
        "cpu": torch.device("cpu"),
        "cuda": torch.device("cuda", index=torch.cuda.current_device()),
    }

    if s not in d:
        raise KeyError(f"Unknown device '{s}' given.")

    return d[s]
