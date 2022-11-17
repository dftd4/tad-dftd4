"""
Type annotations for `tad-dftd4`.
"""

from typing import Any, Callable, TypedDict

import torch
from torch import Tensor

from . import defaults

Size = tuple[int] | torch.Size
Sliceable = list[Tensor] | tuple[Tensor, ...]

CountingFunction = Callable[[Tensor, Tensor], Tensor]
WeightingFunction = Callable[[Tensor, Any], Tensor]
DampingFunction = Callable[[int, Tensor, Tensor, Tensor, Any], Tensor]


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""


class TensorLike:
    """
    Provide `device` and `dtype` for other classes.
    """

    __slots__ = ["__device", "__dtype"]

    def __init__(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        self.__device = (
            device if device is not None else torch.device(defaults.TORCH_DEVICE)
        )
        self.__dtype = dtype if dtype is not None else defaults.TORCH_DTYPE

    @property
    def device(self) -> torch.device:
        """The device on which the class object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by class object."""
        return self.__dtype
