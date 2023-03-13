# This file is part of tad-dftd4.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2022 Marvin Friede
#
# tad-dftd4 is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad-dftd4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-dftd4. If not, see <https://www.gnu.org/licenses/>.
"""
Type annotations
================

This module contains all type annotations for this project.

Since typing still significantly changes across different Python versions,
all the special cases are handled here as well.
"""

from __future__ import annotations

import sys

# pylint: disable=unused-import
from typing import Any, Protocol, TypedDict

import torch
from torch import Tensor

from . import defaults

# Python 3.11
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# Python 3.10
if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

# starting with Python 3.9, type hinting generics have been moved
# from the "typing" to the "collections" module
# (see PEP 585: https://peps.python.org/pep-0585/)
if sys.version_info >= (3, 9):
    from collections.abc import Callable, Generator, Sequence
else:
    from typing import Callable, Generator, Sequence

# type aliases that do not require "from __future__ import annotations"
CountingFunction = Callable[[Tensor, Tensor], Tensor]
WeightingFunction = Callable[[Tensor, Any], Tensor]


if sys.version_info >= (3, 10):
    # "from __future__ import annotations" only affects type annotations
    # not type aliases, hence "|" is not allowed before Python 3.10

    Sliceable = list[Tensor] | tuple[Tensor, ...]
    Size = list[int] | tuple[int] | torch.Size
    TensorOrTensors = list[Tensor] | tuple[Tensor, ...] | Tensor
    DampingFunction = Callable[[int, Tensor, Tensor, dict[str, Tensor]], Tensor]
elif sys.version_info >= (3, 9):
    # in Python 3.9, "from __future__ import annotations" works with type
    # aliases but requires using `Union` from typing
    from typing import Union

    Sliceable = Union[list[Tensor], tuple[Tensor, ...]]
    Size = Union[list[int], tuple[int], torch.Size]
    TensorOrTensors = Union[list[Tensor], tuple[Tensor, ...], Tensor]

    # no Union here, same as 3.10
    DampingFunction = Callable[[int, Tensor, Tensor, dict[str, Tensor]], Tensor]
elif sys.version_info >= (3, 8):
    # in Python 3.8, "from __future__ import annotations" only affects
    # type annotations not type aliases
    from typing import Dict, List, Tuple, Union

    Sliceable = Union[List[Tensor], Tuple[Tensor, ...]]
    Size = Union[List[int], Tuple[int], torch.Size]
    TensorOrTensors = Union[List[Tensor], Tuple[Tensor, ...], Tensor]
    DampingFunction = Callable[[int, Tensor, Tensor, Dict[str, Tensor]], Tensor]
else:
    raise RuntimeError(
        f"'tad_dftd4' requires at least Python 3.8 (Python {sys.version_info.major}."
        f"{sys.version_info.minor}.{sys.version_info.micro} found)."
    )


class Molecule(TypedDict):
    """Representation of fundamental molecular structure (atom types and postions)."""

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""


class TensorLike:
    """
    Provide `device` and `dtype` as well as `to()` and `type()` for other
    classes.
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
    def device(self, *_):
        """
        Instruct users to use the ".to" method if wanting to change device.
        """
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by class object."""
        return self.__dtype

    @dtype.setter
    def dtype(self, *_):
        """
        Instruct users to use the `.type` method if wanting to change dtype.
        """
        raise AttributeError("Change object to dtype using the `.type` method")

    def type(self, dtype: torch.dtype) -> Self:
        """
        Returns a copy of the `TensorLike` instance with specified floating
        point type.
        This method creates and returns a new copy of the `TensorLike` instance
        with the specified dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Floating point type.

        Returns
        -------
        TensorLike
            A copy of the `TensorLike` instance with the specified dtype.

        Notes
        -----
        If the `TensorLike` instance has already the desired dtype `self` will
        be returned.
        """
        if self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `type` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        allowed_dtypes = (torch.float16, torch.float32, torch.float64)
        if dtype not in allowed_dtypes:
            raise ValueError(f"Only float types allowed (received '{dtype}').")

        args = {}
        for s in self.__slots__:
            if not s.startswith("__"):
                attr = getattr(self, s)
                if isinstance(attr, Tensor) or issubclass(type(attr), TensorLike):
                    if attr.dtype in allowed_dtypes:
                        attr = attr.type(dtype)  # type: ignore
                args[s] = attr

        return self.__class__(**args, dtype=dtype)

    def to(self, device: torch.device) -> Self:
        """
        Returns a copy of the `TensorLike` instance on the specified device.
        This method creates and returns a new copy of the `TensorLike` instance
        on the specified device "``device``".

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        TensorLike
            A copy of the `TensorLike` instance placed on the specified device.

        Notes
        -----
        If the `TensorLike` instance is already on the desired device `self`
        will be returned.
        """
        if self.device == device:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        args = {}
        for s in self.__slots__:
            if not s.startswith("__"):
                attr = getattr(self, s)
                if isinstance(attr, Tensor) or issubclass(type(attr), TensorLike):
                    attr = attr.to(device=device)  # type: ignore
                args[s] = attr

        return self.__class__(**args, device=device)
