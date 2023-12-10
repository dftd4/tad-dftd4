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
Utility functions: Miscellaneous
================================

Utilities for working with tensors as well as translating between element
symbols and atomic numbers.
"""
from __future__ import annotations

import torch

from .._typing import Size, Tensor, TensorOrTensors
from ..constants import ATOMIC_NUMBER

__all__ = ["real_atoms", "real_pairs", "real_triples", "pack", "to_number"]


def real_atoms(numbers: Tensor) -> Tensor:
    """
    Create a mask for atoms, discerning padding and actual atoms.
    Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.

    Returns
    -------
    Tensor
        Mask for atoms that discerns padding and real atoms.
    """
    return numbers != 0


def real_pairs(numbers: Tensor, diagonal: bool = False) -> Tensor:
    """
    Create a mask for pairs of atoms from atomic numbers, discerning padding
    and actual atoms. Padding value is zero.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms.
    diagonal : bool, optional
        Flag for also writing `False` to the diagonal, i.e., to all pairs
        with the same indices. Defaults to `False`, i.e., writing False
        to the diagonal.

    Returns
    -------
    Tensor
        Mask for atom pairs that discerns padding and real atoms.
    """
    real = real_atoms(numbers)
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    if diagonal is False:
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask


def real_triples(
    numbers: torch.Tensor, diagonal: bool = False, self: bool = True
) -> Tensor:
    """
    Create a mask for triples from atomic numbers. Padding value is zero.

    Parameters
    ----------
    numbers : torch.Tensor
        Atomic numbers for all atoms.
    diagonal : bool, optional
        Flag for also writing `False` to the space diagonal, i.e., to all
        triples with the same indices. Defaults to `False`, i.e., writing False
        to the diagonal.
    self : bool, optional
        Flag for also writing `False` to all triples where at least two indices
        are identical. Defaults to `True`, i.e., not writing `False`.

    Returns
    -------
    Tensor
        Mask for triples.
    """
    real = real_pairs(numbers, diagonal=True)
    mask = real.unsqueeze(-3) * real.unsqueeze(-2) * real.unsqueeze(-1)

    if diagonal is False:
        mask *= ~torch.diag_embed(torch.ones_like(real))

    if self is False:
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-3, dim2=-2)
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-3, dim2=-1)
        mask *= ~torch.diag_embed(torch.ones_like(real), offset=0, dim1=-2, dim2=-1)

    return mask


def pack(
    tensors: TensorOrTensors,
    axis: int = 0,
    value: int | float = 0,
    size: Size | None = None,
) -> Tensor:
    """
    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Parameters
    ----------
    tensors : list[Tensor] | tuple[Tensor] | Tensor
        List of tensors to be packed, all with identical dtypes.
    axis : int
        Axis along which tensors should be packed; 0 for first axis -1
        for the last axis, etc. This will be a new dimension.
    value : int | float
        The value with which the tensor is to be padded.
    size :
        Size of each dimension to which tensors should be padded.
        This to the largest size encountered along each dimension.

    Returns
    -------
    padded : Tensor
        Input tensors padded and packed into a single tensor.
    """
    if isinstance(tensors, Tensor):
        return tensors

    _count = len(tensors)
    _device = tensors[0].device
    _dtype = tensors[0].dtype

    if size is None:
        size = torch.tensor([i.shape for i in tensors]).max(0).values.tolist()

    padded = torch.full((_count, *size), value, dtype=_dtype, device=_device)

    for n, source in enumerate(tensors):
        padded[(n, *[slice(0, s) for s in source.shape])] = source

    if axis != 0:
        axis = padded.dim() + 1 + axis if axis < 0 else axis
        order = list(range(1, padded.dim()))
        order.insert(axis, 0)
        padded = padded.permute(order)

    return padded


def to_number(symbols: list[str]) -> Tensor:
    """
    Obtain atomic numbers from element symbols.


    Parameters
    ----------
    symbols : list[str]
        List of element symbols.

    Returns
    -------
    Tensor
        Atomic numbers corresponding to the given element symbols.
    """
    return torch.flatten(torch.tensor([ATOMIC_NUMBER[s.title()] for s in symbols]))
