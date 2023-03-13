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
Collection of utility functions for testing.
"""
from __future__ import annotations

import torch

from tad_dftd4._typing import Tensor


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
