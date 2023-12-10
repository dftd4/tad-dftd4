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
Testing dispersion gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4 import dftd4, utils
from tad_dftd4._typing import DD

from ..molecules import mols as samples

tol = 1e-8

device = None

# sample, which previously failed with NaN's in tad-dftd3
numbers = torch.tensor([6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 7, 8, 8, 8])
positions = torch.tensor(
    [
        [-1.0981, +0.1496, +0.1346],
        [-0.4155, +1.2768, +0.3967],
        [+0.9426, +0.7848, +0.1307],
        [+2.1708, +1.3814, -0.0347],
        [+3.3234, +0.5924, -0.1535],
        [+3.1564, -0.8110, -0.0285],
        [+1.8929, -1.4673, +0.0373],
        [+0.8498, -0.5613, +0.0109],
        [-0.7751, +2.2970, +0.5540],
        [+2.3079, +2.4725, -0.1905],
        [+4.3031, +0.9815, -0.4599],
        [+4.0011, -1.4666, -0.0514],
        [+1.8340, -2.5476, -0.1587],
        [-2.5629, -0.0306, -0.1458],
        [-3.0792, +1.0280, -0.3225],
        [-3.0526, -1.1594, +0.1038],
        [-0.4839, -0.9612, -0.0048],
    ],
)
charge = torch.tensor(0.0)

param = {
    "s6": torch.tensor(1.00000000),
    "s8": torch.tensor(0.78981345),
    "s9": torch.tensor(1.00000000),
    "a1": torch.tensor(0.49484001),
    "a2": torch.tensor(5.73083694),
}


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    nums = numbers.to(device=device)
    pos = positions.to(**dd)
    chrg = charge.to(**dd)
    par = {k: v.to(**dd) for k, v in param.items()}

    pos.requires_grad_(True)

    energy = dftd4(nums, pos, chrg, par)
    assert not torch.isnan(energy).any(), "Energy contains NaN values"

    energy.sum().backward()

    assert pos.grad is not None
    grad_backward = pos.grad.clone()

    # also zero out gradients when using `.backward()`
    pos.detach_()
    pos.grad.data.zero_()

    assert not torch.isnan(grad_backward).any(), "Gradient contains NaN values"


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["LiH", "SiH4"])
def test_batch(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    nums = utils.pack(
        (
            numbers.to(device=device),
            samples[name]["numbers"].to(device=device),
        )
    )
    pos = utils.pack(
        (
            positions.to(**dd),
            samples[name]["positions"].to(**dd),
        )
    )
    chrg = torch.tensor([0.0, 0.0], **dd)
    par = {k: v.to(**dd) for k, v in param.items()}

    pos.requires_grad_(True)

    energy = dftd4(nums, pos, chrg, par)
    assert not torch.isnan(energy).any(), "Energy contains NaN values"

    energy.sum().backward()

    assert pos.grad is not None
    grad_backward = pos.grad.clone()

    # also zero out gradients when using `.backward()`
    pos.detach_()
    pos.grad.data.zero_()

    assert not torch.isnan(grad_backward).any(), "Gradient contains NaN values"
