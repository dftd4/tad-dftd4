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
Check autograd with respect to positions and damping parameters.
"""
from __future__ import annotations

import pytest
import torch
from torch.autograd.gradcheck import gradcheck

from tad_dftd4._typing import Tensor
from tad_dftd4.disp import dftd4

from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01"]


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_param(name) -> None:
    dtype = torch.float64
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charge = positions.new_tensor(0.0)

    param = (
        positions.new_tensor(1.00000000).requires_grad_(True),
        positions.new_tensor(0.78981345).requires_grad_(True),
        positions.new_tensor(1.00000000).requires_grad_(True),
        positions.new_tensor(0.49484001).requires_grad_(True),
        positions.new_tensor(5.73083694).requires_grad_(True),
    )
    label = ("s6", "s8", "s9", "a1", "a2")

    def func(*inputs: Tensor) -> Tensor:
        input_param = {label[i]: inputs[i] for i in range(len(inputs))}
        return dftd4(numbers, positions, charge, input_param)

    assert gradcheck(func, param)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_positions(name: str) -> None:
    dtype = torch.float64
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charge = positions.new_tensor(0.0)

    # TPSS0-D4-ATM parameters
    param = {
        "s6": positions.new_tensor(1.0),
        "s8": positions.new_tensor(1.85897750),
        "s9": positions.new_tensor(1.0),
        "s10": positions.new_tensor(0.0),
        "alp": positions.new_tensor(16.0),
        "a1": positions.new_tensor(0.44286966),
        "a2": positions.new_tensor(4.60230534),
    }

    pos = positions.detach().clone().requires_grad_(True)

    def func(positions: Tensor) -> Tensor:
        return dftd4(numbers, positions, charge, param)

    assert gradcheck(func, pos)
