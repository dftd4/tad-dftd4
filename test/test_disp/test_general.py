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
Check shape of tensors.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4.disp import dftd4


def test_fail() -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    charge = torch.tensor(0.0)
    param = {"s6": torch.tensor(1.0)}

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0])
        dftd4(numbers, positions, charge, param, rcov=rcov)

    # expectation valus (r4r2) wrong shape
    with pytest.raises(ValueError):
        r4r2 = torch.tensor([1.0])
        dftd4(numbers, positions, charge, param, r4r2=r4r2)

    # atomic partial charges wrong shape
    with pytest.raises(ValueError):
        q = torch.tensor([1.0])
        dftd4(numbers, positions, charge, param, q=q)

    # wrong numbers (give charges, otherwise test fails in EEQ, not in disp)
    with pytest.raises(ValueError):
        q = torch.tensor([0.5, -0.5])
        nums = torch.tensor([1])
        dftd4(nums, positions, charge, param, q=q)
