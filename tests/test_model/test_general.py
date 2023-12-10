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
Test the correct handling of types in the `D4Model` class.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4.model import D4Model

from ..utils import get_device_from_str


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])
    model = D4Model(numbers).type(dtype)
    assert model.dtype == dtype


def test_change_type_fail() -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])
    model = D4Model(numbers)

    # trying to use setter
    with pytest.raises(AttributeError):
        model.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        model.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = get_device_from_str(device_str)
    numbers = torch.tensor([14, 1, 1, 1, 1])
    model = D4Model(numbers).to(device)
    assert model.device == device


def test_change_device_fail() -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])
    model = D4Model(numbers)

    # trying to use setter
    with pytest.raises(AttributeError):
        model.device = "cpu"
