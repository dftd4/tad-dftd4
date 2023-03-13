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
Test calculation of EEQ coordination number.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4._typing import Tensor
from tad_dftd4.data import cov_rad_d3
from tad_dftd4.ncoord import get_coordination_number_eeq as get_cn
from tad_dftd4.utils import pack

from .samples import samples

sample_list = ["MB16_43_01", "MB16_43_02"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    rcov = cov_rad_d3[numbers].type(dtype)
    cutoff = positions.new_tensor(30.0)
    ref = sample["cn_eeq"].type(dtype)

    cn = get_cn(numbers, positions, cutoff=cutoff, rcov=rcov, cn_max=None)
    assert pytest.approx(ref) == cn


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("cn_max", [49, 51.0, torch.tensor(49)])
def test_single_cnmax(dtype: torch.dtype, cn_max: int | float | Tensor) -> None:
    sample = samples["MB16_43_01"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    ref = sample["cn_eeq"].type(dtype)

    cn = get_cn(numbers, positions, cn_max=cn_max)
    assert pytest.approx(ref, abs=1e-5) == cn


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = pack(
        (
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        )
    )
    ref = pack(
        (
            sample1["cn_eeq"].type(dtype),
            sample2["cn_eeq"].type(dtype),
        )
    )

    cutoff = torch.tensor(30.0, dtype=dtype)
    cn = get_cn(numbers, positions, cutoff=cutoff, cn_max=None)
    assert pytest.approx(ref) == cn
