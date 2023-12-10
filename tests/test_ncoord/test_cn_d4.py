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
Test calculation of DFT-D4 coordination number.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4._typing import DD
from tad_dftd4.data import cov_rad_d3, pauling_en
from tad_dftd4.ncoord import coordination_number_d4 as get_cn
from tad_dftd4.utils import pack

from ..conftest import DEVICE
from .samples import samples

sample_list = ["MB16_43_01", "MB16_43_02", "MB16_43_02"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    rcov = cov_rad_d3.to(**dd)[numbers]
    en = pauling_en.to(**dd)[numbers]
    cutoff = torch.tensor(30.0, **dd)
    ref = sample["cn_d4"].to(**dd)

    cn = get_cn(numbers, positions, rcov=rcov, en=en, cutoff=cutoff)
    assert pytest.approx(cn.cpu()) == ref.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["cn_d4"].to(**dd),
            sample2["cn_d4"].to(**dd),
        )
    )

    cn = get_cn(numbers, positions)
    assert pytest.approx(cn.cpu()) == ref.cpu()
