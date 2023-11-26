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
Test calculation of DFT-D4 model.
"""

import pytest
import torch

from tad_dftd4._typing import DD
from tad_dftd4.charges import get_charges
from tad_dftd4.model import D4Model
from tad_dftd4.ncoord import coordination_number_d4
from tad_dftd4.utils import pack

from ..conftest import DEVICE
from .samples import samples

# only these references use `cn=True` and `q=True` for `gw`
sample_list = ["LiH", "SiH4", "MB16_43_03"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(name: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-5

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = sample["c6"].to(**dd)

    d4 = D4Model(numbers, **dd)

    cn = coordination_number_d4(numbers, positions)
    total_charge = torch.tensor(0.0, **dd)
    q = get_charges(numbers, positions, total_charge)

    gw = d4.weight_references(cn=cn, q=q)
    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == c6.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-5

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    refs = pack(
        [
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        ]
    )

    d4 = D4Model(numbers, **dd)

    cn = coordination_number_d4(numbers, positions)
    total_charge = torch.zeros(numbers.shape[0], **dd)
    q = get_charges(numbers, positions, total_charge)

    gw = d4.weight_references(cn=cn, q=q)
    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(refs.cpu(), abs=tol, rel=tol) == c6.cpu()
