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

from tad_dftd4.charges import get_charges
from tad_dftd4.model import D4Model
from tad_dftd4.ncoord import get_coordination_number_d4
from tad_dftd4.utils import pack

from .samples import samples

# only these references use `cn=True` and `q=True` for `gw`
sample_list = ["LiH", "SiH4", "MB16_43_03"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(name: str, dtype: torch.dtype) -> None:
    tol = 1e-5
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["c6"]

    d4 = D4Model(numbers, dtype=dtype)

    cn = get_coordination_number_d4(numbers, positions)
    q = get_charges(numbers, positions, positions.new_tensor(0.0))

    gw = d4.weight_references(cn=cn, q=q)
    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(ref, abs=tol, rel=tol) == c6


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    tol = 1e-5
    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"],
            sample2["numbers"],
        ]
    )
    positions = pack(
        [
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        ]
    )
    refs = pack(
        [
            sample1["c6"],
            sample2["c6"],
        ]
    )

    d4 = D4Model(numbers, dtype=dtype)

    cn = get_coordination_number_d4(numbers, positions)
    q = get_charges(numbers, positions, positions.new_zeros(numbers.shape[0]))

    gw = d4.weight_references(cn=cn, q=q)
    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(refs, abs=tol, rel=tol) == c6
