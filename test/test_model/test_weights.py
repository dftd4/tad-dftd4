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

from math import sqrt

import pytest
import torch
import torch.nn.functional as F

from tad_dftd4.charges import get_charges
from tad_dftd4.model import D4Model
from tad_dftd4.ncoord import get_coordination_number_d4
from tad_dftd4.utils import pack

from .samples import samples


def single(
    name: str,
    dtype: torch.dtype,
    with_cn: bool,
    with_q: bool,
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 20
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    d4 = D4Model(numbers, dtype=dtype)

    if with_cn is True:
        cn = get_coordination_number_d4(numbers, positions)
    else:
        cn = None  # positions.new_zeros(numbers.shape)

    if with_q is True:
        q = get_charges(numbers, positions, positions.new_tensor(0.0))
    else:
        q = None  # positions.new_zeros(numbers.shape)

    gwvec = d4.weight_references(cn, q)

    # pad reference tensor to always be of shape `(natoms, 7)`
    src = sample["gw"].type(dtype)
    ref = F.pad(
        input=src,
        pad=(0, 0, 0, 7 - src.size(0)),
        mode="constant",
        value=0,
    ).mT

    assert gwvec.shape == ref.shape
    assert pytest.approx(gwvec, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_mb16_43_01(dtype: torch.dtype) -> None:
    single("MB16_43_01", dtype, with_cn=True, with_q=False)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_mb16_43_02(dtype: torch.dtype) -> None:
    single("MB16_43_02", dtype, with_cn=False, with_q=True)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_mb16_43_03(dtype: torch.dtype) -> None:
    single("MB16_43_03", dtype, with_cn=True, with_q=True)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_sih4(dtype: torch.dtype) -> None:
    single("SiH4", dtype, with_cn=True, with_q=True)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_lih(dtype: torch.dtype) -> None:
    single("LiH", dtype, with_cn=True, with_q=True)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4", "MB16_43_03"])
def test_batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 20
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

    d4 = D4Model(numbers, dtype=dtype)

    cn = get_coordination_number_d4(numbers, positions)
    total_charge = positions.new_zeros(numbers.shape[0])
    q = get_charges(numbers, positions, total_charge)

    gwvec = d4.weight_references(cn, q)

    # pad reference tensor to always be of shape `(natoms, 7)`
    src1 = sample1["gw"].type(dtype)
    src2 = sample2["gw"].type(dtype)

    ref = pack(
        [
            F.pad(
                input=src1,
                pad=(0, 0, 0, 7 - src1.size(0)),
                mode="constant",
                value=0,
            ).mT,
            src2.mT,
        ]
    )

    assert gwvec.shape == ref.shape
    assert pytest.approx(gwvec, abs=tol) == ref
