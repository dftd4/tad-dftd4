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

For an explanation of the unusual loose tolerances, see `test_charges.py`.
"""
import pytest
import torch
import torch.nn.functional as F
from tad_mctc.batch import pack
from tad_mctc.ncoord import cn_d4

from tad_dftd4.model import D4Model
from tad_dftd4.typing import DD

from ..conftest import DEVICE
from .samples import samples


def single(
    name: str,
    dtype: torch.dtype,
    with_cn: bool,
    with_q: bool,
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-4 if dtype == torch.float32 else 1e-6

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    d4 = D4Model(numbers, **dd)

    if with_cn is True:
        cn = cn_d4(numbers, positions)
    else:
        cn = None  # positions.new_zeros(numbers.shape)

    if with_q is True:
        q = sample["q"].to(**dd)
    else:
        q = None  # positions.new_zeros(numbers.shape)

    gwvec = d4.weight_references(cn, q)

    # pad reference tensor to always be of shape `(natoms, 7)`
    src = sample["gw"].to(**dd)
    ref = F.pad(
        input=src,
        pad=(0, 0, 0, 7 - src.size(0)),
        mode="constant",
        value=0,
    ).mT

    assert gwvec.dtype == ref.dtype
    assert gwvec.shape == ref.shape
    assert pytest.approx(gwvec.cpu(), abs=tol) == ref.cpu()


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
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-4 if dtype == torch.float32 else 1e-6

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
    q = pack(
        [
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        ]
    )

    d4 = D4Model(numbers, **dd)

    cn = cn_d4(numbers, positions)
    gwvec = d4.weight_references(cn, q)

    # pad reference tensor to always be of shape `(natoms, 7)`
    src1 = sample1["gw"].to(**dd)
    src2 = sample2["gw"].to(**dd)

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

    assert gwvec.dtype == ref.dtype
    assert gwvec.shape == ref.shape
    assert pytest.approx(gwvec.cpu(), abs=tol) == ref.cpu()
