# This file is part of tad-dftd4.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test calculation of two-body and three-body dispersion terms.
"""
import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.typing import DD

from tad_dftd4 import data
from tad_dftd4.cutoff import Cutoff
from tad_dftd4.damping import Param
from tad_dftd4.disp import dftd4
from tad_dftd4.model import D4Model

from ..conftest import DEVICE
from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01", "MB16_43_02", "AmF3", "actinides"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(name: str, dtype: torch.dtype) -> None:
    single(name, dtype)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["vancoh2"])
def test_single_large(name: str, dtype: torch.dtype) -> None:
    single(name, dtype)


def single(name: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 10

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    q = sample["q"].to(**dd)
    charge = torch.tensor(0.0, **dd)
    ref = sample["disp"].to(**dd)

    # TPSSh-D4-ATM parameters
    param = Param(
        s6=torch.tensor(1.00000000, **dd),
        s8=torch.tensor(1.85897750, **dd),
        s9=torch.tensor(1.00000000, **dd),
        s10=torch.tensor(0.0000000, **dd),
        alp=torch.tensor(16.000000, **dd),
        a1=torch.tensor(0.44286966, **dd),
        a2=torch.tensor(4.60230534, **dd),
    )

    model = D4Model(numbers, **dd)
    rcov = data.COV_D3(**dd)[numbers]
    r4r2 = data.R4R2(**dd)[numbers]
    cutoff = Cutoff(**dd)
    rvdw = data.VDW_PAIRWISE(**dd)[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]

    energy = dftd4(
        numbers,
        positions,
        charge,
        param,
        model=model,
        rcov=rcov,
        r4r2=r4r2,
        rvdw=rvdw,
        q=q,
        cutoff=cutoff,
    )

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    batch(name1, name2, dtype)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["vancoh2"])
def test_batch_large(name1: str, name2: str, dtype: torch.dtype) -> None:
    batch(name1, name2, dtype)


def batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 10

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

    charge = positions.new_zeros(numbers.shape[0])
    ref = pack(
        [
            sample1["disp"].to(**dd),
            sample2["disp"].to(**dd),
        ]
    )

    # TPSSh-D4-ATM parameters
    param = Param(
        s6=torch.tensor(1.00000000, **dd),
        s8=torch.tensor(1.85897750, **dd),
        s9=torch.tensor(1.00000000, **dd),
        s10=torch.tensor(0.0000000, **dd),
        alp=torch.tensor(16.000000, **dd),
        a1=torch.tensor(0.44286966, **dd),
        a2=torch.tensor(4.60230534, **dd),
    )

    energy = dftd4(numbers, positions, charge, param)
    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()
