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
from tad_mctc.ncoord import cn_d4

from tad_dftd4 import data
from tad_dftd4.disp import dftd4, dispersion2
from tad_dftd4.model import D4Model
from tad_dftd4.typing import DD

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
    ref = sample["disp2"].to(**dd)

    # TPSSh-D4-ATM parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(1.85897750, **dd),
        "s9": torch.tensor(1.00000000, **dd),
        "s10": torch.tensor(0.0000000, **dd),
        "alp": torch.tensor(16.000000, **dd),
        "a1": torch.tensor(0.44286966, **dd),
        "a2": torch.tensor(4.60230534, **dd),
    }

    r4r2 = data.R4R2.to(**dd)[numbers]
    model = D4Model(numbers, **dd)
    cn = cn_d4(numbers, positions)
    weights = model.weight_references(cn, q)
    c6 = model.get_atomic_c6(weights)

    energy = dispersion2(numbers, positions, param, c6, r4r2)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_s9_zero(name: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 10

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)
    ref = sample["disp2"].to(**dd)

    # TPSSh-D4-ATM parameters
    param = {
        "s8": torch.tensor(1.85897750, **dd),
        "s9": torch.tensor(0.00000000, **dd),  # skip ATM
        "a1": torch.tensor(0.44286966, **dd),
        "a2": torch.tensor(4.60230534, **dd),
    }

    energy = dftd4(numbers, positions, charge, param)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["SiH4"])
def test_single_s10_one(name: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 10

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)
    ref = torch.tensor(
        [
            -8.8928018057670788e-04,
            -3.3765541880036940e-04,
            -3.3765541880036940e-04,
            -3.3765541880036940e-04,
            -3.3765541880036940e-04,
        ],
        **dd,
    )

    # TPSSh-D4-ATM parameters
    param = {
        "s8": torch.tensor(1.85897750, **dd),
        "s9": torch.tensor(0.00000000, **dd),  # skip ATM
        "s10": torch.tensor(1.0000000, **dd),  # quadrupole-quadrupole
        "a1": torch.tensor(0.44286966, **dd),
        "a2": torch.tensor(4.60230534, **dd),
    }

    energy = dftd4(numbers, positions, charge, param)

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
    q = pack(
        [
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        ]
    )

    ref = pack(
        [
            sample1["disp2"].to(**dd),
            sample2["disp2"].to(**dd),
        ]
    )

    # TPSSh-D4-ATM parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(1.85897750, **dd),
        "s9": torch.tensor(1.00000000, **dd),
        "s10": torch.tensor(0.0000000, **dd),
        "alp": torch.tensor(16.000000, **dd),
        "a1": torch.tensor(0.44286966, **dd),
        "a2": torch.tensor(4.60230534, **dd),
    }

    r4r2 = data.R4R2.to(**dd)[numbers]
    model = D4Model(numbers, **dd)
    cn = cn_d4(numbers, positions)
    weights = model.weight_references(cn, q)
    c6 = model.get_atomic_c6(weights)

    energy = dispersion2(numbers, positions, param, c6, r4r2)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()
