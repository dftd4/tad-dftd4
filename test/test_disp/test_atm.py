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

from tad_dftd4.disp import dispersion3
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
    ref = sample["disp3"].to(**dd)

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

    model = D4Model(numbers, **dd)
    cn = cn_d4(numbers, positions)
    weights = model.weight_references(cn, q=None)
    c6 = model.get_atomic_c6(weights)
    cutoff = torch.tensor(40.0, **dd)

    energy = dispersion3(numbers, positions, param, c6, cutoff=cutoff)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu().cpu(), abs=tol) == energy.cpu()


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
    ref = pack(
        [
            sample1["disp3"].to(**dd),
            sample2["disp3"].to(**dd),
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

    model = D4Model(numbers, **dd)
    cn = cn_d4(numbers, positions)
    weights = model.weight_references(cn, q=None)
    c6 = model.get_atomic_c6(weights)

    energy = dispersion3(numbers, positions, param, c6)

    assert energy.dtype == dtype
    assert pytest.approx(ref.cpu(), abs=tol) == energy.cpu()
