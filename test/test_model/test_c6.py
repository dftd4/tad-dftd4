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
Test calculation of DFT-D4 model.
"""

import pytest
import torch
import torch.nn.functional as F
from tad_mctc.batch import pack
from tad_mctc.typing import DD

from tad_dftd4.model import D4Model, D4SModel

from ..conftest import DEVICE
from .samples import samples

# only these references use `cn=True` and `q=True` for `gw`
sample_list = ["LiH", "SiH4", "MB16_43_03"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(name: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    tol = 1e-4 if dtype == torch.float else 1e-5
    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = sample["c6"].to(**dd)

    d4 = D4Model(numbers, **dd)

    # pad reference tensor to always be of shape `(natoms, 7)`
    src = sample["gw"].to(**dd)
    gw = F.pad(
        input=src,
        pad=(0, 0, 0, 7 - src.size(0)),
        mode="constant",
        value=0,
    ).mT

    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(ref.cpu(), rel=tol) == c6.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_d4s(name: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    tol = 1e-4 if dtype == torch.float else 1e-5
    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = sample["c6_d4s"].to(**dd)

    d4 = D4SModel(numbers, **dd)

    # pad reference tensor to always be of shape `(natoms, 7)`
    src = sample["gw_d4s"].to(**dd)
    gw = F.pad(
        input=src,
        pad=(0, 0, 0, 0, 0, 7 - src.size(0)),
        mode="constant",
        value=0,
    ).permute(2, 1, 0)

    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(ref.cpu(), rel=tol) == c6.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    tol = 1e-4 if dtype == torch.float else 1e-5
    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    refs = pack(
        [
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        ]
    )

    d4 = D4Model(numbers, **dd)

    # pad reference tensor to always be of shape `(natoms, 7)`
    src1 = sample1["gw"].to(**dd)
    src2 = sample2["gw"].to(**dd)

    gw = pack(
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

    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(refs.cpu(), rel=tol) == c6.cpu()
