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
from tad_mctc.batch import pack
from tad_mctc.ncoord import cn_d4

from tad_dftd4.model import D4Model
from tad_dftd4.typing import DD

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
    q = sample["q"].to(**dd)
    ref = sample["c6"].to(**dd)

    d4 = D4Model(numbers, **dd)

    cn = cn_d4(numbers, positions)
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
    q = pack(
        [
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        ]
    )
    refs = pack(
        [
            sample1["c6"].to(**dd),
            sample2["c6"].to(**dd),
        ]
    )

    d4 = D4Model(numbers, **dd)

    cn = cn_d4(numbers, positions)
    gw = d4.weight_references(cn=cn, q=q)
    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(refs.cpu(), abs=tol, rel=tol) == c6.cpu()
