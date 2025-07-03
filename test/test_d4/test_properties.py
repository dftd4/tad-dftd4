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
Test calculation of dispersion model properties.
"""
import pytest
import torch
from tad_mctc.batch import pack
from tad_mctc.math import einsum
from tad_mctc.ncoord import cn_d4

from tad_dftd4.cutoff import Cutoff
from tad_dftd4.disp import get_properties
from tad_dftd4.model import D4Model
from tad_dftd4.utils import trapzd, trapzd_noref
from tad_mctc.typing import DD

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

    qref = sample["q"].to(**dd)
    model = D4Model(numbers, **dd)
    cn = cn_d4(numbers, positions)
    weights = model.weight_references(cn, qref)
    c6ref = model.get_atomic_c6(weights).sum((-2, -1))

    _cn, _, _c6, _ = get_properties(numbers, positions)

    assert pytest.approx(cn.cpu(), rel=tol) == _cn.cpu()
    assert pytest.approx(c6ref.cpu(), rel=tol) == _c6.sum((-2, -1)).cpu()

    # Manually calculate C6 values

    aiw = model._get_alpha()  # pylint: disable=protected-access

    alpha1 = einsum("...nr,...nra->...nra", weights, aiw)
    c61 = trapzd(alpha1, alpha1).sum((-4, -3, -2, -1))

    alpha2 = einsum("...nr,...nra->...na", weights, aiw)
    c62 = trapzd_noref(alpha2, alpha2).sum((-2, -1))

    assert c6ref.shape == c61.shape
    assert c6ref.shape == c62.shape
    assert pytest.approx(c6ref.cpu(), rel=tol) == c61.cpu()
    assert pytest.approx(c6ref.cpu(), rel=tol) == c62.cpu()


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
    charge = torch.tensor([0.0, 0.0], **dd)
    qref = pack(
        [
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        ]
    )

    model = D4Model(numbers, **dd)
    cutoff = Cutoff(**dd)
    cn = cn_d4(numbers, positions)
    weights = model.weight_references(cn, qref)
    c6ref = model.get_atomic_c6(weights).sum((-2, -1))

    _cn, _, _c6, _ = get_properties(numbers, positions, charge, cutoff=cutoff)

    assert pytest.approx(cn.cpu(), rel=tol) == _cn.cpu()
    assert pytest.approx(c6ref.cpu(), rel=tol) == _c6.sum((-2, -1)).cpu()

    # Manually calculate C6 values

    aiw = model._get_alpha()  # pylint: disable=protected-access

    alpha1 = einsum("...nr,...nra->...nra", weights, aiw)
    c61 = trapzd(alpha1, alpha1).sum((-4, -3, -2, -1))

    alpha2 = einsum("...nr,...nra->...na", weights, aiw)
    c62 = trapzd_noref(alpha2, alpha2).sum((-2, -1))

    assert c6ref.shape == c61.shape
    assert c6ref.shape == c62.shape
    assert pytest.approx(c6ref.cpu(), rel=tol) == c61.cpu()
    assert pytest.approx(c6ref.cpu(), rel=tol) == c62.cpu()
