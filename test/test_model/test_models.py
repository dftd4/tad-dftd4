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
from tad_mctc.math import einsum
from tad_mctc.ncoord import cn_d4
from tad_mctc.typing import Tensor

from tad_dftd4.model import D4Model, D4SModel
from tad_dftd4.model.d4s import D4SDebug
from tad_dftd4.model.utils import trapzd_noref
from tad_dftd4.typing import DD

from ..conftest import DEVICE
from .samples import samples

# only these references use `cn=True` and `q=True` for `gw`
sample_list = ["LiH", "SiH4", "MB16_43_03"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("model", ["d4", "d4s"])
def test_single(name: str, dtype: torch.dtype, model: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-5

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    q = sample["q"].to(**dd)

    if model == "d4":
        d4 = D4Model(numbers, **dd)
        ref = sample["c6"].to(**dd)
    elif model == "d4s":
        d4 = D4SModel(numbers, **dd)
        ref = sample["c6_d4s"].to(**dd)
    else:
        raise ValueError(f"Unknown model: {model}")

    cn = cn_d4(numbers, positions)
    gw = d4.weight_references(cn=cn, q=q)
    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == c6.cpu()

    # Calculate from weighted pols (only sums equivalent)
    if model == "d4":
        w = einsum("...nr,...nrw->...nw", gw, d4._get_alpha())
        _c6 = trapzd_noref(w).sum()
    elif model == "d4s":
        w = einsum("...jia,...iaw->...jiw", gw, d4._get_alpha())
        _c6 = _trapzd(w).sum()
    else:
        raise ValueError(f"Unknown model: {model}")

    assert pytest.approx(c6.sum().cpu(), rel=tol) == _c6.cpu()
    assert pytest.approx(ref.sum().cpu(), rel=tol) == _c6.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("model", ["d4", "d4s"])
def test_batch(name1: str, name2: str, dtype: torch.dtype, model: str) -> None:
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

    if model == "d4":
        d4 = D4Model(numbers, **dd)
        refs = pack(
            [
                sample1["c6"].to(**dd),
                sample2["c6"].to(**dd),
            ]
        )
    elif model == "d4s":
        d4 = D4SModel(numbers, **dd)
        refs = pack(
            [
                sample1["c6_d4s"].to(**dd),
                sample2["c6_d4s"].to(**dd),
            ]
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    cn = cn_d4(numbers, positions)
    gw = d4.weight_references(cn=cn, q=q)
    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(refs.cpu(), abs=tol, rel=tol) == c6.cpu()

    # Calculate from weighted pols (only sums equivalent)
    if model == "d4":
        w = einsum("...nr,...nrw->...nw", gw, d4._get_alpha())
        _c6 = trapzd_noref(w).sum((-2, -1))
    elif model == "d4s":
        w = einsum("...jia,...iaw->...jiw", gw, d4._get_alpha())
        _c6 = _trapzd(w).sum((-2, -1))
    else:
        raise ValueError(f"Unknown model: {model}")

    assert pytest.approx(c6.sum((-2, -1)).cpu(), rel=tol) == _c6.cpu()
    assert pytest.approx(refs.sum((-2, -1)).cpu(), rel=tol) == _c6.cpu()


@pytest.mark.parametrize("model", ["d4", "d4s"])
def test_ref_charges_d4(model: str) -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])

    if model == "d4":
        model_eeq = D4Model(numbers, ref_charges="eeq")
        model_gfn2 = D4Model(numbers, ref_charges="gfn2")
    elif model == "d4s":
        model_eeq = D4SModel(numbers, ref_charges="eeq")
        model_gfn2 = D4SModel(numbers, ref_charges="gfn2")
    else:
        raise ValueError(f"Unknown model: {model}")

    weights_eeq = model_eeq.weight_references()
    weights_gfn2 = model_gfn2.weight_references()

    assert pytest.approx(weights_eeq.cpu(), abs=1e-1) == weights_gfn2.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("model", ["d4", "d4s"])
def test_weighted_pol(name: str, dtype: torch.dtype, model: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    q = sample["q"].to(**dd)

    if model == "d4":
        d4 = D4Model(numbers, **dd)
        ref = sample["c6"].to(**dd)
    elif model == "d4s":
        d4 = D4SModel(numbers, **dd)
        ref = sample["c6_d4s"].to(**dd)
    else:
        raise ValueError(f"Unknown model: {model}")

    cn = cn_d4(numbers, positions)
    gw = d4.weight_references(cn=cn, q=q)
    aw = d4.get_weighted_pols(gw)
    c6 = trapzd_noref(aw)

    # Molecular C6 is always smaller than sqrt(C6_ii * C6_jj).
    # (Cauchy-Schwarz inequality)
    diff = c6.sum() - ref.sum()
    assert diff < 0.0


def test_d4sdebug() -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])
    m_d4 = D4Model(numbers)
    m_debug = D4SDebug(numbers)

    weights_d4 = m_d4.weight_references()
    weights_debug = m_debug.weight_references()
    assert weights_d4.shape == weights_debug.shape[1:]

    c6_d4 = m_d4.get_atomic_c6(weights_d4)
    c6_debug = m_debug.get_atomic_c6(weights_debug)
    assert c6_d4.shape == c6_debug.shape == (5, 5)
    assert pytest.approx(c6_d4.cpu()) == c6_debug.cpu()


def _trapzd(pol: Tensor) -> Tensor:
    thopi = 3.0 / 3.141592653589793238462643383279502884197

    weights = torch.tensor(
        [
            2.4999500000000000e-002,
            4.9999500000000000e-002,
            7.5000000000000010e-002,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1500000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.3500000000000000,
            0.5000000000000000,
            0.7500000000000000,
            1.0000000000000000,
            1.7500000000000000,
            2.5000000000000000,
            1.2500000000000000,
        ],
        device=pol.device,
        dtype=pol.dtype,
    )

    return thopi * einsum("w,...ijw,...jiw->...ij", *(weights, pol, pol))
