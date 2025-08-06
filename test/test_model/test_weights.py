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

For an explanation of the unusual loose tolerances, see `test_charges.py`.
"""
import pytest
import torch
import torch.nn.functional as F
from tad_mctc._version import __tversion__
from tad_mctc.autograd import jacrev
from tad_mctc.batch import pack
from tad_mctc.ncoord import cn_d4
from tad_mctc.typing import DD

from tad_dftd4.model import D4Model, D4SModel

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


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
@pytest.mark.parametrize("name", ["LiH", "SiH4", "MB16_43_03"])
@pytest.mark.parametrize("model", ["d4", "d4s"])
def test_grad_q(name: str, model: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)

    q = sample["q"].to(**dd)
    q_grad = q.detach().clone().requires_grad_(True)

    if model == "d4":
        d4 = D4Model(numbers, **dd)
    elif model == "d4s":
        d4 = D4SModel(numbers, **dd)
    else:
        raise ValueError(f"Invalid model: {model}")

    cn = cn_d4(numbers, positions)

    # analytical gradient
    _, dgwdq_ana = d4.weight_references(cn, q, with_dgwdq=True)

    # autodiff gradient
    dgwdq_auto = jacrev(d4.weight_references, 1)(cn, q_grad)
    assert isinstance(dgwdq_auto, torch.Tensor)
    dgwdq_auto = dgwdq_auto.sum(-1).detach()

    assert pytest.approx(dgwdq_auto.cpu(), abs=1e-6) == dgwdq_ana.cpu()


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4", "MB16_43_03"])
@pytest.mark.parametrize("model", ["d4", "d4s"])
def test_grad_q_batch(name1: str, name2: str, model: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

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

    q_grad = q.detach().clone().requires_grad_(True)

    if model == "d4":
        d4 = D4Model(numbers, **dd)
    elif model == "d4s":
        d4 = D4SModel(numbers, **dd)
    else:
        raise ValueError(f"Invalid model: {model}")

    cn = cn_d4(numbers, positions)

    # analytical gradient
    _, dgwdq_ana = d4.weight_references(cn, q, with_dgwdq=True)

    # autodiff gradient
    dgwdq_auto = jacrev(d4.weight_references, 1)(cn, q_grad)
    assert isinstance(dgwdq_auto, torch.Tensor)
    dgwdq_auto = dgwdq_auto.sum((-1, -2)).detach()

    assert dgwdq_auto.shape == dgwdq_ana.shape
    assert pytest.approx(dgwdq_auto.cpu(), abs=1e-6) == dgwdq_ana.cpu()


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
@pytest.mark.parametrize("name", ["LiH", "SiH4", "MB16_43_03"])
def test_grad_cn(name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)

    pos = sample["positions"].to(**dd)
    pos_grad = pos.detach().clone().requires_grad_(True)

    d4 = D4Model(numbers, **dd)

    # analytical gradient
    cn = cn_d4(numbers, pos)
    _, dgwdq_ana = d4.weight_references(cn, with_dgwdcn=True)

    # autodiff gradient
    cn_grad = cn_d4(numbers, pos_grad)
    dgwdcn_auto = jacrev(d4.weight_references, 0)(cn_grad)
    assert isinstance(dgwdcn_auto, torch.Tensor)
    dgwdcn_auto = dgwdcn_auto.sum(-1).detach()

    assert pytest.approx(dgwdcn_auto.cpu(), abs=1e-6) == -dgwdq_ana.cpu()


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires torch>=2.0.0")
@pytest.mark.parametrize("name", ["LiH", "SiH4", "MB16_43_03"])
def test_grad_both(name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": torch.float64}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)

    pos = sample["positions"].to(**dd)
    pos_grad = pos.detach().clone().requires_grad_(True)

    q = sample["q"].to(**dd)
    q_grad = q.detach().clone().requires_grad_(True)

    d4 = D4Model(numbers, **dd)

    # analytical gradient
    cn = cn_d4(numbers, pos)
    _, dgwdq_ana, dgwdcn_ana = d4.weight_references(
        cn, q, with_dgwdcn=True, with_dgwdq=True
    )

    # autodiff gradient
    cn_grad = cn_d4(numbers, pos_grad)
    dgwdq_auto, dgwdcn_auto = jacrev(
        d4.weight_references,
        (0, 1),  # type: ignore
    )(cn_grad, q_grad)

    assert isinstance(dgwdcn_auto, torch.Tensor)
    dgwdcn_auto = dgwdcn_auto.sum(-1).detach()

    assert pytest.approx(dgwdcn_auto.cpu(), abs=1e-6) == dgwdcn_ana.cpu()

    assert isinstance(dgwdq_auto, torch.Tensor)
    dgwdq_auto = dgwdq_auto.sum(-1).detach()

    assert pytest.approx(dgwdq_auto.cpu(), abs=1e-6) == -dgwdq_ana.cpu()
