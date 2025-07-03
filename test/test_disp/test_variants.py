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
from tad_mctc.ncoord import cn_d4
from tad_mctc.typing import DD

from tad_dftd4.damping import Damping, Param
from tad_dftd4.damping.functions import (
    MZeroDamping,
    OptimisedPowerDamping,
    RationalDamping,
    ZeroDamping,
)
from tad_dftd4.dispersion import Disp, TwoBodyTerm
from tad_dftd4.dispersion.d4 import D4ATMExact, DispD4Exact

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_classes(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    charge = torch.tensor(0.0, **dd)

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

    d1 = DispD4Exact(**dd)
    e1 = d1.calculate(numbers, positions, charge, param)

    assert e1.dtype == dtype
    assert e1.shape == numbers.shape

    # replicate DispD4Exact class

    d2 = Disp(cn_fn=cn_d4, model="d4", **dd)

    twobody_term = TwoBodyTerm(
        damping_fn=RationalDamping(),
        charge_dependent=True,
    )
    threebody_term = D4ATMExact(
        damping_fn=ZeroDamping(),
        charge_dependent=False,
    )

    d2.register(twobody_term)
    d2.register(threebody_term)

    e2 = d2.calculate(numbers, positions, charge, param)

    assert e2.dtype == dtype
    assert e2.shape == numbers.shape
    assert pytest.approx(e1.cpu()) == e2.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "damping_fn",
    [RationalDamping(), ZeroDamping(), MZeroDamping(), OptimisedPowerDamping()],
)
def test_damping(dtype: torch.dtype, damping_fn: Damping) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    charge = torch.tensor(0.0, **dd)

    # TPSSh-D4-ATM parameters
    param = Param(
        s6=torch.tensor(1.00000000, **dd),
        rs6=torch.tensor(1.00000000, **dd),
        s8=torch.tensor(1.85897750, **dd),
        rs8=torch.tensor(1.00000000, **dd),
        s9=torch.tensor(1.00000000, **dd),
        rs9=torch.tensor(1.00000000, **dd),
        alp=torch.tensor(16.000000, **dd),
        bet=torch.tensor(1.0000000, **dd),
        a1=torch.tensor(0.44286966, **dd),
        a2=torch.tensor(4.60230534, **dd),
    )

    disp = Disp(model="d4s", **dd)
    twobody_term = TwoBodyTerm(
        damping_fn=damping_fn,
        charge_dependent=False,
    )
    disp.register(twobody_term)

    energy = disp.calculate(numbers, positions, charge, param)

    assert energy.dtype == dtype
    assert energy.shape == numbers.shape
