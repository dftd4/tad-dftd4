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
General checks for dispersion classes.
"""
import pytest
import torch

from tad_dftd4.cutoff import Cutoff
from tad_dftd4.damping import Damping, Param
from tad_dftd4.damping.functions import OptimisedPowerDamping, RationalDamping
from tad_dftd4.dispersion import Disp, DispD4, DispTerm, TwoBodyTerm
from tad_dftd4.model import D4Model, ModelInst


def test_fail_class() -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    charge = torch.tensor(0.0)
    param = Param(s6=torch.tensor(1.0))

    disp = Disp()
    disp.register(
        TwoBodyTerm(damping_fn=RationalDamping(), charge_dependent=True)
    )

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0])
        disp.calculate(numbers, positions, charge, param, rcov=rcov)

    # expectation valus (r4r2) wrong shape
    with pytest.raises(ValueError):
        r4r2 = torch.tensor([1.0])
        disp.calculate(numbers, positions, charge, param, r4r2=r4r2)

    # Van-der-Waals radii (rvdw) wrong shape
    with pytest.raises(ValueError):
        rvdw = torch.tensor([1.0])
        disp.calculate(numbers, positions, charge, param, rvdw=rvdw)

    # atomic partial charges wrong shape
    with pytest.raises(ValueError):
        q = torch.tensor([1.0])
        disp.calculate(numbers, positions, charge, param, q=q)

    # wrong numbers (give charges, otherwise test fails in EEQ, not in disp)
    with pytest.raises(ValueError):
        q = torch.tensor([0.5, -0.5])
        nums = torch.tensor([1])
        disp.calculate(nums, positions, charge, param, q=q)


def test_fail_charges_not_required() -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    charge = torch.tensor(0.0)
    param = Param(s6=torch.tensor(1.0))

    disp = Disp()
    disp.register(
        TwoBodyTerm(damping_fn=RationalDamping(), charge_dependent=False)
    )

    with pytest.raises(RuntimeError):
        disp.calculate(
            numbers=numbers,
            positions=positions,
            charge=charge,
            param=param,
            q=torch.tensor([0.5, -0.5]),
        )


@pytest.mark.parametrize(
    "damping_fn",
    [RationalDamping(), OptimisedPowerDamping()],
)
def test_fail_damping_param(damping_fn: Damping) -> None:
    param = Param(s6=torch.tensor(1.00000000))
    disp = Disp()
    disp.register(TwoBodyTerm(damping_fn=damping_fn))

    with pytest.raises(TypeError):
        disp.calculate(
            numbers=torch.tensor([1, 1]),
            positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            charge=torch.tensor(0.0),
            param=param,
        )


##############################################################################


def test_fail_model() -> None:
    with pytest.raises(ValueError):
        DispD4(model="wrong")  # type: ignore


def test_fail_model_later() -> None:
    disp = Disp(model="d4s")
    disp._model_key = "wrong"  # type: ignore

    with pytest.raises(ValueError):
        disp.get_model(numbers=torch.tensor([1, 1]))


def test_terms() -> None:
    disp = Disp()
    assert len(disp.terms) == 0

    disp.register(TwoBodyTerm())
    assert len(disp.terms) == 1

    disp.deregister(TwoBodyTerm())
    assert len(disp.terms) == 0


def test_disp_model_input() -> None:
    numbers = torch.tensor([1, 1])
    model = D4Model(numbers)

    with pytest.warns(RuntimeWarning):
        Disp(model=model, model_kwargs={"damping_fn": "wrong"})


def test_disp_cn_input() -> None:
    # pylint: disable=import-outside-toplevel
    from tad_mctc.ncoord import cn_d3, cn_d4

    disp = Disp(model="d3", cn_fn=None)
    assert disp.cn_fn is cn_d3

    disp = Disp(model="d4s", cn_fn=None)
    assert disp.cn_fn is cn_d4

    disp = Disp(model="d5", cn_fn=None)
    assert disp.cn_fn is cn_d3
