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
from __future__ import annotations

import torch

from tad_dftd4.cutoff import Cutoff
from tad_dftd4.damping import Param
from tad_dftd4.damping.functions import OptimisedPowerDamping, RationalDamping
from tad_dftd4.dispersion import DispTerm
from tad_dftd4.model import ModelInst


class DummyDispTerm(DispTerm):
    """A dummy dispersion term for testing."""

    def calculate(
        self,
        numbers: torch.Tensor,
        positions: torch.Tensor,
        param: Param,
        cn: torch.Tensor,
        model: ModelInst,
        q: torch.Tensor | None,
        r4r2: torch.Tensor,
        rvdw: torch.Tensor,
        cutoff: Cutoff,
    ) -> torch.Tensor:
        return torch.tensor(0.0)


def test_eq_equal() -> None:
    """Test that two identical DispTerm instances are equal."""
    term1 = DummyDispTerm(damping_fn=RationalDamping(), charge_dependent=True)
    term2 = DummyDispTerm(damping_fn=RationalDamping(), charge_dependent=True)
    assert term1 == term2


def test_eq_different_damping() -> None:
    """Test that instances with different damping functions are not equal."""
    term1 = DummyDispTerm(
        damping_fn=RationalDamping(),
        charge_dependent=True,
    )
    term2 = DummyDispTerm(
        damping_fn=OptimisedPowerDamping(),
        charge_dependent=True,
    )
    assert term1 != term2


def test_eq_different_charge_dependence() -> None:
    """Test that instances with different charge dependence are not equal."""
    term1 = DummyDispTerm(damping_fn=RationalDamping(), charge_dependent=True)
    term2 = DummyDispTerm(damping_fn=RationalDamping(), charge_dependent=False)
    assert term1 != term2


def test_eq_different_type() -> None:
    """Test that a DispTerm instance is not equal to an object of a different type."""
    term1 = DummyDispTerm(damping_fn=RationalDamping(), charge_dependent=True)
    assert term1 != "not a disp term"


def test_eq_different_subclass() -> None:
    """Test that instances of different DispTerm subclasses are not equal."""

    class AnotherDummyDispTerm(DispTerm):
        def calculate(
            self,
            numbers: torch.Tensor,
            positions: torch.Tensor,
            param: Param,
            cn: torch.Tensor,
            model: ModelInst,
            q: torch.Tensor | None,
            r4r2: torch.Tensor,
            rvdw: torch.Tensor,
            cutoff: Cutoff,
        ) -> torch.Tensor:
            return torch.tensor(1.0)

    term1 = DummyDispTerm(
        damping_fn=RationalDamping(),
        charge_dependent=True,
    )
    term2 = AnotherDummyDispTerm(
        damping_fn=RationalDamping(),
        charge_dependent=True,
    )
    assert term1 != term2
