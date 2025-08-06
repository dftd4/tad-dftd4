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
r"""
Dispersion: 3-body terms
========================

Implementation of the 3-body Axilrod-Teller-Muto dispersion terms.

.. math::

    E_\text{disp}^{(3), \text{ATM}} &=
    \sum_\text{ABC} E^{\text{ABC}} f_\text{damp}\left(\overline{R}_\text{ABC}\right) \\
    E^{\text{ABC}} &=
    \dfrac{C^{\text{ABC}}_9
    \left(3 \cos\theta_\text{A} \cos\theta_\text{B} \cos\theta_\text{C} + 1 \right)}
    {\left(r_\text{AB} r_\text{BC} r_\text{AC} \right)^3} \\
    f_\text{damp} &=
    \dfrac{1}{1+ 6 \left(\overline{R}_\text{ABC}\right)^{-16}}
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs, real_triples
from tad_mctc.convert import any_to_tensor
from tad_mctc.typing import DD, Tensor

from .. import defaults
from ..cutoff import Cutoff
from ..damping import Damping, Param, ZeroDamping
from ..model import ModelInst
from .base import DispTerm

__all__ = ["ATM", "get_atm_dispersion"]


def get_atm_dispersion(
    numbers: Tensor,
    positions: Tensor,
    c9: Tensor,
    radii: Tensor,
    cutoff: Tensor,
    damping_function: Damping = ZeroDamping(),
    s9: Tensor | float | int = defaults.S9,
    alp: Tensor | float | int = defaults.ALP,
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    c9 : Tensor
        Atomic C9 dispersion coefficients.
    radii : Tensor
        Pairwise critical radii for all atom pairs (shape: ``(..., nat, nat)``).
    cutoff : Tensor
        Real-space cutoff.
    damping_function : Damping, optional
        Damping function to use. Defaults to :func:`zero_damping`.
    s9 : Tensor, optional
        Scaling for dispersion coefficients. Defaults to ``1.0``.
    alp : Tensor, optional
        Exponent of zero damping function. Defaults to ``14.0``.

    Returns
    -------
    Tensor
        Atom-resolved ATM dispersion energy.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    s9 = any_to_tensor(s9, **dd)
    alp = any_to_tensor(alp, **dd)

    cutoff2 = cutoff * cutoff

    mask_pairs = real_pairs(numbers, mask_diagonal=True)
    mask_triples = real_triples(numbers, mask_diagonal=True, mask_self=True)

    # filler values for masks
    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)
    one = torch.tensor(1.0, **dd)

    r0ij = radii.unsqueeze(-1)
    r0ik = radii.unsqueeze(-2)
    r0jk = radii.unsqueeze(-3)
    r0 = r0ij * r0ik * r0jk

    # actually faster than other alternatives
    # very slow: (pos.unsqueeze(-2) - pos.unsqueeze(-3)).pow(2).sum(-1)
    distances = torch.pow(
        torch.where(
            mask_pairs,
            storch.cdist(positions, positions, p=2),
            eps,
        ),
        2.0,
    )

    r2ij = distances.unsqueeze(-1)
    r2ik = distances.unsqueeze(-2)
    r2jk = distances.unsqueeze(-3)
    r2 = r2ij * r2ik * r2jk
    r1 = torch.sqrt(r2)
    # add epsilon to avoid zero division later
    r3 = torch.where(mask_triples, r1 * r2, eps)
    r5 = torch.where(mask_triples, r2 * r3, eps)

    # to fix the previous mask, we mask again (not strictly necessary because
    # `ang` is also masked and we later multiply with `ang`)
    fdamp = torch.where(
        mask_triples,
        damping_function(
            r0,
            # dividing by tiny numbers leads to huge numbers, which result in
            # NaN's upon exponentiation in the subsequent step
            torch.where(mask_triples, r1, one),
            order=9,
            alp=alp / 3.0,
            only_damping=True,
        ),
        zero,
    )

    s = torch.where(
        mask_triples,
        (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik),
        zero,
    )

    ang = torch.where(
        mask_triples
        * (r2ij <= cutoff2)
        * (r2jk <= cutoff2)
        * (r2jk <= cutoff2),
        0.375 * s / r5 + 1.0 / r3,
        torch.tensor(0.0, **dd),
    )

    energy = ang * fdamp * s9 * c9
    return torch.sum(energy, dim=(-2, -1)) / 6.0


class ThreeBodyTerm(DispTerm):
    """Base class for three-body dispersion terms."""

    def __init__(
        self,
        *,
        damping_fn: Damping = ZeroDamping(),
        charge_dependent: bool = False,
    ):
        super().__init__(damping_fn, charge_dependent)


class ATM(ThreeBodyTerm, ABC):
    r"""
    D4's Axilrod-Teller-Muto dispersion term.
    - C9 coefficients are approximated from C6 coefficients
    - Becke--Johnson cutoff radii (`a1 * sqrt(3.0 * r4r2) + a2`)
    - zero damping

    .. math::

        E_\text{disp}^{(3), \text{ATM}} &=
        \sum_\text{ABC} E^{\text{ABC}} f_\text{damp}\left(\overline{R}_\text{ABC}\right) \\
        E^{\text{ABC}} &=
        \dfrac{C^{\text{ABC}}_9
        \left(3 \cos\theta_\text{A} \cos\theta_\text{B} \cos\theta_\text{C} + 1 \right)}
        {\left(r_\text{AB} r_\text{BC} r_\text{AC} \right)^3} \\
        f_\text{damp} &=
        \dfrac{1}{1+ 6 \left(\overline{R}_\text{ABC}\right)^{-16}}
    """

    @abstractmethod
    def get_c9(self, model: ModelInst, cn: Tensor, q: Tensor | None) -> Tensor:
        """Approximate or exact C9 coefficients."""

    @abstractmethod
    def get_radii(
        self,
        param: Param,
        r4r2: Tensor,
        rvdw: Tensor,
    ) -> Tensor:
        """Compute critical radii used in damping function."""

    def calculate(
        self,
        numbers: Tensor,
        positions: Tensor,
        param: Param,
        cn: Tensor,
        model: ModelInst,
        q: Tensor | None,
        r4r2: Tensor,
        rvdw: Tensor,
        cutoff: Cutoff,
    ):
        # ATM-specific C9 coefficients and radii
        c9 = self.get_c9(model, cn, q)
        radii = self.get_radii(param, r4r2, rvdw)

        return get_atm_dispersion(
            numbers,
            positions,
            c9,
            radii,
            damping_function=self.damping_fn,
            cutoff=cutoff.disp3,
            s9=param.get("s9", torch.tensor(defaults.S9, **self.dd)),
            alp=param.get("alp", torch.tensor(defaults.ALP, **self.dd)),
        )


# Radii mixins


class RadiiBJMixin:
    """Becke--Johnson critical radii: a1 * sqrt(3 * r4r2_i * r4r2_j) + a2."""

    def get_radii(
        self,
        param: Param,
        r4r2: Tensor,
        rvdw: Tensor,
    ) -> Tensor:
        dd = self.dd  # type: ignore

        a1 = param.get("a1", torch.tensor(defaults.A1, **dd))
        a2 = param.get("a2", torch.tensor(defaults.A2, **dd))
        return (
            a1 * storch.sqrt(3.0 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)) + a2
        )


class RadiiVDWMixin:
    """Scaled pair-wise VDW radii: rs9 * rvdw."""

    def get_radii(
        self,
        param: Param,
        r4r2: Tensor,
        rvdw: Tensor,
    ) -> Tensor:
        dd = self.dd  # type: ignore
        rs9 = param.get("rs9", torch.tensor(defaults.RS9, **dd))
        return rs9 * rvdw


# C9 mixins


class C9ExactMixin:
    """Exact C9 via Casimir–Polder integration of polarizabilities."""

    charge_dependent: bool
    """Whether the C9 coefficients depend on atomic charges."""

    def get_c9(self, model: ModelInst, cn: Tensor, q: Tensor | None) -> Tensor:
        r"""
        Approximate C9 coefficients from C6 coefficients.

        .. math::

            C_9 = \sqrt{|C_{6}^{AB} C_{6}^{AC} C_{6}^{BC}|}
        """
        # pylint: disable=import-outside-toplevel
        from ..utils import trapzd_atm

        weights = model.weight_references(
            cn, q if self.charge_dependent else None
        )
        aiw = model.get_weighted_pols(weights)

        # C9_ABC = integral (aiw_A * aiw_B * aiw_C)
        aiwi = aiw.unsqueeze(-2).unsqueeze(-2)
        aiwj = aiw.unsqueeze(-3).unsqueeze(-2)
        aiwk = aiw.unsqueeze(-3).unsqueeze(-3)
        return trapzd_atm(aiwi * aiwj * aiwk)


class C9ApproxMixin:
    """Approximate C9 coefficients from C6 coefficients."""

    charge_dependent: bool
    """Whether the C9 coefficients depend on atomic charges."""

    def get_c9(self, model: ModelInst, cn: Tensor, q: Tensor | None) -> Tensor:

        weights = model.weight_references(
            cn, q if self.charge_dependent else None
        )
        c6 = model.get_atomic_c6(weights)

        # C9_ABC = sqrt(|C6_AB * C6_AC * C6_BC|)
        return storch.sqrt(
            torch.abs(c6.unsqueeze(-1) * c6.unsqueeze(-2) * c6.unsqueeze(-3)),
        )
