# This file is part of tad-dftd4.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2022 Marvin Friede
#
# tad-dftd4 is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad-dftd4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-dftd4. If not, see <https://www.gnu.org/licenses/>.
r"""
Axilrod-Teller-Muto (ATM) dispersion term
=========================================

This module provides the dispersion energy evaluation for the three-body
Axilrod-Teller-Muto dispersion term.

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

import torch

from .. import defaults
from .._typing import Tensor
from ..data import r4r2
from ..utils import real_pairs, real_triples


def get_atm_dispersion(
    numbers: Tensor,
    positions: Tensor,
    cutoff: Tensor,
    c6: Tensor,
    s9: Tensor = torch.tensor(defaults.S9),
    a1: Tensor = torch.tensor(defaults.A1),
    a2: Tensor = torch.tensor(defaults.A2),
    alp: Tensor = torch.tensor(defaults.ALP),
) -> Tensor:
    """
    Axilrod-Teller-Muto dispersion term.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system.
    cutoff : Tensor
        Real-space cutoff.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    s9 : Tensor, optional
        Scaling for dispersion coefficients. Defaults to `1.0`.
    a1 : Tensor, optional
        Scaling for the C8 / C6 ratio in the critical radius within the
        Becke-Johnson damping function.
    a2 : Tensor, optional
        Offset parameter for the critical radius within the Becke-Johnson
        damping function.
    alp : Tensor, optional
        Exponent of zero damping function. Defaults to `14.0`.

    Returns
    -------
    Tensor
        Atom-resolved ATM dispersion energy.
    """
    dd = {"device": positions.device, "dtype": positions.dtype}

    s9 = s9.type(positions.dtype).to(positions.device)
    alp = alp.type(positions.dtype).to(positions.device)

    cutoff2 = cutoff * cutoff

    # C9_ABC = s9 * sqrt(|C6_AB * C6_AC * C6_BC|)
    c9 = s9 * torch.sqrt(
        torch.abs(c6.unsqueeze(-1) * c6.unsqueeze(-2) * c6.unsqueeze(-3))
    )

    temp = (
        a1 * torch.sqrt(3.0 * r4r2[numbers].unsqueeze(-1) * r4r2[numbers].unsqueeze(-2))
        + a2
    )
    r0ij = temp.unsqueeze(-1)
    r0ik = temp.unsqueeze(-2)
    r0jk = temp.unsqueeze(-3)
    r0 = r0ij * r0ik * r0jk

    # actually faster than other alternatives
    # very slow: (pos.unsqueeze(-2) - pos.unsqueeze(-3)).pow(2).sum(-1)
    distances = torch.pow(
        torch.where(
            real_pairs(numbers, diagonal=False),
            torch.cdist(
                positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"
            ),
            torch.tensor(torch.finfo(positions.dtype).eps, **dd),
        ),
        2.0,
    )

    r2ij = distances.unsqueeze(-1)
    r2ik = distances.unsqueeze(-2)
    r2jk = distances.unsqueeze(-3)
    r2 = r2ij * r2ik * r2jk
    r1 = torch.sqrt(r2)
    r3 = r1 * r2
    r5 = r2 * r3

    fdamp = 1.0 / (1.0 + 6.0 * (r0 / r1) ** (alp / 3.0))

    s = (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik)
    ang = torch.where(
        real_triples(numbers, diagonal=False)
        * (r2ij <= cutoff2)
        * (r2jk <= cutoff2)
        * (r2jk <= cutoff2),
        0.375 * s / r5 + 1.0 / r3,
        torch.tensor(0.0, **dd),
    )

    energy = ang * fdamp * c9
    return torch.sum(energy, dim=(-2, -1)) / 6.0
