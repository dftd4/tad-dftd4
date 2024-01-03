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
from tad_mctc import storch
from tad_mctc.batch import real_pairs, real_triples

from .. import data, defaults
from ..typing import DD, Tensor

__all__ = ["get_atm_dispersion"]


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
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    s9 = s9.to(**dd)
    alp = alp.to(**dd)

    cutoff2 = cutoff * cutoff

    mask_pairs = real_pairs(numbers, mask_diagonal=True)
    mask_triples = real_triples(numbers, mask_diagonal=True, mask_self=True)

    # filler values for masks
    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)
    one = torch.tensor(1.0, **dd)

    # C9_ABC = s9 * sqrt(|C6_AB * C6_AC * C6_BC|)
    c9 = s9 * storch.sqrt(
        torch.abs(c6.unsqueeze(-1) * c6.unsqueeze(-2) * c6.unsqueeze(-3)),
    )

    rad = data.R4R2[numbers]
    radii = rad.unsqueeze(-1) * rad.unsqueeze(-2)
    temp = a1 * storch.sqrt(3.0 * radii) + a2

    r0ij = temp.unsqueeze(-1)
    r0ik = temp.unsqueeze(-2)
    r0jk = temp.unsqueeze(-3)
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

    # dividing by tiny numbers leads to huge numbers, which result in NaN's
    # upon exponentiation in the subsequent step
    base = r0 / torch.where(mask_triples, r1, one)

    # to fix the previous mask, we mask again (not strictly necessary because
    # `ang` is also masked and we later multiply with `ang`)
    fdamp = torch.where(
        mask_triples,
        1.0 / (1.0 + 6.0 * base ** (alp / 3.0)),
        zero,
    )

    s = torch.where(
        mask_triples,
        (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik),
        zero,
    )

    ang = torch.where(
        mask_triples * (r2ij <= cutoff2) * (r2jk <= cutoff2) * (r2jk <= cutoff2),
        0.375 * s / r5 + 1.0 / r3,
        torch.tensor(0.0, **dd),
    )

    energy = ang * fdamp * c9
    return torch.sum(energy, dim=(-2, -1)) / 6.0
