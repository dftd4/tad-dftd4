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
"""
Coordination number: D4
=======================

Calculation of D4 coordination number. Includes electronegativity-
dependent term.
"""
from __future__ import annotations

import torch

from .. import defaults
from .._typing import Any, CountingFunction, Tensor
from ..data import cov_rad_d3, pauling_en
from ..utils import real_pairs
from .count import erf_count

__all__ = ["get_coordination_number_d4"]


def get_coordination_number_d4(
    numbers: Tensor,
    positions: Tensor,
    counting_function: CountingFunction = erf_count,
    rcov: Tensor | None = None,
    en: Tensor | None = None,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Compute fractional coordination number using an exponential counting function.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system (batch, natoms, 3).
    counting_function : CountingFunction
        Calculate weight for pairs. Defaults to `erf_count`.
    rcov : Tensor | None, optional
        Covalent radii for each species. Defaults to `None`.
    en : Tensor | None, optional
        Electronegativities for all atoms. Defaults to `None`.
    cutoff : Tensor | None, optional
        Real-space cutoff. Defaults to `None`.
    kwargs : dict[str, Any]
        Pass-through arguments for counting function.

    Returns
    -------
    Tensor
        Coordination numbers for all atoms.

    Raises
    ------
    ValueError
        If shape mismatch between `numbers`, `positions` and `rcov` is detected.
    """

    if cutoff is None:
        cutoff = positions.new_tensor(defaults.D4_CN_CUTOFF)

    if rcov is None:
        rcov = cov_rad_d3[numbers]
    rcov = rcov.type(positions.dtype).to(positions.device)

    if en is None:
        en = pauling_en[numbers]
    en = en.type(positions.dtype).to(positions.device)

    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii {rcov.shape} is not consistent with "
            f"({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )

    mask = real_pairs(numbers)

    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"),
        positions.new_tensor(torch.finfo(positions.dtype).eps),
    )

    # Eq. 6
    endiff = torch.abs(en.unsqueeze(-2) - en.unsqueeze(-1))
    den = defaults.D4_K4 * torch.exp(
        -((endiff + defaults.D4_K5) ** 2.0) / defaults.D4_K6
    )

    rc = rcov.unsqueeze(-2) + rcov.unsqueeze(-1)
    cf = torch.where(
        mask * (distances <= cutoff),
        den * counting_function(distances, rc, **kwargs),
        positions.new_tensor(0.0),
    )
    return torch.sum(cf, dim=-1)
