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
Rational (Becke-Johnson) damping function
=========================================

This module defines the rational damping function, also known as Becke-Johnson
damping.

.. math::

    f^n_{\text{damp}}\left(R_0^{\text{AB}}\right) =
    \dfrac{R^n_{\text{AB}}}{R^n_{\text{AB}} +
    \left( a_1 R_0^{\text{AB}} + a_2 \right)^n}
"""
from __future__ import annotations

import torch

from .. import defaults
from .._typing import Tensor


def rational_damping(
    order: int,
    distances: Tensor,
    qq: Tensor,
    param: dict[str, Tensor],
) -> Tensor:
    """
    Rational damped dispersion interaction between pairs.

    Parameters
    ----------
    order : int
        Order of the dispersion interaction, e.g.
        6 for dipole-dipole, 8 for dipole-quadrupole and so on.
    distances : Tensor
        Pairwise distances between atoms in the system.
    qq : Tensor
        Quotient of C8 and C6 dispersion coefficients.
    param : dict[str, Tensor]
        DFT-D4 damping parameters.

    Returns
    -------
    Tensor
        Values of the damping function.
    """

    a1 = param.get("a1", distances.new_tensor(defaults.A1))
    a2 = param.get("a2", distances.new_tensor(defaults.A2))
    return 1.0 / (distances.pow(order) + (a1 * torch.sqrt(qq) + a2).pow(order))
