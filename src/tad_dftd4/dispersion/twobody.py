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
Dispersion: 2-body terms
==========================

Implementation of the 2-body dispersion terms.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs
from tad_mctc.typing import DD, Tensor

from .. import defaults
from ..cutoff import Cutoff
from ..damping import Damping, Param, RationalDamping
from ..model import D4Model, D4SModel
from .base import DispTerm


class TwoBodyTerm(DispTerm):
    """Base class for two-body dispersion terms."""

    def __init__(
        self,
        *,
        damping_fn: Damping = RationalDamping(),
        charge_dependent: bool = True,
    ):
        super().__init__(damping_fn, charge_dependent)
        self.damping_fn = damping_fn

    def compute(
        self,
        numbers: Tensor,
        positions: Tensor,
        param: Param,
        cn: Tensor,
        model: D4Model | D4SModel,
        q: Tensor | None,
        r4r2: Tensor,
        rvdw: Tensor,
        cutoff: Cutoff,
    ) -> Tensor:
        cutoff_val = getattr(cutoff, "disp2")

        weights = model.weight_references(
            cn, q if self.charge_dependent else None
        )
        c6 = model.get_atomic_c6(weights)

        return dispersion2(
            numbers,
            positions,
            param,
            c6,
            r4r2,
            rvdw,
            self.damping_fn,
            cutoff_val,
        )


def dispersion2(
    numbers: Tensor,
    positions: Tensor,
    param: Param,
    c6: Tensor,
    r4r2: Tensor,
    rvdw: Tensor | None = None,
    damping_function: Damping = RationalDamping(),
    cutoff: Tensor | None = None,
    as_matrix: bool = False,
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    param : Param
        DFT-D3 damping parameters.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    r4r2 : Tensor
        r⁴ over r² expectation values of the atoms in the system.
    rvdw : Tensor
        Pair-wise van der Waals radii of the atoms in the system.
    damping_function : Damping, optional
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.
        Defaults to :class:`.RationalDamping`.
    cutoff : Tensor | None, optional
        Real-space cutoff for two-body dispersion. Defaults to `None`, which
        will be evaluated to `defaults.D4_DISP2_CUTOFF`.
    as_matrix : bool, optional
        Return dispersion energy as a matrix. If you sum up the dispersion
        energy from the matrix, do not forget the factor `0.5` that fixes the
        double counting. Defaults to `False`.

    Returns
    -------
    Tensor
        Atom-resolved two-body dispersion energy.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    zero = torch.tensor(0.0, **dd)

    if cutoff is None:
        cutoff = torch.tensor(defaults.D4_DISP2_CUTOFF, **dd)

    mask = real_pairs(numbers, mask_diagonal=True)
    distances = torch.where(
        mask,
        storch.cdist(positions, positions, p=2),
        torch.tensor(torch.finfo(positions.dtype).eps, **dd),
    )

    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    if damping_function.radius_type == "rvdw":
        if rvdw is None:
            raise ValueError("`rvdw` must be provided for `rvdw` radius type.")
        radii = rvdw
    elif damping_function.radius_type == "r4r2":
        radii = qq
    else:
        raise ValueError(f"Unknown radius type: {damping_function.radius_type}")

    t6 = torch.where(
        mask * (distances <= cutoff),
        damping_function(distances, radii, 6, **param),
        zero,
    )
    t8 = torch.where(
        mask * (distances <= cutoff),
        damping_function(distances, radii, 8, **param),
        zero,
    )

    if as_matrix is True:
        e6 = c6 * t6
        e8 = c8 * t8
    else:
        e6 = torch.sum(c6 * t6, dim=-1)
        e8 = torch.sum(c8 * t8, dim=-1)

    s6 = param.get("s6", torch.tensor(defaults.S6, **dd))
    s8 = param.get("s8", torch.tensor(defaults.S8, **dd))

    edisp = s6 * e6 + s8 * e8

    # With `if "s10" in param and param["s10"] != 0.0`, the gradcheck tests fail
    # if s10 is exactly 0 (other values are fine).
    if "s10" in param:
        c10 = c6 * torch.pow(qq, 2) * 49.0 / 40.0
        t10 = torch.where(
            mask * (distances <= cutoff),
            damping_function(distances, radii, 10, **param),
            zero,
        )

        if as_matrix is True:
            e10 = c10 * t10
        else:
            e10 = torch.sum(c10 * t10, dim=-1)

        edisp += param["s10"] * e10

    if as_matrix is True:
        return -edisp
    return -0.5 * edisp
