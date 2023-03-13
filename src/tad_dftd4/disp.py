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
Dispersion energy
=================

This module provides the dispersion energy evaluation for the pairwise
interactions. It contains the main entrypoint for the dispersion energy
(`dftd4`) as well as wrappers for the two-body (`dispersion2`) and the
three-body (`dispersion3`) dispersion energy.
"""
from __future__ import annotations

import torch

from . import data, defaults
from ._typing import Any, CountingFunction, DampingFunction, Tensor
from .charges import get_charges
from .cutoff import Cutoff
from .damping import get_atm_dispersion, rational_damping
from .model import D4Model
from .ncoord import erf_count, get_coordination_number_d4
from .utils import real_pairs


def dftd4(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    param: dict[str, Tensor],
    *,
    model: D4Model | None = None,
    rcov: Tensor | None = None,
    r4r2: Tensor | None = None,
    q: Tensor | None = None,
    cutoff: Cutoff | None = None,
    counting_function: CountingFunction = erf_count,
    damping_function: DampingFunction = rational_damping,
) -> Tensor:
    """
    Evaluate DFT-D4 dispersion energy for a (batch of) molecule(s).

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system (batch, natoms, 3).
    charge : Tensor
        Total charge of the system.
    param : dict[str, Tensor]
        DFT-D4 damping parameters.
    model : D4Model | None, optional
        The DFT-D4 dispersion model for the evaluation of the C6 coefficients.
        Defaults to `None`.
    rcov : Tensor | None, optional
        Covalent radii of the atoms in the system. Defaults to
        `None`, i.e., default values are used.
    r4r2 : Tensor | None, optional
        r⁴ over r² expectation values of the atoms in the system. Defaults to
        `None`, i.e., default values are used.
    q : Tensor | None, optional
        Atomic partial charges. Defaults to `None`, i.e., EEQ charges are
        calculated using the total `charge`.
    cutoff : Cutoff | None, optional
        Collection of real-space cutoffs. Defaults to `None`, i.e., `Cutoff` is
        initialized with its defaults.
    counting_function : CountingFunction, optional
        Counting function used for the DFT-D4 coordination number. Defaults to
        the error function counting function `erf_count`.
    damping_function : DampingFunction, optional
        Damping function to evaluate distance dependent contributions. Defaults
        to the Becke-Johnson rational damping function `rational_damping`.

    Returns
    -------
    Tensor
        Atom-resolved DFT-D4 dispersion energy.

    Raises
    ------
    ValueError
        Shape inconsistencies between `numbers`, `positions`, `r4r2`, or,
        `rcov`.
    """
    if model is None:
        model = D4Model(numbers, device=positions.device, dtype=positions.dtype)
    if cutoff is None:
        cutoff = Cutoff(device=positions.device, dtype=positions.dtype)

    if rcov is None:
        rcov = data.cov_rad_d3[numbers].type(positions.dtype).to(positions.device)
    if r4r2 is None:
        r4r2 = data.r4r2[numbers].type(positions.dtype).to(positions.device)
    if q is None:
        q = get_charges(numbers, positions, charge, cutoff=cutoff.cn_eeq)

    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape}) is not consistent "
            f"with atomic numbers ({numbers.shape}).",
        )
    if numbers.shape != r4r2.shape:
        raise ValueError(
            f"Shape of expectation values r4r2 ({r4r2.shape}) is not "
            f"consistent with atomic numbers ({numbers.shape}).",
        )
    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii ({rcov.shape}) is not consistent with "
            f"atomic numbers ({numbers.shape}).",
        )
    if numbers.shape != q.shape:
        raise ValueError(
            f"Shape of atomic charges ({q.shape}) is not consistent with "
            f"atomic numbers ({numbers.shape}).",
        )

    cn = get_coordination_number_d4(
        numbers, positions, counting_function, rcov, cutoff=cutoff.cn
    )
    weights = model.weight_references(cn, q)
    c6 = model.get_atomic_c6(weights)

    energy = dispersion2(
        numbers,
        positions,
        param,
        c6,
        r4r2=r4r2,
        damping_function=damping_function,
        cutoff=cutoff.disp2,
    )

    # three-body dispersion
    if "s9" in param and param["s9"] != 0.0:
        weights = model.weight_references(cn, q=None)
        c6 = model.get_atomic_c6(weights)

        energy += dispersion3(numbers, positions, param, c6, cutoff.disp3)

    return energy


def dispersion2(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    c6: Tensor,
    r4r2: Tensor,
    damping_function: DampingFunction = rational_damping,
    cutoff: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """
    Calculate dispersion energy between pairs of atoms.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system (batch, natoms, 3).
    param : dict[str, Tensor]
        DFT-D3 damping parameters.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    r4r2 : Tensor
        r⁴ over r² expectation values of the atoms in the system.
    damping_function : DampingFunction, optional
        Damping function evaluate distance dependent contributions.
        Additional arguments are passed through to the function.
        Defaults to `rational_damping`.
    cutoff : Tensor | None, optional
        Real-space cutoff for two-body dispersion. Defaults to `None`, which
        will be evaluated to `defaults.D4_DISP2_CUTOFF`.

    Returns
    -------
    Tensor
        Atom-resolved two-body dispersion energy.
    """
    if cutoff is None:
        cutoff = positions.new_tensor(defaults.D4_DISP2_CUTOFF)

    mask = real_pairs(numbers, diagonal=False)
    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"),
        positions.new_tensor(torch.finfo(positions.dtype).eps),
    )

    qq = 3 * r4r2.unsqueeze(-1) * r4r2.unsqueeze(-2)
    c8 = c6 * qq

    t6 = torch.where(
        mask * (distances <= cutoff),
        damping_function(6, distances, qq, param, **kwargs),
        positions.new_tensor(0.0),
    )
    t8 = torch.where(
        mask * (distances <= cutoff),
        damping_function(8, distances, qq, param, **kwargs),
        positions.new_tensor(0.0),
    )

    e6 = torch.sum(c6 * t6, dim=-1)
    e8 = torch.sum(c8 * t8, dim=-1)

    s6 = param.get("s6", positions.new_tensor(defaults.S6))
    s8 = param.get("s8", positions.new_tensor(defaults.S8))

    edisp = s6 * e6 + s8 * e8

    if "s10" in param and param["s10"] != 0.0:
        c10 = c6 * torch.pow(qq, 2) * 49.0 / 40.0
        t10 = torch.where(
            mask * (distances <= cutoff),
            damping_function(10, distances, qq, param, **kwargs),
            positions.new_tensor(0.0),
        )
        e10 = torch.sum(c10 * t10, dim=-1)
        edisp += param["s10"] * e10

    return -0.5 * edisp


def dispersion3(
    numbers: Tensor,
    positions: Tensor,
    param: dict[str, Tensor],
    c6: Tensor,
    cutoff: Tensor | None = None,
) -> Tensor:
    """
    Three-body dispersion term. Currently this is only a wrapper for the
    Axilrod-Teller-Muto dispersion term.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the atoms in the system.
    positions : Tensor
        Cartesian coordinates of the atoms in the system (batch, natoms, 3).
    param : dict[str, Tensor]
        Dictionary of dispersion parameters. Default values are used for
        missing keys.
    c6 : Tensor
        Atomic C6 dispersion coefficients.
    cutoff : Tensor | None
        Real-space cutoff. Defaults to `None`, i.e, `defaults.D4_DISP3_CUTOFF`.

    Returns
    -------
    Tensor
        Atom-resolved three-body dispersion energy.
    """
    if cutoff is None:
        cutoff = positions.new_tensor(defaults.D4_DISP3_CUTOFF)

    s9 = param.get("s9", positions.new_tensor(defaults.S9))
    a1 = param.get("a1", positions.new_tensor(defaults.A1))
    a2 = param.get("a2", positions.new_tensor(defaults.A2))
    alp = param.get("alp", positions.new_tensor(defaults.ALP))

    return get_atm_dispersion(numbers, positions, cutoff, c6, s9, a1, a2, alp)
