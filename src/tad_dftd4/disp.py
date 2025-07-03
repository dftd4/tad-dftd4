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
Dispersion energy
=================

This module provides the dispersion energy evaluation for the pairwise
interactions. It contains the main entrypoint for the dispersion energy
(:func:`.dftd4`) as well as wrappers for the two-body
(:func:`.dispersion2`) and the three-body
(:func:`.dispersion3`) dispersion energy.
"""
from __future__ import annotations

import torch
from tad_mctc.typing import DD, CountingFunction, Tensor
from tad_multicharge import get_eeq_charges

from . import data
from .cutoff import Cutoff
from .damping import Damping, Param, RationalDamping
from .dispersion.threebody import dispersion3
from .dispersion.twobody import dispersion2
from .model import D4Model, D4SModel
from .ncoord import cn_d4, erf_count

__all__ = ["dftd4", "dispersion3", "get_properties"]


def dftd4(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    param: Param,
    *,
    model: D4Model | D4SModel | None = None,
    rcov: Tensor | None = None,
    r4r2: Tensor | None = None,
    rvdw: Tensor | None = None,
    q: Tensor | None = None,
    cutoff: Cutoff | None = None,
    counting_function: CountingFunction = erf_count,
    damping_function: Damping = RationalDamping(),
) -> Tensor:
    """
    Evaluate DFT-D4 dispersion energy for a (batch of) molecule(s).

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    charge : Tensor
        Total charge of the system.
    param : Param
        DFT-D4 damping parameters.
    model : D4Model | D4SModel | None, optional
        The DFT-D4 dispersion model for the evaluation of the C6 coefficients.
        Defaults to ``None``, which creates :class:`tad_dftd4.model.d4.D4Model`.
    rcov : Tensor | None, optional
        Covalent radii of the atoms in the system. Defaults to
        ``None``, i.e., default values are used.
    r4r2 : Tensor | None, optional
        r⁴ over r² expectation values of the atoms in the system. Defaults to
        ``None``, i.e., default values are used.
    rvdw : Tensor | None, optional
        Pairwise van der Waals radii of the atoms in the system. Defaults to
        ``None``, i.e., default values are used.
    q : Tensor | None, optional
        Atomic partial charges. Defaults to ``None``, i.e., EEQ charges are
        calculated using the total ``charge``.
    cutoff : Cutoff | None, optional
        Collection of real-space cutoffs. Defaults to ``None``, i.e.,
        :class:`tad_dftd4.cutoff.Cutoff` is initialized with its defaults.
    counting_function : CountingFunction, optional
        Counting function used for the DFT-D4 coordination number. Defaults to
        the error function counting function
        :func:`tad_mctc.ncoord.count.erf_count`.
    damping_function : DampingFunction, optional
        Damping function to evaluate distance dependent contributions. Defaults
        to the Becke-Johnson rational damping function
        :func:`tad_dftd4.damping.rational.rational_damping`.

    Returns
    -------
    Tensor
        Atom-resolved DFT-D4 dispersion energy.

    Raises
    ------
    ValueError
        Shape inconsistencies between ``numbers``, ``positions``, ``r4r2``,
        or, ``rcov``.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape}) is not consistent "
            f"with atomic numbers ({numbers.shape}).",
        )

    if model is None:
        model = D4Model(numbers, **dd)
    if cutoff is None:
        cutoff = Cutoff(**dd)

    if r4r2 is None:
        r4r2 = data.R4R2(**dd)[numbers]
    if numbers.shape != r4r2.shape:
        raise ValueError(
            f"Shape of expectation values r4r2 ({r4r2.shape}) is not "
            f"consistent with atomic numbers ({numbers.shape}).",
        )

    if rcov is None:
        rcov = data.COV_D3(**dd)[numbers]
    if numbers.shape != rcov.shape:
        raise ValueError(
            f"Shape of covalent radii ({rcov.shape}) is not consistent with "
            f"atomic numbers ({numbers.shape}).",
        )

    if rvdw is None:
        rvdw = data.VDW_PAIRWISE(**dd)[
            numbers.unsqueeze(-1), numbers.unsqueeze(-2)
        ]
    if numbers.shape != rvdw.shape[:-1]:
        raise ValueError(
            f"Shape of van der Waals radii ({rvdw.shape}) is not consistent "
            f"with atomic numbers ({numbers.shape}).",
        )

    # No charges required for ATM only (e.g. for GFN2 non-sc part)
    s6_nonzero = "s6" in param and param["s6"] != 0.0
    s8_nonzero = "s8" in param and param["s8"] != 0.0
    if q is None and (s6_nonzero or s8_nonzero):
        q = get_eeq_charges(numbers, positions, charge, cutoff=cutoff.cn_eeq)

    if q is not None:
        if numbers.shape != q.shape:
            raise ValueError(
                f"Shape of atomic charges ({q.shape}) is not consistent with "
                f"atomic numbers ({numbers.shape}).",
            )

    energy = torch.zeros(numbers.shape, **dd)

    cn = cn_d4(
        numbers,
        positions,
        counting_function=counting_function,
        rcov=rcov,
        cutoff=cutoff.cn,
    )

    weights = model.weight_references(cn, q)
    c6 = model.get_atomic_c6(weights)

    if s6_nonzero or s8_nonzero:
        energy += dispersion2(
            numbers,
            positions,
            param,
            c6,
            r4r2=r4r2,
            rvdw=rvdw,
            damping_function=damping_function,
            cutoff=cutoff.disp2,
        )

    # three-body dispersion
    if "s9" in param and param["s9"] != 0.0:
        weights = model.weight_references(cn, q=None)
        c6 = model.get_atomic_c6(weights)

        energy += dispersion3(numbers, positions, param, c6, r4r2, cutoff.disp3)

    return energy


def get_properties(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor | None = None,
    cutoff: Cutoff | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Wrapper to evaluate properties related to this dispersion model.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    charge : Tensor | None
        Total charge of the system. Defaults to ``None``, i.e., ``0``.
    cutoff : Cutoff | None
        Real-space cutoff. Defaults to ``None``, i.e., the defaults for
        all cutoffs are used.

    Returns
    -------
    (Tensor, Tensor, Tensor, Tensor)
        Properties related to the dispersion model:
        - DFT-D4 coordination number
        - Atomic partial charges
        - Atom-resolved C6 dispersion coefficients
        - Static polarizabilities
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if cutoff is None:
        cutoff = Cutoff(**dd)

    if charge is None:
        charge = torch.tensor(0.0, **dd)

    cn = cn_d4(numbers, positions, cutoff=cutoff.cn)
    q = get_eeq_charges(numbers, positions, charge, cutoff=cutoff.cn_eeq)

    model = D4Model(numbers, **dd)
    weights = model.weight_references(cn, q)
    c6 = model.get_atomic_c6(weights)
    alpha = model.get_polarizabilities(weights)

    return cn, q, c6, alpha
