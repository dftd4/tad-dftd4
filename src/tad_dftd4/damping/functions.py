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
Damping: Functions
==================

Collections of damping functions including:
- Rational damping (Becke--Johnson)
- Zero damping (Chai--Head-Gordon)
- Modified zero damping
- Optimised power damping
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tad_mctc import storch

from .. import defaults
from ..typing import Literal, Tensor

__all__ = [
    "Damping",
    "RationalDamping",
    "ZeroDamping",
    "MZeroDamping",
    "OptimisedPowerDamping",
]


class Damping(ABC):
    """
    Base interface for damping functions.

    This class defines the interface for damping functions used in DFT-D4
    calculations. It provides a callable interface that takes the order of the
    dispersion interaction, pairwise distances, radii, and parameters as input
    and returns the damping function values.
    """

    radius_type: Literal["r4r2", "rvdw"]
    """Type of radius used in the damping function."""

    def __call__(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        only_damping: bool = False,
        *,
        a1: Tensor | float | int = defaults.A1,
        a2: Tensor | float | int = defaults.A2,
        s6: Tensor | float | int = defaults.S6,
        rs6: Tensor | float | int = defaults.RS6,
        s8: Tensor | float | int = defaults.S8,
        rs8: Tensor | float | int = defaults.RS8,
        s9: Tensor | float | int = defaults.S9,
        rs9: Tensor | float | int = defaults.RS9,
        s10: Tensor | float | int = defaults.S10,
        alp: Tensor | float | int = defaults.ALP,
        bet: Tensor | float | int = defaults.BET,
        doi: str | None = None,
    ) -> Tensor:
        return self._f(
            distances,
            radii,
            order,
            a1=a1,
            a2=a2,
            s6=s6,
            rs6=rs6,
            s8=s8,
            rs8=rs8,
            s9=s9,
            rs9=rs9,
            s10=s10,
            alp=alp,
            bet=bet,
            only_damping=only_damping,
        )

    @abstractmethod
    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int = defaults.A1,
        a2: Tensor | float | int = defaults.A2,
        s6: Tensor | float | int = defaults.S6,
        rs6: Tensor | float | int = defaults.RS6,
        s8: Tensor | float | int = defaults.S8,
        rs8: Tensor | float | int = defaults.RS8,
        s9: Tensor | float | int = defaults.S9,
        rs9: Tensor | float | int = defaults.RS9,
        s10: Tensor | float | int = defaults.S10,
        alp: Tensor | float | int = defaults.ALP,
        bet: Tensor | float | int = defaults.BET,
        only_damping: bool = False,
    ) -> Tensor: ...


# concrete damping models


class RationalDamping(Damping):
    r"""
    Becke--Johnson rational damping.

    .. math::

        f^n_{\text{damp}}\left(R_0^{\text{AB}}\right) =
        \dfrac{R^n_{\text{AB}}}{R^n_{\text{AB}} +
        \left( a_1 R_0^{\text{AB}} + a_2 \right)^n}
    """

    radius_type = "r4r2"
    """Expectation value of r⁴/r² for the atoms in the system."""

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int = defaults.A1,
        a2: Tensor | float | int = defaults.A2,
        s6: Tensor | float | int = defaults.S6,
        rs6: Tensor | float | int = defaults.RS6,
        s8: Tensor | float | int = defaults.S8,
        rs8: Tensor | float | int = defaults.RS8,
        s9: Tensor | float | int = defaults.S9,
        rs9: Tensor | float | int = defaults.RS9,
        s10: Tensor | float | int = defaults.S10,
        alp: Tensor | float | int = defaults.ALP,
        bet: Tensor | float | int = defaults.BET,
        only_damping: bool = False,
    ) -> Tensor:
        """
        Rational damped dispersion interaction between pairs.

        Parameters
        ----------
        distances : Tensor
            Pairwise distances between atoms in the system.
            (shape: ``(..., nat, nat)``).
        radii : Tensor
            Critical radii for all atom pairs (shape: ``(..., nat, nat)``).
        param : Param
            DFT-D4 damping parameters.
        order : int
            Order of the dispersion interaction, e.g.
            6 for dipole-dipole, 8 for dipole-quadrupole and so on.

        Returns
        -------
        Tensor
            Values of the damping function.
        """
        radius = a1 * torch.sqrt(radii) + a2
        return 1.0 / (distances.pow(order) + radius.pow(order))


class ZeroDamping(Damping):
    r"""
    Zero damping (also known as Chai--Head-Gordon damping).

    .. math::

        f^n_{\text{damp}}\left(R^{\text{AB}}\right) =
        \dfrac{1}{1 + 6 \left( \dfrac{ R^{\text{AB}} }{ R_0^{\text{AB}} } \right)^{-a}} =
        \dfrac{1}{1 + 6 \left( \dfrac{ R_0^{\text{AB}} }{ R^{\text{AB}} } \right)^{a}}
    """

    radius_type = "rvdw"
    """Pair-wise van-der-Waals radii."""

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int = defaults.A1,
        a2: Tensor | float | int = defaults.A2,
        s6: Tensor | float | int = defaults.S6,
        rs6: Tensor | float | int = defaults.RS6,
        s8: Tensor | float | int = defaults.S8,
        rs8: Tensor | float | int = defaults.RS8,
        s9: Tensor | float | int = defaults.S9,
        rs9: Tensor | float | int = defaults.RS9,
        s10: Tensor | float | int = defaults.S10,
        alp: Tensor | float | int = defaults.ALP,
        bet: Tensor | float | int = defaults.BET,
        only_damping: bool = False,
    ) -> Tensor:
        if order not in (6, 8, 9):
            raise ValueError(
                "Zero-damping is only implemented for order 6 and 8."
            )

        if order == 6:
            alp_n = alp
            rs = rs6
        elif order == 8:
            alp_n = alp + 2.0
            rs = rs8
        elif order == 9:
            alp_n = alp
            rs = rs9

        # rs6 * r0ij / r1 = rs6 * rvdw(i,j) / sqrt(r2)
        t_n = rs * storch.divide(distances, radii)

        f_n = 1.0 / (1.0 + 6.0 * t_n**alp_n)

        if only_damping is True:
            return f_n

        # f6 / r6
        return storch.divide(f_n, distances**order)


class MZeroDamping(Damping):
    """Modified zero damping."""

    radius_type = "rvdw"
    """Pair-wise van-der-Waals radii."""

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int = defaults.A1,
        a2: Tensor | float | int = defaults.A2,
        s6: Tensor | float | int = defaults.S6,
        rs6: Tensor | float | int = defaults.RS6,
        s8: Tensor | float | int = defaults.S8,
        rs8: Tensor | float | int = defaults.RS8,
        s9: Tensor | float | int = defaults.S9,
        rs9: Tensor | float | int = defaults.RS9,
        s10: Tensor | float | int = defaults.S10,
        alp: Tensor | float | int = defaults.ALP,
        bet: Tensor | float | int = defaults.BET,
        only_damping: bool = False,
    ) -> Tensor:
        if order not in (6, 8):
            raise ValueError(
                "Zero-damping is only implemented for order 6 and 8."
            )

        if order == 6:
            alp_n = alp
            rs = rs6
        elif order == 8:
            alp_n = alp + 2.0
            rs = rs8

        # t6 = (r1 / (rs6*r0ij) + bet*r0ij)**(-alp6)
        t_n = distances / (rs * radii) + bet * radii

        # f6 / r6
        fn = 1.0 / (1.0 + 6.0 * t_n ** (-alp_n))
        return storch.divide(fn, distances**order)


class OptimisedPowerDamping(Damping):
    """Optimised-power damping."""

    radius_type = "r4r2"
    """Expectation value of r⁴/r² for the atoms in the system."""

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int = defaults.A1,
        a2: Tensor | float | int = defaults.A2,
        s6: Tensor | float | int = defaults.S6,
        rs6: Tensor | float | int = defaults.RS6,
        s8: Tensor | float | int = defaults.S8,
        rs8: Tensor | float | int = defaults.RS8,
        s9: Tensor | float | int = defaults.S9,
        rs9: Tensor | float | int = defaults.RS9,
        s10: Tensor | float | int = defaults.S10,
        alp: Tensor | float | int = defaults.ALP,
        bet: Tensor | float | int = defaults.BET,
        only_damping: bool = False,
    ) -> Tensor:
        if order not in (6, 8):
            raise ValueError(
                "OP-damping is only implemented for order 6 and 8."
            )

        radius = a1 * torch.sqrt(radii) + a2
        ab = radius**bet

        # r2**(bet * 0.5)
        rb = distances**bet

        # t6 = rb / (rb * r2**3 + ab * r0ij**6)
        return rb / (rb * distances**order + ab * distances**order)
