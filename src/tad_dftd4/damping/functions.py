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
- Becke's Z-damping
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from functools import wraps

import torch
from tad_mctc import storch

from .. import defaults
from ..typing import Literal, Tensor

_MISSING = object()

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

    doi: str | None = None
    """Digital Object Identifier (DOI) for the damping function."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)

        req_map: dict[int, tuple[str, ...]] = getattr(
            cls, "REQUIRED_BY_ORDER", {}
        )
        if not req_map:  # nothing special declared
            return

        # sanity-check: every declared name must really exist in _f
        sig = inspect.signature(cls._f)
        unknown = {
            name
            for seq in req_map.values()
            for name in seq
            if name not in sig.parameters
        }
        if unknown:
            raise TypeError(
                f"{cls.__name__}.REQUIRED_BY_ORDER refers to "
                f"unknown parameter(s): {', '.join(unknown)}"
            )

        # wrap _f once so every call is validated fast
        original_f = cls._f

        @wraps(original_f)
        def _checked_f(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()

            order_val = bound.arguments["order"]
            required = req_map.get(order_val, ())

            missing = [
                name
                for name in required
                if bound.arguments.get(name, _MISSING) in (None, _MISSING)
            ]
            if missing:
                raise TypeError(
                    f"{cls.__name__} (order {order_val}) "
                    f"requires keyword(s): {', '.join(missing)}"
                )
            return original_f(self, *args, **kwargs)

        cls._f = _checked_f

    def __call__(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        only_damping: bool = False,
        *,
        a1: Tensor | float | int | None = None,
        a2: Tensor | float | int | None = None,
        s6: Tensor | float | int | None = defaults.S6,
        rs6: Tensor | float | int | None = None,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = None,
        s9: Tensor | float | int | None = defaults.S9,
        rs9: Tensor | float | int | None = defaults.RS9,
        s10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = defaults.ALP,
        bet: Tensor | float | int | None = None,
        doi: str | None = None,
    ) -> Tensor:
        self.doi = doi

        # Do not skip the cases where the parameter value is 0.0, as this leads
        # to issues for gradient calculations.

        if order == 6 and (s6 is None and rs6 is None):
            return torch.zeros_like(distances)

        if order == 8 and (s8 is None and rs8 is None):
            return torch.zeros_like(distances)

        if order == 9 and (s9 is None and rs9 is None):
            return torch.zeros_like(distances)

        if order == 10 and s10 is None:
            return torch.zeros_like(distances)

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
        a1: Tensor | float | int | None = None,
        a2: Tensor | float | int | None = None,
        s6: Tensor | float | int | None = None,
        rs6: Tensor | float | int | None = None,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = None,
        s9: Tensor | float | int | None = None,
        rs9: Tensor | float | int | None = None,
        s10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = None,
        bet: Tensor | float | int | None = None,
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

    REQUIRED_BY_ORDER = {
        6: ("a1", "a2"),
        8: ("a1", "a2"),
        10: ("a1", "a2"),
    }
    """Required parameters for the damping function."""

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int | None = defaults.A1,
        a2: Tensor | float | int | None = defaults.A2,
        s6: Tensor | float | int | None = None,
        rs6: Tensor | float | int | None = None,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = None,
        s9: Tensor | float | int | None = None,
        rs9: Tensor | float | int | None = None,
        s10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = None,
        bet: Tensor | float | int | None = None,
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
        assert a1 is not None and a2 is not None

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

    REQUIRED_BY_ORDER = {
        6: ("rs6", "alp"),
        8: ("rs8", "alp"),
        9: ("rs9", "alp"),
    }
    """Required parameters for the damping function."""

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int | None = None,
        a2: Tensor | float | int | None = None,
        s6: Tensor | float | int | None = None,
        rs6: Tensor | float | int | None = defaults.RS6,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = defaults.RS8,
        s9: Tensor | float | int | None = None,
        rs9: Tensor | float | int | None = defaults.RS9,
        s10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = defaults.ALP,
        bet: Tensor | float | int | None = None,
        only_damping: bool = False,
    ) -> Tensor:
        assert alp is not None

        if order not in (6, 8, 9):
            raise ValueError(
                "Zero-damping is only implemented for order 6, 8, and 9. "
                f"Got {order} instead."
            )

        if order == 6:
            assert rs6 is not None
            alp_n = alp
            rs = rs6
        elif order == 8:
            assert rs8 is not None
            alp_n = alp + 2.0
            rs = rs8
        elif order == 9:
            assert rs9 is not None
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

    REQUIRED_BY_ORDER = {
        6: ("rs6", "alp", "bet"),
        8: ("rs8", "alp", "bet"),
    }
    """Required parameters for the damping function."""

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int | None = None,
        a2: Tensor | float | int | None = None,
        s6: Tensor | float | int | None = None,
        rs6: Tensor | float | int | None = defaults.RS6,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = defaults.RS8,
        s9: Tensor | float | int | None = None,
        rs9: Tensor | float | int | None = None,
        s10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = defaults.ALP,
        bet: Tensor | float | int | None = defaults.BET,
        only_damping: bool = False,
    ) -> Tensor:
        assert alp is not None
        assert bet is not None

        if order not in (6, 8):
            raise ValueError(
                "Zero-damping is only implemented for order 6 and 8."
            )

        if order == 6:
            assert rs6 is not None
            alp_n = alp
            rs = rs6
        elif order == 8:
            assert rs8 is not None
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

    REQUIRED_BY_ORDER = {
        6: ("a1", "a2", "bet"),
        8: ("a1", "a2", "bet"),
    }
    """Required parameters for the damping function."""

    def _f(
        self,
        distances: Tensor,
        radii: Tensor,
        order: int,
        *,
        a1: Tensor | float | int | None = defaults.A1,
        a2: Tensor | float | int | None = defaults.A2,
        s6: Tensor | float | int | None = None,
        rs6: Tensor | float | int | None = None,
        s8: Tensor | float | int | None = None,
        rs8: Tensor | float | int | None = None,
        s9: Tensor | float | int | None = None,
        rs9: Tensor | float | int | None = None,
        s10: Tensor | float | int | None = None,
        alp: Tensor | float | int | None = None,
        bet: Tensor | float | int | None = defaults.BET,
        only_damping: bool = False,
    ) -> Tensor:
        assert a1 is not None
        assert a2 is not None
        assert bet is not None

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


class ZDamping(Damping):
    """Becke's Z-damping."""
