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
Damping Parameters
==================

Read damping parameters from toml file. The TOML file is coped from the DFT-D4
Fortran GitHub repository.
(https://github.com/dftd4/dftd4/blob/main/assets/parameters.toml)
"""

from __future__ import annotations

from enum import Enum
from typing import Type

from pydantic import BaseModel, Field


from tad_mctc.typing import Tensor, TypedDict
from typing_extensions import NotRequired

__all__ = [
    "Param",
    "DispersionMethod",
    "Variant",
    "BaseParams",
    "RationalParams",
    "ZeroParams",
    "MZeroParams",
    "OptimisedPowerParams",
    "PARAM_MODELS",
]


class Param(TypedDict, total=False):
    """Type annotation for dispersion parameters."""

    a1: NotRequired[Tensor | float]
    """Scaling for the C8 / C6 ratio in the critical radius (0.4)."""

    a2: NotRequired[Tensor | float]
    """Offset parameter for the critical radius (5.0)."""

    s6: NotRequired[Tensor | float]
    """Scaling of dipole-dipole term (1.0 to retain correct limit)."""

    rs6: NotRequired[Tensor | float]
    """Scaling of dipole-dipole term for (modified) zero damping (1.0)."""

    s8: NotRequired[Tensor | float]
    """Scaling of dipole-quadrupole term (1.0)."""

    rs8: NotRequired[Tensor | float]
    """Scaling of dipole-quadrupole term for (modified) zero damping (1.0)."""

    s9: NotRequired[Tensor | float]
    """Scaling of three-body term (1.0)."""

    rs9: NotRequired[Tensor | float]
    """Scaling for van-der-Waals radii in damping function (4.0/3.0)."""

    s10: NotRequired[Tensor | float]
    """Scaling of quadrupole-quadrupole term (0.0)."""

    alp: NotRequired[Tensor | float]
    """Exponent of zero damping function (16.0)."""

    bet: NotRequired[Tensor | float]
    """Exponent of mzero or optimized-power damping function (0.0)."""

    doi: NotRequired[str]
    """DOI of the reference paper."""


class DispersionMethod(str, Enum):
    d3 = "d3"
    d4 = "d4"
    d5 = "d5"


class Variant(str, Enum):
    bj = "bj"
    bj_eeq_atm = "bj-eeq-atm"  # D4 default
    zero = "zero"
    mzero = "mzero"
    optimizedpower = "optimizedpower"


# Parameter data classes


class BaseParams(BaseModel):
    """Fields shared by *all* damping families."""

    s6: float = Field(1.0)
    s8: float = Field(1.0)
    s10: float = Field(0.0)
    doi: str | None = Field(None, description="Optional reference")


class RationalParams(BaseParams):
    a1: float
    a2: float


class ZeroParams(BaseParams):
    alp: float
    rs6: float
    rs8: float
    rs9: float = Field(0.0, description="Radius scaling for MB term")
    bet: float = Field(0.0, description="exponent shift β (optional for zero)")


class MZeroParams(ZeroParams):
    bet: float = Field(
        ...,  # ellipsis indicates this field is required
        description="exponent shift β (required for m-zero damping)",
    )


class OptimisedPowerParams(BaseParams):
    a1: float
    a2: float
    bet: float


PARAM_MODELS: dict[Variant, Type[BaseParams]] = {
    Variant.bj: RationalParams,
    Variant.bj_eeq_atm: RationalParams,  # same functional form
    Variant.zero: ZeroParams,
    Variant.mzero: MZeroParams,
    Variant.optimizedpower: OptimisedPowerParams,
}
"""Registry of parameter models for each damping variant."""
