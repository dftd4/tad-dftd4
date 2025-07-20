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
Dispersion Methods: D3
======================

DFT-D3 dispersion method implementation.
"""
from __future__ import annotations

from ..damping import RationalDamping, ZeroDamping
from .base import Disp
from .threebody import ATM, C9ApproxMixin, RadiiVDWMixin
from .twobody import TwoBodyTerm

__all__ = ["DispD3BJ", "DispD3Zero"]


class _DispD3(Disp):
    """Standard DFT-D3 dispersion method."""

    ALLOWED_MODELS = ("d3",)
    """Allowed DFT-D models for the calculation."""


##############################################################################


class D3ATM(C9ApproxMixin, RadiiVDWMixin, ATM):
    """D3 ATM: approximate C9 + pair-wise VDW radii."""


class DispD3BJ(_DispD3):
    """Standard DFT-D3 dispersion method."""

    TERMS = [
        (
            TwoBodyTerm,
            {"damping_fn": RationalDamping(), "charge_dependent": False},
        ),
        (
            D3ATM,
            {"damping_fn": ZeroDamping(), "charge_dependent": False},
        ),
    ]
    """List of dispersion terms to be registered in the constructor."""


##############################################################################


class DispD3Zero(_DispD3):
    """Zero-damping DFT-D3 dispersion method."""

    TERMS = [
        (
            TwoBodyTerm,
            {"damping_fn": ZeroDamping(), "charge_dependent": False},
        ),
        (
            D3ATM,
            {"damping_fn": ZeroDamping(), "charge_dependent": False},
        ),
    ]
    """List of dispersion terms to be registered in the constructor."""
