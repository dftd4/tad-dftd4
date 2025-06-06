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
Damping: Zero (Chai--Head-Gordon) damping function
==================================================

This module defines the rational damping function, also known as Becke-Johnson
damping.

.. math::

    f^n_{\text{damp}}\left(R^{\text{AB}}\right) =
    \dfrac{1}{1 + 6 \left( \dfrac{ R^{\text{AB}} }{ R_0^{\text{AB}} } \right)^{-a}} =
    \dfrac{1}{1 + 6 \left( \dfrac{ R_0^{\text{AB}} }{ R^{\text{AB}} } \right)^{a}}
"""
from __future__ import annotations

from tad_mctc import storch

from ..typing import Tensor

__all__ = ["zero_damping"]


def zero_damping(x: Tensor, alp: Tensor) -> Tensor:
    """
    Zero damping function.

    Parameters
    ----------
    x : Tensor
        Input tensor, typically a ratio of radii.
    alp : Tensor
        Exponent of the zero damping function.

    Returns
    -------
    Tensor
        Damping function values.
    """
    return storch.reciprocal(1.0 + 6.0 * x**alp)
