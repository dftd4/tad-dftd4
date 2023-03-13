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
Atomic data: Electronegativities
================================

Pauling electronegativities, used for the covalent coordination number.
"""

import torch

__all__ = ["pauling_en"]

# fmt: off
pauling_en = torch.tensor([
    0.00,  # None
    2.20,3.00,  # H,He
    0.98,1.57,2.04,2.55,3.04,3.44,3.98,4.50,  # Li-Ne
    0.93,1.31,1.61,1.90,2.19,2.58,3.16,3.50,  # Na-Ar
    0.82,1.00,  # K,Ca
    1.36,1.54,1.63,1.66,1.55,  # Sc-
    1.83,1.88,1.91,1.90,1.65,  # -Zn
    1.81,2.01,2.18,2.55,2.96,3.00,  # Ga-Kr
    0.82,0.95,  # Rb,Sr
    1.22,1.33,1.60,2.16,1.90,  # Y-
    2.20,2.28,2.20,1.93,1.69,  # -Cd
    1.78,1.96,2.05,2.10,2.66,2.60,  # In-Xe
    0.79,0.89,  # Cs,Ba
    1.10,1.12,1.13,1.14,1.15,1.17,1.18,  # La-Eu
    1.20,1.21,1.22,1.23,1.24,1.25,1.26,  # Gd-Yb
    1.27,1.30,1.50,2.36,1.90,  # Lu-
    2.20,2.20,2.28,2.54,2.00,  # -Hg
    1.62,2.33,2.02,2.00,2.20,2.20,  # Tl-Rn
    # only dummies below
    1.50,1.50,  # Fr,Ra
    1.50,1.50,1.50,1.50,1.50,1.50,1.50,  # Ac-Am
    1.50,1.50,1.50,1.50,1.50,1.50,1.50,  # Cm-No
    1.50,1.50,1.50,1.50,1.50,  # Rf-
    1.50,1.50,1.50,1.50,1.50,  # Rf-Cn
    1.50,1.50,1.50,1.50,1.50,1.50  # Nh-Og
])
# fmt: on
"""Pauling electronegativities, used for the covalent coordination number."""
