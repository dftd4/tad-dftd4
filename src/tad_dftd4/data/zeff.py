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
Atomic data: Effective nuclear charges
=======================================

Effective nuclear charges from the def2-ECPs used for calculating the reference
polarizibilities for DFT-D4.
"""

import torch

__all__ = ["zeff"]

# fmt: off
zeff = torch.tensor([
     0, # None
     1,                                                 2,   # H-He
     3, 4,                               5, 6, 7, 8, 9,10,   # Li-Ne
    11,12,                              13,14,15,16,17,18,   # Na-Ar
    19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,   # K-Kr
     9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,   # Rb-Xe
     9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,   # Cs-Lu
    12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,  # Hf-Rn
    # just copy and paste from above
     9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,   # Fr-Lr
    12,13,14,15,16,17,18,19,20,21,22,23,24,25,26  # Rf-Og
])
# fmt: on
"""Effective nuclear charges."""
