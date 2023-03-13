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
Atomic data: Chemical hardnesses
================================

Element-specific chemical hardnesses for the charge scaling function used
to extrapolate the C6 coefficients in DFT-D4.
"""

import torch

__all__ = ["gam"]

gam = torch.tensor(
    [
        0.00000000,  # None
        0.47259288,
        0.92203391,
        0.17452888,
        0.25700733,
        0.33949086,
        0.42195412,
        0.50438193,
        0.58691863,
        0.66931351,
        0.75191607,
        0.17964105,
        0.22157276,
        0.26348578,
        0.30539645,
        0.34734014,
        0.38924725,
        0.43115670,
        0.47308269,
        0.17105469,
        0.20276244,
        0.21007322,
        0.21739647,
        0.22471039,
        0.23201501,
        0.23933969,
        0.24665638,
        0.25398255,
        0.26128863,
        0.26859476,
        0.27592565,
        0.30762999,
        0.33931580,
        0.37235985,
        0.40273549,
        0.43445776,
        0.46611708,
        0.15585079,
        0.18649324,
        0.19356210,
        0.20063311,
        0.20770522,
        0.21477254,
        0.22184614,
        0.22891872,
        0.23598621,
        0.24305612,
        0.25013018,
        0.25719937,
        0.28784780,
        0.31848673,
        0.34912431,
        0.37976593,
        0.41040808,
        0.44105777,
        0.05019332,
        0.06762570,
        0.08504445,
        0.10247736,
        0.11991105,
        0.13732772,
        0.15476297,
        0.17218265,
        0.18961288,
        0.20704760,
        0.22446752,
        0.24189645,
        0.25932503,
        0.27676094,
        0.29418231,
        0.31159587,
        0.32902274,
        0.34592298,
        0.36388048,
        0.38130586,
        0.39877476,
        0.41614298,
        0.43364510,
        0.45104014,
        0.46848986,
        0.48584550,
        0.12526730,
        0.14268677,
        0.16011615,
        0.17755889,
        0.19497557,
        0.21240778,
        0.07263525,
        0.09422158,
        0.09920295,
        0.10418621,
        0.14235633,
        0.16394294,
        0.18551941,
        0.22370139,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
    ]
)
"""Element-specific chemical hardnesses."""
