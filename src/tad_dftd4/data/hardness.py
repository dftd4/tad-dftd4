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
Data: Chemical hardnesses
=========================

Element-specific chemical hardnesses for the charge scaling function used
to extrapolate the C6 coefficients in DFT-D4.
"""
import torch

__all__ = ["GAM"]


GAM = torch.tensor(
    [
        0.00000000,  # None
        0.47259288,  # H
        0.92203391,  # He
        0.17452888,  # Li (2nd)
        0.25700733,  # Be
        0.33949086,  # B
        0.42195412,  # C
        0.50438193,  # N
        0.58691863,  # O
        0.66931351,  # F
        0.75191607,  # Ne
        0.17964105,  # Na (3rd)
        0.22157276,  # Mg
        0.26348578,  # Al
        0.30539645,  # Si
        0.34734014,  # P
        0.38924725,  # S
        0.43115670,  # Cl
        0.47308269,  # Ar
        0.17105469,  # K  (4th)
        0.20276244,  # Ca
        0.21007322,  # Sc
        0.21739647,  # Ti
        0.22471039,  # V
        0.23201501,  # Cr
        0.23933969,  # Mn
        0.24665638,  # Fe
        0.25398255,  # Co
        0.26128863,  # Ni
        0.26859476,  # Cu
        0.27592565,  # Zn
        0.30762999,  # Ga
        0.33931580,  # Ge
        0.37235985,  # As
        0.40273549,  # Se
        0.43445776,  # Br
        0.46611708,  # Kr
        0.15585079,  # Rb (5th)
        0.18649324,  # Sr
        0.19356210,  # Y
        0.20063311,  # Zr
        0.20770522,  # Nb
        0.21477254,  # Mo
        0.22184614,  # Tc
        0.22891872,  # Ru
        0.23598621,  # Rh
        0.24305612,  # Pd
        0.25013018,  # Ag
        0.25719937,  # Cd
        0.28784780,  # In
        0.31848673,  # Sn
        0.34912431,  # Sb
        0.37976593,  # Te
        0.41040808,  # I
        0.44105777,  # Xe
        0.05019332,  # Cs (6th)
        0.06762570,  # Ba
        0.08504445,  # La
        0.10247736,  # Ce
        0.11991105,  # Pr
        0.13732772,  # Nd
        0.15476297,  # Pm
        0.17218265,  # Sm
        0.18961288,  # Eu
        0.20704760,  # Gd
        0.22446752,  # Tb
        0.24189645,  # Dy
        0.25932503,  # Ho
        0.27676094,  # Er
        0.29418231,  # Tm
        0.31159587,  # Yb
        0.32902274,  # Lu
        0.34592298,  # Hf
        0.36388048,  # Ta
        0.38130586,  # W
        0.39877476,  # Re
        0.41614298,  # Os
        0.43364510,  # Ir
        0.45104014,  # Pt
        0.46848986,  # Au
        0.48584550,  # Hg
        0.12526730,  # Tl
        0.14268677,  # Pb
        0.16011615,  # Bi
        0.17755889,  # Po
        0.19497557,  # At
        0.21240778,  # Rn
        0.07263525,  # Fr (7th)
        0.09422158,  # Ra
        0.09920295,  # Ac
        0.10418621,  # Th
        0.14235633,  # Pa
        0.16394294,  # U
        0.18551941,  # Np
        0.22370139,  # Pu
        0.25110000,  # Am
        0.25030000,  # Cm
        0.28840000,  # Bk
        0.31000000,  # Cf
        0.33160000,  # Es
        0.35320000,  # Fm
        0.36820000,  # Md
        0.39630000,  # No
        0.40140000,  # Lr
        0.00000000,  # Rf
        0.00000000,  # Db
        0.00000000,  # Sg
        0.00000000,  # Bh
        0.00000000,  # Hs
        0.00000000,  # Mt
        0.00000000,  # Ds
        0.00000000,  # Rg
        0.00000000,  # Cn
        0.00000000,  # Nh
        0.00000000,  # Fl
        0.00000000,  # Lv
        0.00000000,  # Mc
        0.00000000,  # Ts
        0.00000000,  # Og
    ]
)
"""Element-specific chemical hardnesses."""
