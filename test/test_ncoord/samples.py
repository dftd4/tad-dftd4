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
Data for testing D4 coordination number (taken from D4 testsuite).
"""
from __future__ import annotations

import torch

from tad_dftd4._typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference values."""

    cn_d4: Tensor
    """DFT-D4 coordination number"""

    cn_eeq: Tensor
    """DFT-D4 coordination number"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values."""


refs: dict[str, Refs] = {
    # "SiH4": {
    #     "cn_d4": torch.tensor(
    #         [
    #             3.64990496420721e0,
    #             9.12476241051802e-1,
    #             9.12476241051802e-1,
    #             9.12476241051802e-1,
    #             9.12476241051802e-1,
    #         ]
    #     )
    # },
    "MB16_43_01": {
        "cn_d4": torch.tensor(
            [
                3.07349677110402e0,
                9.31461605116103e-1,
                1.43709439375839e0,
                1.33309431581960e0,
                7.20743527030337e-1,
                8.59659004770982e-1,
                1.35782158177921e0,
                1.53940006996025e0,
                3.19400368195259e0,
                8.12162111631342e-1,
                8.59533443784854e-1,
                1.53347108155587e0,
                4.23314989525721e0,
                3.03048504567396e0,
                3.45229319488306e0,
                4.28478289652264e0,
            ],
            dtype=torch.double,
        ),
        "cn_eeq": torch.tensor(
            [
                4.03670396918677e0,
                9.72798721502297e-1,
                1.98698465669657e0,
                1.47312608051590e0,
                9.97552155866795e-1,
                9.96862039916965e-1,
                1.45188437942218e0,
                1.99267278111197e0,
                3.84566220624764e0,
                1.00242959599510e0,
                9.96715113655073e-1,
                1.92505296745902e0,
                4.62015142034058e0,
                3.81973465175781e0,
                3.95710919750442e0,
                5.33862698412205e0,
            ],
            dtype=torch.double,
        ),
    },
    "MB16_43_02": {
        "cn_d4": torch.tensor(
            [
                9.20259141516190e-1,
                3.29216939906043e0,
                3.51944438412931e0,
                2.25877973040028e0,
                4.46999073626179e0,
                8.18916367808423e-1,
                9.28914937407466e-1,
                9.30833050893587e-1,
                4.60708718003244e0,
                8.18343168300509e-1,
                3.70959638795740e0,
                2.87405845608016e0,
                1.24015900552686e0,
                9.11079070954527e-1,
                1.57258868344791e0,
                1.67284525339418e0,
            ],
            dtype=torch.double,
        ),
        "cn_eeq": torch.tensor(
            [
                9.61099101791137e-1,
                3.87581247819995e0,
                3.80155140067831e0,
                2.96990277678560e0,
                5.43508021969867e0,
                1.01156705157372e0,
                9.70139042949472e-1,
                9.72142268717279e-1,
                4.98780441573354e0,
                1.01084927946071e0,
                3.92876025928151e0,
                3.88754303198463e0,
                1.99577129500205e0,
                9.71947229716782e-1,
                1.66031989216595e0,
                1.97969868901054e0,
            ],
            dtype=torch.double,
        ),
    },
    "MB16_43_03": {
        "cn_d4": torch.tensor(
            [
                3.7032957567263e00,
                2.1147658285163e00,
                9.2368282669771e-01,
                3.8899976705596e00,
                6.1749008156797e00,
                4.1059550685889e00,
                4.2291653493870e00,
                1.0428768741572e00,
                9.2368698514396e-01,
                9.2448784893139e-01,
                1.2292763680072e00,
                2.5985300198546e00,
                4.3001547096965e00,
                8.2908165089510e-01,
                2.6523901063779e00,
                1.1984043133662e00,
            ],
            dtype=torch.double,
        ),
        "cn_eeq": torch.tensor(
            [],
            dtype=torch.double,
        ),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
