"""
Data for testing D4 coordination number (taken from D4 testsuite).
"""

import torch

from tad_dftd4.typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference values."""

    cn: Tensor
    """DFT-D4 coordination number"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values."""


refs: dict[str, Refs] = {
    "SiH4": {
        "cn": torch.tensor(
            [
                3.64990496420721e0,
                9.12476241051802e-1,
                9.12476241051802e-1,
                9.12476241051802e-1,
                9.12476241051802e-1,
            ]
        )
    },
    "MB16_43_01": {
        "cn": torch.tensor(
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
            ]
        )
    },
    "MB16_43_02": {
        "cn": torch.tensor(
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
            ]
        )
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
