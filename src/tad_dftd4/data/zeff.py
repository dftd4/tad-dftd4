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
