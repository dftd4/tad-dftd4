"""
Default settings for `tad-dftd4`.
"""

import torch

# PyTorch

TORCH_DTYPE = torch.float32
"""Default data type for floating point tensors."""

TORCH_DTYPE_CHOICES = ["float32", "float64", "double", "sp", "dp"]
"""List of possible choices for `TORCH_DTYPE`."""

TORCH_DEVICE = "cpu"
"""Default device for tensors."""


# DFT-D4

D4_CN_CUTOFF = 30.0
"""Coordination number cutoff (30.0)."""

D4_DISP2_CUTOFF = 60.0
"""Two-body interaction cutoff (60.0)."""

D4_DISP3_CUTOFF = 40.0
"""Three-body interaction cutoff (40.0)."""

D4_KCN = 7.5
"""Steepness of counting function (7.5)."""

D4_K4 = 4.10451
"""Parameter for electronegativity scaling."""

D4_K5 = 19.08857
"""Parameter for electronegativity scaling."""

D4_K6 = 2 * 11.28174**2  # 254.56
"""Parameter for electronegativity scaling."""
