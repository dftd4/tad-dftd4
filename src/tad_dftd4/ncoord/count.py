"""
Counting functions.
"""

import torch

from .. import defaults
from ..typing import Tensor


def erf_count(r: Tensor, r0: Tensor, kcn: float = defaults.D4_KCN) -> Tensor:
    """
    Error function counting function for coordination number contributions.

    Parameters
    ----------
    r : Tensor
        Internuclear distances.
    r0 : Tensor
        Covalent atomic radii (R_AB = R_A + R_B).
    kcn : float, optional
        Steepness of the counting function. Defaults to `defaults.D4_KCN`.

    Returns
    -------
    Tensor
        Count of coordination number contribution.
    """
    return 0.5 * (1.0 + torch.erf(-kcn * (r / r0 - 1.0)))
