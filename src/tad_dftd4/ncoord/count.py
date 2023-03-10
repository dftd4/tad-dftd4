"""
Coordination number: Counting functions
=======================================

This module contains the exponential and the error function counting functions
for the determination of the coordination number.

The exponential counting function is used within the D3 model but not within
the D4 model. Nevertheless, it is included here.
Additionally, the analytical derivatives for both counting functions are also
provided and can be used for checking the autograd results.
"""

from math import pi, sqrt

import torch

from .. import defaults
from .._typing import Tensor


def exp_count(r: Tensor, r0: Tensor, kcn: float = defaults.D4_KCN) -> Tensor:
    """
    Exponential counting function for coordination number contributions.

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
    return 1.0 / (1.0 + torch.exp(-kcn * (r0 / r - 1.0)))


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


# analytical derivatives


def dexp_count(r: Tensor, r0: Tensor, kcn: float = defaults.D4_KCN) -> Tensor:
    """
    Derivative of the exponential counting function w.r.t. the distance.

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
        Derivative of count of coordination number contribution.
    """
    expterm = torch.exp(-kcn * (r0 / r - 1.0))
    return (-kcn * r0 * expterm) / (r**2 * ((expterm + 1.0) ** 2))


def derf_count(r: Tensor, r0: Tensor, kcn: float = defaults.D4_KCN) -> Tensor:
    """
    Derivative of error function counting function w.r.t. the distance.

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
        Derivative of count of coordination number contribution.
    """
    return -kcn / sqrt(pi) / r0 * torch.exp(-(kcn**2) * (r - r0) ** 2 / r0**2)
