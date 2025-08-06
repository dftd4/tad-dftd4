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
Utils
=====

Utility functions for this project.
"""
from __future__ import annotations

import torch
from tad_mctc.math import einsum
from tad_mctc.typing import Tensor

__all__ = ["trapzd", "trapzd_atm", "trapzd_noref", "is_exceptional"]


def trapzd(pol1: Tensor, pol2: Tensor | None = None) -> Tensor:
    """
    Numerical Casimir--Polder integration.

    Parameters
    ----------
    pol1 : Tensor
        Polarizabilities of shape ``(..., nat, nref, 23)``.
    pol2 : Tensor | None, optional
        Polarizabilities of shape ``(..., nat, nref, 23)``. Defaults to
        ``None``, in which case ``pol2`` is set to ``pol1``.

    Returns
    -------
    Tensor
        C6 coefficients of shape ``(..., nat, nat, nref, nref)``.
    """
    thopi = 3.0 / 3.141592653589793238462643383279502884197

    weights = torch.tensor(
        [
            2.4999500000000000e-002,
            4.9999500000000000e-002,
            7.5000000000000010e-002,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1500000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.3500000000000000,
            0.5000000000000000,
            0.7500000000000000,
            1.0000000000000000,
            1.7500000000000000,
            2.5000000000000000,
            1.2500000000000000,
        ],
        device=pol1.device,
        dtype=pol1.dtype,
    )

    # NOTE: In the old version, a memory inefficient intermediate tensor was
    # created. The new version uses `einsum` to avoid this.
    #
    # (..., 1, nat, 1, nref, 23) * (..., nat, 1, nref, 1, 23) =
    # (..., nat, nat, nref, nref, 23) -> (..., nat, nat, nref, nref)
    # a = alpha.unsqueeze(-4).unsqueeze(-3) * alpha.unsqueeze(-3).unsqueeze(-2)
    #
    # rc6 = thopi * torch.sum(weights * a, dim=-1)

    return thopi * einsum(
        "w,...iaw,...jbw->...ijab",
        *(weights, pol1, pol1 if pol2 is None else pol2),
    )


def trapzd_noref(pol1: Tensor, pol2: Tensor | None = None) -> Tensor:
    """
    Numerical Casimir--Polder integration.

    This version takes polarizabilities of shape ``(..., nat, 23)``, i.e.,
    the reference dimension has already been summed over.

    Parameters
    ----------
    pol1 : Tensor
        Polarizabilities of shape ``(..., nat, 23)``.
    pol2 : Tensor | None, optional
        Polarizabilities of shape ``(..., nat, 23)``. Defaults to
        ``None``, in which case ``pol2`` is set to ``pol1``.

    Returns
    -------
    Tensor
        C6 coefficients of shape ``(..., nat, nat)``.
    """
    thopi = 3.0 / 3.141592653589793238462643383279502884197

    weights = torch.tensor(
        [
            2.4999500000000000e-002,
            4.9999500000000000e-002,
            7.5000000000000010e-002,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1500000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.3500000000000000,
            0.5000000000000000,
            0.7500000000000000,
            1.0000000000000000,
            1.7500000000000000,
            2.5000000000000000,
            1.2500000000000000,
        ],
        device=pol1.device,
        dtype=pol1.dtype,
    )

    return thopi * einsum(
        "w,...iw,...jw->...ij",
        *(weights, pol1, pol1 if pol2 is None else pol2),
    )


def trapzd_atm(pol: Tensor) -> Tensor:
    """
    Numerical Casimir--Polder integration for ATM term.

    This version takes polarizabilities of shape
    ``(..., nat, nat, nat, 23)``, i.e., the reference dimension has
    already been summed over.

    Parameters
    ----------
    pol : Tensor
        Polarizabilities of shape ``(..., nat, nat, nat, 23)``.

    Returns
    -------
    Tensor
        C9 coefficients of shape ``(..., nat, nat, nat)``.
    """
    thopi = 3.0 / 3.141592653589793238462643383279502884197

    weights = torch.tensor(
        [
            2.4999500000000000e-002,
            4.9999500000000000e-002,
            7.5000000000000010e-002,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1500000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.3500000000000000,
            0.5000000000000000,
            0.7500000000000000,
            1.0000000000000000,
            1.7500000000000000,
            2.5000000000000000,
            1.2500000000000000,
        ],
        device=pol.device,
        dtype=pol.dtype,
    )

    return thopi * einsum("...ijkw,w->...ijk", pol, weights)


def is_exceptional(x: Tensor, dtype: torch.dtype) -> Tensor:
    """
    Check if a tensor is exceptional (``NaN`` or too large).

    Parameters
    ----------
    x : Tensor
        Tensor to check.
    dtype : torch.dtype
        Data type of the tensor.

    Returns
    -------
    Tensor
        Boolean tensor indicating exceptional values.
    """
    return torch.isnan(x) | (x > torch.finfo(dtype).max)
