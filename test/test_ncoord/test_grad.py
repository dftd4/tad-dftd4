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
Test derivative (w.r.t. positions) of the exponential and error counting
functions used for the coordination number within the EEQ model and D4.
"""
from __future__ import annotations

from math import sqrt

import pytest
import torch

from tad_dftd4._typing import CountingFunction
from tad_dftd4.ncoord import derf_count, dexp_count, erf_count, exp_count


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize(
    "function",
    [
        (exp_count, dexp_count),
        (erf_count, derf_count),
    ],
)
def test_count_grad(
    dtype: torch.dtype, function: tuple[CountingFunction, CountingFunction]
) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    cf, dcf = function

    a = torch.rand(4, dtype=dtype)
    b = torch.rand(4, dtype=dtype)

    a_grad = a.detach().clone().requires_grad_(True)
    count = cf(a_grad, b)

    grad_auto = torch.autograd.grad(count.sum(-1), a_grad)[0]
    grad_expl = dcf(a, b)

    assert pytest.approx(grad_auto, abs=tol) == grad_expl
