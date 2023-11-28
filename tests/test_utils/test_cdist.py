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
Test the utility functions.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4 import utils
from tad_dftd4._typing import DD

from ..conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all(dtype: torch.dtype) -> None:
    """
    The single precision test sometimes fails on my GPU with the following
    thresholds:

    ```
    tol = 1e-6 if dtype == torch.float else 1e-14
    ```

    Only one matrix element seems to be affected. It also appears that the
    failure only happens if `torch.rand` was run before. To be precise,

    ```
    pytest -vv test/test_ncoord/test_grad.py test/test_utils/ --cuda --slow
    ```

    fails, while

    ```
    pytest -vv test/test_utils/ --cuda --slow
    ```

    works. It also works if I remove the random tensors in the gradient test
    (test/test_ncoord/test_grad.py).

    It can be fixed with

    ```
    torch.use_deterministic_algorithms(True)
    ```

    and following the PyTorch instructions to set a specific
    environment variable.

    ```
    CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -vv test/test_ncoord/test_grad.py test/test_utils/ --cuda --slow
    ```

    (For simplicity, I just reduced the tolerances for single precision.)
    """
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-6 if dtype == torch.float else 1e-14

    # only one element actually fails
    if "cuda" in str(DEVICE) and dtype == torch.float:
        tol = 1e-3

    x = torch.randn(2, 3, 4, **dd)

    d1 = utils.cdist(x)
    d2 = utils.distance.cdist_direct_expansion(x, x, p=2)
    d3 = utils.distance.euclidean_dist_quadratic_expansion(x, x)

    assert pytest.approx(d1.cpu(), abs=tol) == d2.cpu()
    assert pytest.approx(d2.cpu(), abs=tol) == d3.cpu()
    assert pytest.approx(d3.cpu(), abs=tol) == d1.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p", [2, 3, 4, 5])
def test_ps(dtype: torch.dtype, p: int) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-6 if dtype == torch.float else 1e-14

    x = torch.randn(2, 4, 5, **dd)
    y = torch.randn(2, 4, 5, **dd)

    d1 = utils.cdist(x, y, p=p)
    d2 = torch.cdist(x, y, p=p)

    assert pytest.approx(d1.cpu(), abs=tol) == d2.cpu()
