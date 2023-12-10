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
Testing the charges module
==========================

This module tests the EEQ charge model including:
 - single molecule
 - batched
 - ghost atoms
 - autograd via `gradcheck`

Note that `torch.linalg.solve` gives slightly different results (around 1e-5
to 1e-6) across different PyTorch versions (1.11.0 vs 1.13.0) for single
precision. For double precision, however the results are identical.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4 import charges
from tad_dftd4._typing import Tensor

from .samples import samples


@pytest.mark.grad
def test_gradcheck(dtype: torch.dtype = torch.double):
    sample = samples["NH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype).detach()
    total_charge = sample["total_charge"].type(dtype).detach()
    cn = torch.tensor(
        [3.0, 1.0, 1.0, 1.0],
        dtype=dtype,
    )
    eeq = charges.ChargeModel.param2019().type(dtype)

    positions.requires_grad_(True)
    total_charge.requires_grad_(True)

    def func(positions: Tensor, total_charge: Tensor):
        return torch.sum(
            charges.solve(numbers, positions, total_charge, eeq, cn)[0], -1
        )

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, (positions, total_charge))
