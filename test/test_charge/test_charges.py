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

The tests surrounding the EEQ charge model include:
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
from tad_dftd4._typing import DD
from tad_dftd4.utils import pack

from ..conftest import DEVICE
from .samples import samples


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5 * 10

    sample = samples["NH3-dimer"]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    total_charge = sample["total_charge"].to(**dd)

    qref = sample["q"].to(**dd)
    eref = sample["energy"].to(**dd)

    cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], **dd)
    eeq = charges.ChargeModel.param2019(**dd)
    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)
    tot = torch.sum(qat, -1)

    torch.set_printoptions(precision=16)
    print()
    print(qref)
    print(qat)
    print(qat - qref)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge.cpu(), abs=1e-6) == tot.cpu()
    assert pytest.approx(qref.cpu(), abs=tol) == qat.cpu()
    assert pytest.approx(eref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ghost(dtype: torch.dtype):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples["NH3-dimer"]
    numbers = sample["numbers"].clone().detach().to(DEVICE)
    numbers[[1, 5, 6, 7]] = 0
    positions = sample["positions"].to(**dd)
    total_charge = sample["total_charge"].to(**dd)
    cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], **dd)

    qref = torch.tensor(
        [
            -0.8189238943,
            +0.0000000000,
            +0.2730378155,
            +0.2728482633,
            +0.2730378155,
            +0.0000000000,
            +0.0000000000,
            +0.0000000000,
        ],
        **dd,
    )
    eref = torch.tensor(
        [
            -0.5722096424,
            +0.0000000000,
            +0.1621556977,
            +0.1620431236,
            +0.1621556977,
            +0.0000000000,
            +0.0000000000,
            +0.0000000000,
        ],
        **dd,
    )

    eeq = charges.ChargeModel.param2019(**dd)
    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)
    tot = torch.sum(qat, -1)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge.cpu(), abs=1e-6) == tot.cpu()
    assert pytest.approx(qref.cpu(), abs=tol) == qat.cpu()
    assert pytest.approx(eref.cpu(), abs=tol) == energy.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample1, sample2 = (
        samples["PbH4-BiH3"],
        samples["C6H5I-CH3SH"],
    )
    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    positions = pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )
    total_charge = torch.tensor([0.0, 0.0], **dd)
    eref = pack(
        (
            sample1["energy"].to(**dd),
            sample2["energy"].to(**dd),
        )
    )
    qref = pack(
        (
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        )
    )

    cn = torch.tensor(
        [
            [
                3.9195758978,
                0.9835975866,
                0.9835977083,
                0.9835977083,
                0.9832391350,
                2.9579090955,
                0.9874520816,
                0.9874522118,
                0.9874520816,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
                0.0000000000,
            ],
            [
                3.0173754479,
                3.0134898523,
                3.0173773978,
                3.1580192128,
                3.0178688039,
                3.1573804880,
                1.3525004230,
                0.9943449208,
                0.9943846525,
                0.9942776053,
                0.9943862103,
                0.9942779112,
                2.0535643452,
                0.9956985559,
                3.9585744304,
                0.9940553724,
                0.9939077317,
                0.9939362885,
            ],
        ],
        **dd,
    )
    eeq = charges.ChargeModel.param2019(**dd)
    energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)
    tot = torch.sum(qat, -1)

    assert qat.dtype == energy.dtype == dtype
    assert pytest.approx(total_charge.cpu(), abs=1e-6) == tot.cpu()
    assert pytest.approx(qref.cpu(), abs=tol) == qat.cpu()
    assert pytest.approx(eref.cpu(), abs=tol) == energy.cpu()
