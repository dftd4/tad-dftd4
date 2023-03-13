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
Test calculation of two-body and three-body dispersion terms.
"""
from math import sqrt

import pytest
import torch

from tad_dftd4 import data
from tad_dftd4.charges import get_charges
from tad_dftd4.disp import dftd4, dispersion2
from tad_dftd4.model import D4Model
from tad_dftd4.ncoord import get_coordination_number_d4
from tad_dftd4.utils import pack

from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01", "MB16_43_02"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(name: str, dtype: torch.dtype) -> None:
    single(name, dtype)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["vancoh2"])
def test_single_large(name: str, dtype: torch.dtype) -> None:
    single(name, dtype)


def single(name: str, dtype: torch.dtype) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charge = positions.new_tensor(0.0)
    ref = sample["disp2"].type(dtype)

    # TPSS0-D4-ATM parameters
    param = {
        "s6": positions.new_tensor(1.0),
        "s8": positions.new_tensor(1.85897750),
        "s9": positions.new_tensor(1.0),
        "s10": positions.new_tensor(0.0),
        "alp": positions.new_tensor(16.0),
        "a1": positions.new_tensor(0.44286966),
        "a2": positions.new_tensor(4.60230534),
    }

    r4r2 = data.r4r2[numbers].type(positions.dtype)
    model = D4Model(numbers, device=positions.device, dtype=positions.dtype)
    cn = get_coordination_number_d4(numbers, positions)
    q = get_charges(numbers, positions, charge)
    weights = model.weight_references(cn, q)
    c6 = model.get_atomic_c6(weights)

    energy = dispersion2(numbers, positions, param, c6, r4r2)

    assert energy.dtype == dtype
    assert pytest.approx(ref, abs=tol) == energy


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_s9_zero(name: str, dtype: torch.dtype) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charge = positions.new_tensor(0.0)
    ref = sample["disp2"].type(dtype)

    # TPSS0-D4-ATM parameters
    param = {
        "s8": positions.new_tensor(1.85897750),
        "s9": positions.new_tensor(0.0),  # skip ATM
        "a1": positions.new_tensor(0.44286966),
        "a2": positions.new_tensor(4.60230534),
    }

    energy = dftd4(numbers, positions, charge, param)

    assert energy.dtype == dtype
    assert pytest.approx(ref, abs=tol) == energy


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["SiH4"])
def test_single_s10_one(name: str, dtype: torch.dtype) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charge = positions.new_tensor(0.0)
    ref = torch.tensor(
        [
            -8.8928018057670788e-04,
            -3.3765541880036940e-04,
            -3.3765541880036940e-04,
            -3.3765541880036940e-04,
            -3.3765541880036940e-04,
        ],
        dtype=dtype,
    )

    # TPSS0-D4-ATM parameters
    param = {
        "s8": positions.new_tensor(1.85897750),
        "s9": positions.new_tensor(0.0),  # skip ATM
        "s10": positions.new_tensor(1.0),  # quadrupole-quadrupole
        "a1": positions.new_tensor(0.44286966),
        "a2": positions.new_tensor(4.60230534),
    }

    energy = dftd4(numbers, positions, charge, param)

    assert energy.dtype == dtype
    assert pytest.approx(ref, abs=tol) == energy


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    batch(name1, name2, dtype)


@pytest.mark.large
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["vancoh2"])
def test_batch_large(name1: str, name2: str, dtype: torch.dtype) -> None:
    batch(name1, name2, dtype)


def batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"],
            sample2["numbers"],
        ]
    )

    positions = pack(
        [
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        ]
    )
    charge = positions.new_zeros(numbers.shape[0])
    ref = pack(
        [
            sample1["disp2"].type(dtype),
            sample2["disp2"].type(dtype),
        ]
    )

    # TPSS0-D4-ATM parameters
    param = {
        "s6": positions.new_tensor(1.0),
        "s8": positions.new_tensor(1.85897750),
        "s9": positions.new_tensor(1.0),
        "s10": positions.new_tensor(0.0),
        "alp": positions.new_tensor(16.0),
        "a1": positions.new_tensor(0.44286966),
        "a2": positions.new_tensor(4.60230534),
    }

    r4r2 = data.r4r2[numbers].type(positions.dtype)
    model = D4Model(numbers, device=positions.device, dtype=positions.dtype)
    cn = get_coordination_number_d4(numbers, positions)
    q = get_charges(numbers, positions, charge)
    weights = model.weight_references(cn, q)
    c6 = model.get_atomic_c6(weights)

    energy = dispersion2(numbers, positions, param, c6, r4r2)

    assert energy.dtype == dtype
    assert pytest.approx(ref, abs=tol) == energy
