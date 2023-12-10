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
Testing dispersion gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4 import dftd4, utils
from tad_dftd4._typing import Callable, Tensor, DD

from ..molecules import mols as samples
from ..utils import dgradcheck, dgradgradcheck

sample_list = ["LiH", "AmF3", "SiH4"]

tol = 1e-8

device = None


def gradchecker(
    dtype: torch.dtype, name: str
) -> tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
]:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device=device)
    positions = sample["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    # TPSS0-D4-ATM parameters, variables to be differentiated
    param = (
        torch.tensor(1.00000000, **dd, requires_grad=True),
        torch.tensor(0.78981345, **dd, requires_grad=True),
        torch.tensor(1.00000000, **dd, requires_grad=True),
        torch.tensor(0.00000000, **dd, requires_grad=True),  # s10
        torch.tensor(0.49484001, **dd, requires_grad=True),
        torch.tensor(5.73083694, **dd, requires_grad=True),
        torch.tensor(16.0000000, **dd, requires_grad=True),
    )
    label = ("s6", "s8", "s9", "s10", "a1", "a2", "alp")

    def func(*inputs: Tensor) -> Tensor:
        input_param = {label[i]: input for i, input in enumerate(inputs)}
        return dftd4(numbers, positions, charge, input_param)

    return func, param


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_gradgradcheck(dtype: torch.dtype, name: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", ["MB16_43_01"])
def test_gradgradcheck_slow(dtype: torch.dtype, name: str) -> None:
    """
    These fail with `fast_mode=True`.
    """
    func, diffvars = gradchecker(dtype, name)
    assert dgradgradcheck(func, diffvars, atol=1e-6, fast_mode=False)


def gradchecker_batch(
    dtype: torch.dtype, name1: str, name2: str
) -> tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],  # autograd function
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
]:
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = utils.pack(
        [
            sample1["numbers"].to(device=device),
            sample2["numbers"].to(device=device),
        ]
    )
    positions = utils.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    # TPSS0-D4-ATM parameters, variables to be differentiated
    param = (
        torch.tensor(1.00000000, **dd, requires_grad=True),
        torch.tensor(0.78981345, **dd, requires_grad=True),
        torch.tensor(1.00000000, **dd, requires_grad=True),
        torch.tensor(0.00000000, **dd, requires_grad=True),  # s10
        torch.tensor(0.49484001, **dd, requires_grad=True),
        torch.tensor(5.73083694, **dd, requires_grad=True),
        torch.tensor(16.0000000, **dd, requires_grad=True),
    )
    label = ("s6", "s8", "s9", "s10", "a1", "a2", "alp")

    def func(*inputs: Tensor) -> Tensor:
        input_param = {label[i]: input for i, input in enumerate(inputs)}
        return dftd4(numbers, positions, charge, input_param)

    return func, param


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_gradgradcheck_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    Check a single analytical gradient of parameters against numerical
    gradient from `torch.autograd.gradgradcheck`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=tol)


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["MB16_43_01"])
def test_gradgradcheck_batch_slow(dtype: torch.dtype, name1: str, name2: str) -> None:
    """
    These fail with `fast_mode=True`.
    """
    func, diffvars = gradchecker_batch(dtype, name1, name2)
    assert dgradgradcheck(func, diffvars, atol=1e-6, fast_mode=False)
