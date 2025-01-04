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
Test the correct handling of types in the `D4Model` class.
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from tad_dftd4.model import D4Model


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])
    model = D4Model(numbers).type(dtype)
    assert model.dtype == dtype


def test_change_type_fail() -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])
    model = D4Model(numbers)

    # trying to use setter
    with pytest.raises(AttributeError):
        model.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        model.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = str_to_device(device_str)
    numbers = torch.tensor([14, 1, 1, 1, 1])
    model = D4Model(numbers).to(device)
    assert model.device == device


def test_change_device_fail() -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])
    model = D4Model(numbers)

    # trying to use setter
    with pytest.raises(AttributeError):
        model.device = torch.device("cpu")


# raise error when creating the model in `_set_refalpha_eeq`
def test_ref_charges_fail() -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])

    with pytest.raises(ValueError):
        D4Model(numbers, ref_charges="wrong")  # type: ignore


# raise error in `weight_references` when trying to change the ref_charges
def test_ref_charges_fail_2() -> None:
    numbers = torch.tensor([14, 1, 1, 1, 1])
    model = D4Model(numbers, ref_charges="eeq")

    model.ref_charges = "wrong"  # type: ignore
    with pytest.raises(ValueError):
        model.weight_references()
