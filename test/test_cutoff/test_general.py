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
Test the correct handling of types in the `Cutoff` class.
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from tad_dftd4.cutoff import Cutoff


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    cutoff = Cutoff().type(dtype)
    assert cutoff.dtype == dtype
    assert cutoff.disp2.dtype == dtype
    assert cutoff.disp3.dtype == dtype
    assert cutoff.cn.dtype == dtype
    assert cutoff.cn_eeq.dtype == dtype


def test_change_type_fail() -> None:
    cutoff = Cutoff()

    # trying to use setter
    with pytest.raises(AttributeError):
        cutoff.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        cutoff.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = str_to_device(device_str)
    cutoff = Cutoff().to(device)
    assert cutoff.device == device
    assert cutoff.disp2.device == device
    assert cutoff.disp3.device == device
    assert cutoff.cn.device == device
    assert cutoff.cn_eeq.device == device


def test_change_device_fail() -> None:
    cutoff = Cutoff()

    # trying to use setter
    with pytest.raises(AttributeError):
        cutoff.device = torch.device("cpu")
