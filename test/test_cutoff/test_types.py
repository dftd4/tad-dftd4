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
from tad_mctc.typing import Tensor

from tad_dftd4 import defaults
from tad_dftd4.cutoff import Cutoff


def test_defaults() -> None:
    cutoff = Cutoff()
    assert pytest.approx(defaults.D4_DISP2_CUTOFF) == cutoff.disp2.cpu()
    assert pytest.approx(defaults.D4_DISP3_CUTOFF) == cutoff.disp3.cpu()
    assert pytest.approx(defaults.D4_CN_CUTOFF) == cutoff.cn.cpu()
    assert pytest.approx(defaults.D4_CN_EEQ_CUTOFF) == cutoff.cn_eeq.cpu()


def test_tensor() -> None:
    tmp = torch.tensor([1.0])
    cutoff = Cutoff(disp2=tmp)

    assert isinstance(cutoff.disp2, Tensor)
    assert isinstance(cutoff.disp3, Tensor)
    assert isinstance(cutoff.cn, Tensor)
    assert isinstance(cutoff.cn_eeq, Tensor)

    assert pytest.approx(tmp.cpu()) == cutoff.disp2.cpu()


@pytest.mark.parametrize("vals", [(1, 2, -3, 4), (1.0, 2.0, 3.0, -4.0)])
def test_int_float(vals: tuple[int | float, ...]) -> None:
    disp2, disp3, cn, cn_eeq = vals
    cutoff = Cutoff(disp2, disp3, cn, cn_eeq)

    assert isinstance(cutoff.disp2, Tensor)
    assert isinstance(cutoff.disp3, Tensor)
    assert isinstance(cutoff.cn, Tensor)
    assert isinstance(cutoff.cn_eeq, Tensor)

    assert pytest.approx(vals[0]) == cutoff.disp2.cpu()
    assert pytest.approx(vals[1]) == cutoff.disp3.cpu()
    assert pytest.approx(vals[2]) == cutoff.cn.cpu()
    assert pytest.approx(vals[3]) == cutoff.cn_eeq.cpu()
