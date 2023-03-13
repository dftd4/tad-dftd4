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
Test the correct handling of types in the `Cutoff` class.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4 import defaults
from tad_dftd4._typing import Tensor
from tad_dftd4.cutoff import Cutoff


def test_defaults():
    cutoff = Cutoff()
    assert pytest.approx(defaults.D4_DISP2_CUTOFF) == cutoff.disp2
    assert pytest.approx(defaults.D4_DISP3_CUTOFF) == cutoff.disp3
    assert pytest.approx(defaults.D4_CN_CUTOFF) == cutoff.cn
    assert pytest.approx(defaults.D4_CN_EEQ_CUTOFF) == cutoff.cn_eeq


def test_tensor():
    tmp = torch.randn(1)
    cutoff = Cutoff(disp2=tmp)

    assert isinstance(cutoff.disp2, Tensor)
    assert isinstance(cutoff.disp3, Tensor)
    assert isinstance(cutoff.cn, Tensor)
    assert isinstance(cutoff.cn_eeq, Tensor)

    assert pytest.approx(tmp) == cutoff.disp2


@pytest.mark.parametrize("vals", [(1, 2, -3, 4), (1.0, 2.0, 3.0, -4.0)])
def test_int_float(vals: tuple[int | float, ...]):
    disp2, disp3, cn, cn_eeq = vals
    cutoff = Cutoff(disp2, disp3, cn, cn_eeq)

    assert isinstance(cutoff.disp2, Tensor)
    assert isinstance(cutoff.disp3, Tensor)
    assert isinstance(cutoff.cn, Tensor)
    assert isinstance(cutoff.cn_eeq, Tensor)

    assert pytest.approx(vals[0]) == cutoff.disp2
    assert pytest.approx(vals[1]) == cutoff.disp3
    assert pytest.approx(vals[2]) == cutoff.cn
    assert pytest.approx(vals[3]) == cutoff.cn_eeq
