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
Sanity check for parameters since they are created from the Fortran parameters
with a script.
"""
from __future__ import annotations

import torch

from tad_dftd4 import data, params


def test_params_shape() -> None:
    maxel = 104  # 103 elements + dummy
    assert params.refc.shape == torch.Size((maxel, 7))
    assert params.refascale.shape == torch.Size((maxel, 7))
    assert params.refcn.shape == torch.Size((maxel, 7))
    assert params.refsys.shape == torch.Size((maxel, 7))
    assert params.refq.shape == torch.Size((maxel, 7))
    assert params.refalpha.shape == torch.Size((maxel, 7, 23))


def test_data_shape() -> None:
    assert data.gam.shape == torch.Size((119,))
    assert data.pauling_en.shape == torch.Size((119,))
    assert data.cov_rad_d3.shape == torch.Size((119,))
    assert data.r4r2.shape == torch.Size((119,))
