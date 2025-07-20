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
Sanity check for parameters since they are created from the Fortran parameters
with a script.
"""
from __future__ import annotations

import torch

from tad_dftd4 import data
from tad_dftd4.reference import d4 as d4ref
from tad_dftd4.reference.d4 import charge_eeq, charge_gfn2


def test_params_shape() -> None:
    maxel = 104  # 103 elements + dummy
    assert d4ref.refc.shape == torch.Size((maxel, 7))
    assert d4ref.refascale.shape == torch.Size((maxel, 7))
    assert d4ref.refcovcn.shape == torch.Size((maxel, 7))
    assert d4ref.refsys.shape == torch.Size((maxel, 7))
    assert d4ref.refalpha.shape == torch.Size((maxel, 7, 23))

    assert charge_eeq.clsq.shape == torch.Size((maxel, 7))
    assert charge_eeq.clsh.shape == torch.Size((maxel, 7))

    # GFN2 charges only up to Rn
    assert charge_gfn2.refq.shape == torch.Size((87, 7))
    assert charge_gfn2.refh.shape == torch.Size((87, 7))


def test_data_shape() -> None:
    assert data.GAM().shape == torch.Size((119,))
    assert data.R4R2().shape == torch.Size((119,))
