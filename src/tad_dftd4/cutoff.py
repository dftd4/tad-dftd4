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
Cutoff
======

Real-space cutoffs for the two-body and three-body dispersion energy
as well as the coordination number within D4 and the EEQ Model.
"""
from __future__ import annotations

import torch
from tad_mctc.convert import any_to_tensor
from tad_mctc.typing import Tensor, TensorLike

from . import defaults

__all__ = ["Cutoff"]


class Cutoff(TensorLike):
    """
    Collection of real-space cutoffs.
    """

    disp2: Tensor
    """
    Two-body interaction cutoff.

    :default: `60.0`
    """

    disp3: Tensor
    """
    Three-body interaction cutoff.

    :default: `40.0`
    """

    cn: Tensor
    """
    Coordination number cutoff.

    :default: `30.0`
    """

    cn_eeq: Tensor
    """
    Coordination number cutoff within EEQ.

    :default: `25.0`
    """

    __slots__ = ("disp2", "disp3", "cn", "cn_eeq")

    def __init__(
        self,
        disp2: int | float | Tensor = defaults.D4_DISP2_CUTOFF,
        disp3: int | float | Tensor = defaults.D4_DISP3_CUTOFF,
        cn: int | float | Tensor = defaults.D4_CN_CUTOFF,
        cn_eeq: int | float | Tensor = defaults.D4_CN_EEQ_CUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        disp2 = any_to_tensor(disp2, device=device, dtype=dtype)
        disp3 = any_to_tensor(disp3, device=device, dtype=dtype)
        cn = any_to_tensor(cn, device=device, dtype=dtype)
        cn_eeq = any_to_tensor(cn_eeq, device=device, dtype=dtype)

        self.disp2 = disp2
        self.disp3 = disp3
        self.cn = cn
        self.cn_eeq = cn_eeq
