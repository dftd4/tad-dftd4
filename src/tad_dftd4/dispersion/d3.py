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
Dispersion Methods: D4
======================

DFT-D4 dispersion method implementation.
"""
from __future__ import annotations

import torch
from tad_mctc.ncoord import cn_d3
from tad_mctc.typing import CNFunction

from ..damping import Damping, RationalDamping, ZeroDamping
from ..typing import Tensor
from .base import Disp
from .threebody import ATM, C9ApproxMixin, RadiiVDWMixin
from .twobody import TwoBodyTerm


class D3ATM(C9ApproxMixin, RadiiVDWMixin, ATM):
    """D3 ATM: approximate C9 + pair-wise VDW radii."""


class DispD3(Disp):
    """Standard DFT-D4 dispersion method."""

    def __init__(
        self,
        cn_fn: CNFunction = cn_d3,
        damping_fn_2: Damping = RationalDamping(),
        damping_fn_3: Damping = ZeroDamping(),
        charge_dependent_2: bool = False,
        charge_dependent_3: bool = False,
    ):
        super().__init__(cn_fn=cn_fn, model="d3")

        # D3(BJ)-ATM
        super().register(
            TwoBodyTerm(
                damping_fn=damping_fn_2, charge_dependent=charge_dependent_2
            )
        )
        super().register(
            D3ATM(damping_fn=damping_fn_3, charge_dependent=charge_dependent_3)
        )

    # def _default_model(
    #     self,
    #     numbers: Tensor,
    #     device: torch.device | None = None,
    #     dtype: torch.dtype | None = None,
    # ) -> D4Model:
    #     return D4Model(numbers=numbers, device=device, dtype=dtype)

    def _default_rcov(
        self,
        numbers: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        # pylint: disable=import-outside-toplevel
        from tad_mctc.data import COV_D3

        return COV_D3(device=device, dtype=dtype)[numbers]

    def _default_r4r2(
        self,
        numbers: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        # pylint: disable=import-outside-toplevel
        from ..data import R4R2

        return R4R2(device=device, dtype=dtype)[numbers]
