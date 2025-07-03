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
from tad_mctc.ncoord import cn_d4
from tad_mctc.typing import Any, CNFunction, Literal

from ..damping import RationalDamping, ZeroDamping
from .base import Disp
from .threebody import ATM, C9ApproxMixin, C9ExactMixin, RadiiBJMixin
from .twobody import TwoBodyTerm

__all__ = ["DispD4"]


class D4ATMApprox(C9ApproxMixin, RadiiBJMixin, ATM):
    """D4 ATM: approximate C9 + Becke-Johnson radii."""


class DispD4(Disp):
    """Standard DFT-D4 dispersion method."""

    ALLOWED_MODELS = ("d4", "d4s")

    def __init__(
        self,
        model: Literal["d4", "d4s"] = "d4",
        model_kwargs: dict[str, Any] | None = None,
        cn_fn: CNFunction = cn_d4,
        cn_fn_kwargs: dict[str, Any] | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            model=model,
            model_kwargs=model_kwargs,
            cn_fn=cn_fn,
            cn_fn_kwargs=cn_fn_kwargs,
            device=device,
            dtype=dtype,
        )

        # D4(BJ)-ATM(approx)
        super().register(
            TwoBodyTerm(damping_fn=RationalDamping(), charge_dependent=True)
        )
        super().register(
            D4ATMApprox(damping_fn=ZeroDamping(), charge_dependent=False)
        )


##############################################################################


class D4ATMExact(C9ExactMixin, RadiiBJMixin, ATM):
    """D4 ATM: approximate C9 + Becke-Johnson radii."""


class DispD4Exact(Disp):
    """DFT-D4 with exact C9 coefficients via Casimir--Polder formula."""

    ALLOWED_MODELS = ("d4", "d4s")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # D4(BJ)-ATM(exact)
        super().register(
            TwoBodyTerm(damping_fn=RationalDamping(), charge_dependent=True)
        )
        super().register(
            D4ATMExact(damping_fn=ZeroDamping(), charge_dependent=False)
        )
