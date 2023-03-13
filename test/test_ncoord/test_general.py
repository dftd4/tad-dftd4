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
Test error handling in coordination number calculation.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4._typing import Any, CountingFunction, Protocol, Tensor
from tad_dftd4.ncoord import (
    erf_count,
    exp_count,
    get_coordination_number_d4,
    get_coordination_number_eeq,
)


class CNFunction(Protocol):
    """
    Type annotation for coordination number function.
    """

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        counting_function: CountingFunction = erf_count,
        rcov: Tensor | None = None,
        en: Tensor | None = None,
        cutoff: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        ...


@pytest.mark.parametrize(
    "function",
    [get_coordination_number_d4, get_coordination_number_eeq],
)
@pytest.mark.parametrize(
    "counting_function",
    [erf_count, exp_count],
)
def test_fail(function: CNFunction, counting_function: CountingFunction) -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0])
        function(numbers, positions, counting_function, rcov)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1])
        function(numbers, positions, counting_function)
