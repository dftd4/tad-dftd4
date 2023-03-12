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
