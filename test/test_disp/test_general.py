"""
Check shape of tensors.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4.disp import dftd4


def test_fail() -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    charge = torch.tensor(0.0)
    param = {"s6": torch.tensor(1.0)}

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0])
        dftd4(numbers, positions, charge, param, rcov=rcov)

    # expectation valus (r4r2) wrong shape
    with pytest.raises(ValueError):
        r4r2 = torch.tensor([1.0])
        dftd4(numbers, positions, charge, param, r4r2=r4r2)

    # atomic partial charges wrong shape
    with pytest.raises(ValueError):
        q = torch.tensor([1.0])
        dftd4(numbers, positions, charge, param, q=q)

    # wrong numbers (give charges, otherwise test fails in EEQ, not in disp)
    with pytest.raises(ValueError):
        q = torch.tensor([0.5, -0.5])
        nums = torch.tensor([1])
        dftd4(nums, positions, charge, param, q=q)
