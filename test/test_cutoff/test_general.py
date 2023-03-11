"""
Test the correct handling of types in the `Cutoff` class.
"""
from __future__ import annotations

import pytest
import torch

from tad_dftd4.cutoff import Cutoff

from ..utils import get_device_from_str


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_change_type(dtype: torch.dtype) -> None:
    model = Cutoff().type(dtype)
    assert model.dtype == dtype


def test_change_type_fail() -> None:
    model = Cutoff()

    # trying to use setter
    with pytest.raises(AttributeError):
        model.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        model.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = get_device_from_str(device_str)
    model = Cutoff().to(device)
    assert model.device == device


def test_change_device_fail() -> None:
    model = Cutoff()

    # trying to use setter
    with pytest.raises(AttributeError):
        model.device = "cpu"
