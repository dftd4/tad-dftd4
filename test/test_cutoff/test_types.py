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
