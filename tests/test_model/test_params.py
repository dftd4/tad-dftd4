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
