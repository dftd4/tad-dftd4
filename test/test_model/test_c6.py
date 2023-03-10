"""
Test calculation of DFT-D4 model.
"""

import pytest
import torch
import torch.nn.functional as F

from tad_dftd4.model import D4Model
from tad_dftd4.utils import pack

from .samples import samples

# only these references use `cn=True` and `q=True` for `gw`
sample_list = ["LiH", "SiH4", "MB16_43_03"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(name: str, dtype: torch.dtype) -> None:
    tol = 1e-4 if dtype == torch.float else 1e-5
    sample = samples[name]
    numbers = sample["numbers"]
    ref = sample["c6"]

    d4 = D4Model(dtype=dtype)

    # pad reference tensor to always be of shape `(natoms, 7)`
    src = sample["gw"].type(dtype)
    gw = F.pad(
        input=src,
        pad=(0, 0, 0, 7 - src.size(0)),
        mode="constant",
        value=0,
    ).mT

    c6 = d4.get_atomic_c6(numbers, gw)
    assert pytest.approx(ref, rel=tol) == c6


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    tol = 1e-4 if dtype == torch.float else 1e-5
    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"],
            sample2["numbers"],
        ]
    )
    refs = pack(
        [
            sample1["c6"],
            sample2["c6"],
        ]
    )

    d4 = D4Model(dtype=dtype)

    # pad reference tensor to always be of shape `(natoms, 7)`
    src1 = sample1["gw"].type(dtype)
    src2 = sample2["gw"].type(dtype)

    gw = pack(
        [
            F.pad(
                input=src1,
                pad=(0, 0, 0, 7 - src1.size(0)),
                mode="constant",
                value=0,
            ).mT,
            src2.mT,
        ]
    )

    c6 = d4.get_atomic_c6(numbers, gw)
    assert pytest.approx(refs, rel=tol) == c6
