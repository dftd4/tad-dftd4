"""
Test calculation of DFT-D4 model.
"""

import pytest
import torch

from tad_dftd4.charges import get_charges
from tad_dftd4.model import D4Model
from tad_dftd4.ncoord import get_coordination_number_d4
from tad_dftd4.utils import pack

from .samples import samples

# only these references use `cn=True` and `q=True` for `gw`
sample_list = ["LiH", "SiH4", "MB16_43_03"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(name: str, dtype: torch.dtype) -> None:
    tol = 1e-5
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["c6"]

    d4 = D4Model(numbers, dtype=dtype)

    cn = get_coordination_number_d4(numbers, positions)
    q = get_charges(numbers, positions, positions.new_tensor(0.0))

    gw = d4.weight_references(cn=cn, q=q)
    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(ref, abs=tol, rel=tol) == c6


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch(name1: str, name2: str, dtype: torch.dtype) -> None:
    tol = 1e-5
    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"],
            sample2["numbers"],
        ]
    )
    positions = pack(
        [
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        ]
    )
    refs = pack(
        [
            sample1["c6"],
            sample2["c6"],
        ]
    )

    d4 = D4Model(numbers, dtype=dtype)

    cn = get_coordination_number_d4(numbers, positions)
    q = get_charges(numbers, positions, positions.new_zeros(numbers.shape[0]))

    gw = d4.weight_references(cn=cn, q=q)
    c6 = d4.get_atomic_c6(gw)
    assert pytest.approx(refs, abs=tol, rel=tol) == c6
