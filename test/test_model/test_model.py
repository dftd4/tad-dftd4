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


@pytest.mark.parametrize("name", ["LiH", "SiH4"])
def test_single_float(name: str) -> None:
    single(name, torch.float, tol=1e-5)


@pytest.mark.xfail
@pytest.mark.parametrize("name", ["MB16_43_03"])
def test_single_float_fail(name: str) -> None:
    single(name, torch.float, tol=1e-5)


@pytest.mark.parametrize("name", sample_list)
def test_single_double(name: str) -> None:
    single(name, torch.double, tol=1e-5)


def single(name: str, dtype: torch.dtype, tol: float) -> None:
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = sample["c6"]

    d4 = D4Model(dtype=dtype)

    cn = get_coordination_number_d4(numbers, positions)
    q = get_charges(numbers, positions, positions.new_tensor(0.0))

    gw = d4.weight_references(numbers, cn=cn, q=q)
    c6 = d4.get_atomic_c6(numbers, gw)
    assert pytest.approx(ref, abs=tol, rel=tol) == c6


@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4"])
def test_batch_float(name1: str, name2: str) -> None:
    batch(name1, name2, torch.float, tol=1e-5)


@pytest.mark.xfail
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["MB16_43_03"])
def test_batch_float_fail(name1: str, name2: str) -> None:
    batch(name1, name2, torch.float, tol=1e-5)


@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch_double(name1: str, name2: str) -> None:
    batch(name1, name2, torch.double, tol=1e-5)


def batch(name1: str, name2: str, dtype: torch.dtype, tol: float) -> None:
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

    d4 = D4Model(dtype=dtype)

    cn = get_coordination_number_d4(numbers, positions)
    q = get_charges(numbers, positions, positions.new_zeros(numbers.shape[0]))

    gw = d4.weight_references(numbers, cn=cn, q=q)
    c6 = d4.get_atomic_c6(numbers, gw)
    assert pytest.approx(refs, abs=tol, rel=tol) == c6
