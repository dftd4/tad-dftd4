"""
Test calculation of DFT-D4 model.
"""

from math import sqrt
import pytest
import torch

from tad_dftd4.charges import get_charges
from tad_dftd4.model import D4Model
from tad_dftd4.ncoord import get_coordination_number_d4
from tad_dftd4.util import pack

from .samples import samples

sample_list = ["MB16_43_01", "MB16_43_02"]
sample_list = ["NO2"]


def gw_gen_single(name: str, dtype: torch.dtype, with_cn: bool, with_q: bool):
    tol = sqrt(torch.finfo(dtype).eps)
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)

    d4 = D4Model()

    if with_cn is True:
        cn = get_coordination_number_d4(numbers, positions)
    else:
        cn = positions.new_zeros(numbers.shape)

    if with_q is True:
        q = get_charges(numbers, positions, positions.new_tensor(0.0))
    else:
        q = positions.new_zeros(numbers.shape)

    print(q)
    print(cn)

    gwvec = d4.weight_references(numbers, cn, q)
    ref = sample["gw"].type(dtype)

    assert gwvec.shape == ref.shape
    assert pytest.approx(gwvec, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float])
def test_mb16_43_01(dtype: torch.dtype) -> None:
    gw_gen_single("MB16_43_01", dtype, with_cn=True, with_q=False)


@pytest.mark.parametrize("dtype", [torch.float])
def test_mb16_43_02(dtype: torch.dtype) -> None:
    gw_gen_single("MB16_43_02", dtype, with_cn=False, with_q=True)


@pytest.mark.parametrize("dtype", [torch.float])
def test_mb16_43_03(dtype: torch.dtype) -> None:
    gw_gen_single("MB16_43_03", dtype, with_cn=True, with_q=True)
