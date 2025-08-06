# This file is part of tad-dftd4.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Testing dispersion Hessian (autodiff).

The reference values are calculated with the dftd4 standalone (Fortran) program,
version 3.6.0. However, some minor modifications are required to obtained a
compatible array ordering from Fortran. In Fortran, the shape of the Hessian is
`(3, mol%nat, 3, mol%nat)`, which we change to `(mol%nat, 3, mol%nat, 3)`.
Correspondingly, the calculation in `get_dispersion_hessian` must also be
adapted: We replace `hessian(:, :, ix, iat) = (gl - gr) / (2 * step)` by
`hessian(:, :, iat, ix) = (transpose(gl) - transpose(gr)) / (2 * step)`. The
Hessian can then simply be printed via `write(*, '(SP,es23.16e2,",")') hessian`
and the Python resorting is handled by the reshape function.
"""
from __future__ import annotations

import pytest
import torch
from tad_mctc._version import __tversion__
from tad_mctc.autograd import hess_fn_rev, hessian
from tad_mctc.batch import pack
from tad_mctc.convert import reshape_fortran
from tad_mctc.typing import DD, Tensor

from tad_dftd4 import dftd4

from ..conftest import DEVICE
from .samples_hessian import samples

sample_list = ["LiH", "SiH4", "PbH4-BiH3", "MB16_43_01"]

tol = 1e-7


def test_fail() -> None:
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"]
    param = {"a1": numbers}

    # differentiable variable is not a tensor
    with pytest.raises(ValueError):
        hessian(dftd4, (numbers, positions, param), argnums=2)


def test_zeros() -> None:
    d = torch.randn(2, 3, requires_grad=True)
    zeros = torch.zeros([*d.shape, *d.shape])

    def dummy(x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    hess = hessian(dummy, (d,), argnums=0)
    assert pytest.approx(zeros.cpu()) == hess.detach().cpu()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    # TPSS0-ATM parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(1.62438102, **dd),
        "s9": torch.tensor(1.00000000, **dd),
        "a1": torch.tensor(0.40329022, **dd),
        "a2": torch.tensor(4.80537871, **dd),
    }

    ref = reshape_fortran(
        sample["hessian"].to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    # variable to be differentiated
    positions.requires_grad_(True)

    hess = hessian(dftd4, (numbers, positions, charge, param), argnums=1)
    positions.detach_()

    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires PyTorch>=2.0.0")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_v2(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    charge = torch.tensor(0.0, **dd)

    # TPSS0-ATM parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(1.62438102, **dd),
        "s9": torch.tensor(1.00000000, **dd),
        "a1": torch.tensor(0.40329022, **dd),
        "a2": torch.tensor(4.80537871, **dd),
    }

    ref = reshape_fortran(
        sample["hessian"].to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )

    def _energy(num: Tensor, pos: Tensor, chrg: Tensor) -> Tensor:
        """
        Closure over non-tensor argument `param` for `dftd4` function.

        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return dftd4(num, pos, chrg, param).sum(-1)

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    hess = hess_fn_rev(_energy, argnums=1)(numbers, pos, charge)
    assert isinstance(hess, Tensor)
    hess = hess.detach().cpu()

    assert hess.shape == (numbers.shape[-1], 3, numbers.shape[-1], 3)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess

    pos.detach_()


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires PyTorch>=2.0.0")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def test_batch_d4(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    # TPSS0-ATM parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(1.62438102, **dd),
        "s9": torch.tensor(1.00000000, **dd),
        "a1": torch.tensor(0.40329022, **dd),
        "a2": torch.tensor(4.80537871, **dd),
    }

    ref = pack(
        [
            reshape_fortran(
                sample1["hessian"].to(**dd),
                torch.Size(2 * (sample1["numbers"].shape[-1], 3)),
            ),
            reshape_fortran(
                sample2["hessian"].to(**dd),
                torch.Size(2 * (sample2["numbers"].shape[-1], 3)),
            ),
        ]
    )

    def _energy(num: Tensor, pos: Tensor, chrg: Tensor) -> Tensor:
        """
        Closure over non-tensor argument `param` for `dftd4` function.

        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return dftd4(num, pos, chrg, param, model="d4").sum(-1)

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    hess_fn = hess_fn_rev(_energy, argnums=1)
    hess_fn_batch = torch.func.vmap(hess_fn, in_dims=(0, 0, 0))

    hess = hess_fn_batch(numbers, pos, charge)
    assert isinstance(hess, Tensor)
    hess = hess.detach().cpu()

    assert hess.shape == (2, numbers.shape[-1], 3, numbers.shape[-1], 3)
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess

    pos.detach_()


@pytest.mark.skipif(__tversion__ < (2, 0, 0), reason="Requires PyTorch>=2.0.0")
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", ["LiH", "SiH4"])
def test_batch_d4s(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )
    charge = torch.tensor([0.0, 0.0], **dd)

    # TPSS0-ATM parameters
    param = {
        "s6": torch.tensor(1.00000000, **dd),
        "s8": torch.tensor(1.62438102, **dd),
        "s9": torch.tensor(1.00000000, **dd),
        "a1": torch.tensor(0.40329022, **dd),
        "a2": torch.tensor(4.80537871, **dd),
    }

    def _energy(num: Tensor, pos: Tensor, chrg: Tensor) -> Tensor:
        """
        Closure over non-tensor argument `param` for `dftd4` function.

        Returns energy as scalar, which is required for Hessian computation
        to obtain the correct shape of ``(..., nat, 3, nat, 3)``.
        """
        return dftd4(num, pos, chrg, param, model="d4s").sum(-1)

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    hess_fn = hess_fn_rev(_energy, argnums=1)
    hess_fn_batch = torch.func.vmap(hess_fn, in_dims=(0, 0, 0))

    hess = hess_fn_batch(numbers, pos, charge)
    assert isinstance(hess, Tensor)
    hess = hess.detach().cpu()

    assert hess.shape == (2, numbers.shape[-1], 3, numbers.shape[-1], 3)

    # numerical Hessian for comparison (use `positions`, not `pos`!)
    num_hess = calc_num_hessian_batch(
        numbers, positions, charge, param, model="d4s"
    )
    assert num_hess.shape == hess.shape

    assert pytest.approx(num_hess.cpu(), abs=tol, rel=tol) == hess

    pos.detach_()


def calc_num_hessian(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    param: dict[str, Tensor],
    model: str,
    step: float = 5.0e-4,  # sensitive!
) -> Tensor:
    """
    Numerically approximate the full Hessian of the energy with respect to
    atomic positions via 4-point central-difference.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(nat, )``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(nat, 3)``).
    charge : Tensor
        Total charge of the system (shape: ``(, )``).
    step : float, optional
        Step size for numerical differentiation, by default 5.0e-4.

    Returns
    -------
    Tensor
        Tensor of second derivatives (shape: ``(nat, 3, nat, 3)``)
    """
    # assume single-sample (no leading batch dims) for simplicity
    nat = positions.shape[-2]

    def _energy(pos: Tensor) -> Tensor:
        return dftd4(numbers, pos, charge, param, model=model).sum(-1)

    H_pqrs = []
    for p in range(nat):
        H_qrs = []
        for q in range(3):
            H_rs = []
            # prebuild the (p,q) kick
            e_pq = torch.zeros_like(positions)
            e_pq[p, q] = step

            for r in range(nat):
                row = []
                for s in range(3):
                    e_rs = torch.zeros_like(positions)
                    e_rs[r, s] = step

                    E_pp = _energy(positions + e_pq + e_rs)
                    E_pm = _energy(positions + e_pq - e_rs)
                    E_mp = _energy(positions - e_pq + e_rs)
                    E_mm = _energy(positions - e_pq - e_rs)

                    val = (E_pp - E_pm - E_mp + E_mm) / (4 * step * step)
                    row.append(val)
                # now one row of shape (3,)
                H_rs.append(torch.stack(row))
            # stack over r -> shape (nat, 3)
            H_qrs.append(torch.stack(H_rs, dim=0))
        # stack over q -> shape (3, nat, 3)
        H_pqrs.append(torch.stack(H_qrs, dim=0))
    # stack over p -> shape (nat, 3, nat, 3)
    return torch.stack(H_pqrs, dim=0)


def calc_num_hessian_batch(
    numbers: Tensor,
    positions: Tensor,
    charge: Tensor,
    param: dict[str, Tensor],
    model: str,
    step: float = 5.0e-4,  # sensitive!
) -> Tensor:
    """
    Compute a batch of Hessians.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system of shape ``(..., nat)``.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    charge : Tensor
        Total charge of the system (shape: ``(..., )``).
    step : float, optional
        Step size for numerical differentiation, by default 5.0e-4.

    Returns
    -------
    Tensor
        Numerical Hessian for each system in the batch of shape
        ``(..., nat, 3, nat, 3)``.
    """

    def _calc_num_hessian(nums: Tensor, pos: Tensor, ch: Tensor) -> Tensor:
        """Calculate the numerical Hessian for a single system."""
        return calc_num_hessian(nums, pos, ch, param, model=model, step=step)

    # vmap over axis=0 of each input
    hess_fn = torch.func.vmap(_calc_num_hessian, in_dims=(0, 0, 0), out_dims=0)
    return hess_fn(numbers, positions, charge)
