# This file is part of tad-dftd4.
#
# SPDX-Identifier: LGPL-3.0
# Copyright (C) 2022 Marvin Friede
#
# tad-dftd4 is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tad-dftd4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tad-dftd4. If not, see <https://www.gnu.org/licenses/>.
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

from tad_dftd4 import dftd4, utils
from tad_dftd4._typing import DD, Tensor

from ..utils import reshape_fortran
from .samples_hessian import samples

sample_list = ["LiH", "SiH4", "PbH4-BiH3", "MB16_43_01"]

tol = 1e-8

device = None


def test_fail() -> None:
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"]
    param = {"a1": numbers}

    # differentiable variable is not a tensor
    with pytest.raises(RuntimeError):
        utils.hessian(dftd4, (numbers, positions, param), argnums=2)


def test_zeros() -> None:
    d = torch.randn(2, 3, requires_grad=True)
    zeros = torch.zeros([*d.shape, *d.shape])

    def dummy(x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    hess = utils.hessian(dummy, (d,), argnums=0)
    assert pytest.approx(zeros.cpu()) == hess.detach().cpu()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
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
    print(ref)

    # variable to be differentiated
    positions.requires_grad_(True)

    hess = utils.hessian(dftd4, (numbers, positions, charge, param), argnums=1)
    positions.detach_()

    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()


# TODO: Figure out batched Hessian computation
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
def skip_test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = utils.pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = utils.pack(
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

    ref = utils.pack(
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

    # variable to be differentiated
    positions.requires_grad_(True)

    hess = utils.hessian(dftd4, (numbers, positions, charge, param), argnums=1)
    positions.detach_()

    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()
