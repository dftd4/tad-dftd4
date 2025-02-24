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
DFT-D4 Model
============

This module contains the definition of the D4 dispersion model for the
evaluation of C6 coefficients.

Upon instantiation, the reference polarizabilities are calculated for the
unique species/elements of the molecule(s) and stored in the model class.


Example
-------
>>> import torch
>>> import tad_dftd4 as d4
>>>
>>> numbers = torch.tensor([14, 1, 1, 1, 1]) # SiH4
>>> model = d4.D4Model(numbers)
>>>
>>> # calculate Gaussian weights, optionally pass CN and partial charges
>>> gw = model.weight_references()
>>> c6 = model.get_atomic_c6(gw)
"""
from __future__ import annotations

import torch

from .. import data, reference
from ..typing import Tensor, Literal, overload
from .base import BaseModel
from .utils import is_exceptional
from tad_mctc.math import einsum

__all__ = ["D4SModel"]


class D4SModel(BaseModel):
    """
    The D4 dispersion model.
    """

    def _get_wf(self) -> Tensor:
        """Pairwise weighting factor."""
        from ..data.wfpair import wfpair

        return wfpair.to(**self.dd)[self.numbers][:, self.numbers]

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        with_dgwdq: Literal[False] = False,
        with_dgwdcn: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        with_dgwdq: Literal[True] = True,
        with_dgwdcn: Literal[False] = False,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        with_dgwdq: Literal[False] = False,
        with_dgwdcn: Literal[True] = True,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        with_dgwdq: Literal[True] = True,
        with_dgwdcn: Literal[True] = True,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        with_dgwdq: bool = False,
        with_dgwdcn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the weights of the reference system
        (shape: ``(..., nat, nref)``).

        Parameters
        ----------
        cn : Tensor | None, optional
            Coordination number of every atom. Defaults to `None` (0).
        q : Tensor | None, optional
            Partial charge of every atom. Defaults to `None` (0).
        with_dgwdq : bool, optional
            Whether to also calculate the derivative of the weights with
            respect to the partial charges. Defaults to `False`.
        with_dgwdcn : bool, optional
            Whether to also calculate the derivative of the weights with
            respect to the coordination numbers. Defaults to `False`.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]
            Weights for the atomic reference systems (shape:
            ``(..., nat, ref)``). If ``with_dgwdq`` is ``True``, also returns
            the derivative of the weights with respect to the partial charges.
            If ``with_dgwdcn`` is ``True``, also returns the derivative of the
            weights with respect to the coordination numbers.
        """
        if cn is None:
            cn = torch.zeros(self.numbers.shape, **self.dd)
        if q is None:
            q = torch.zeros(self.numbers.shape, **self.dd)

        if self.ref_charges == "eeq":
            from ..reference.charge_eeq import clsq as _refq

            refq = _refq.to(**self.dd)[self.numbers]
        elif self.ref_charges == "gfn2":
            from ..reference.charge_gfn2 import refq as _refq

            refq = _refq.to(**self.dd)[self.numbers]
        else:
            raise ValueError(f"Unknown reference charges: {self.ref_charges}")

        zero = torch.tensor(0.0, **self.dd)
        zero_double = torch.tensor(0.0, device=self.device, dtype=torch.double)

        refc = reference.refc.to(self.device)[self.numbers]
        mask = refc > 0

        # Due to the exponentiation, `norm` and `expw` may become very small
        # (down to 1e-300). This causes problems for the division by `norm`,
        # since single precision, i.e. `torch.float`, only goes to around 1e-38.
        # Consequently, some values become zero although the actual result
        # should be close to one. The problem does not arise when using `torch.
        # double`. In order to avoid this error, which is also difficult to
        # detect, this part always uses `torch.double`. `params.refcovcn` is
        # saved with `torch.double`, but I still made sure...
        refcn = reference.refcovcn.to(device=self.device, dtype=torch.double)[
            self.numbers
        ]

        # For vectorization, we reformulate the Gaussian weighting function:
        # exp(-wf * igw * (cn - cn_ref)^2) = [exp(-(cn - cn_ref)^2)]^(wf * igw)
        # Gaussian weighting function part 1: exp(-(cn - cn_ref)^2)
        dcn = cn.unsqueeze(-1).type(torch.double) - refcn
        tmp = torch.exp(einsum("...nr,...nr,nm->...nmr", -dcn, dcn, self.wf))

        print()
        print(tmp)

        # Gaussian weighting function part 2: tmp^(wf * igw)
        # (While the Fortran version just loops over the number of gaussian
        # weights `igw`, we have to use masks and explicitly implement the
        # formulas for exponentiation. Luckily, `igw` only takes on the values
        # 1 and 3.)
        def refc_pow(n: int) -> Tensor:
            return sum(
                (torch.pow(tmp, i) for i in range(1, n + 1)),
                torch.tensor(0.0, device=tmp.device),
            )

        refc_pow_1 = torch.where(refc == 1, refc_pow(1), tmp)
        refc_pow_final = torch.where(refc == 3, refc_pow(3), refc_pow_1)

        expw = torch.where(mask, refc_pow_final, zero_double)

        # Normalize weights, but keep shape. This needs double precision.
        # Moreover, we need to mask the normalization to avoid division by zero
        # for autograd. Strangely, `storch.divide` gives erroneous results for
        # some elements (Mg, e.g. in MB16_43/03).
        norm = torch.where(
            mask,
            torch.sum(expw, dim=-1, keepdim=True),
            torch.tensor(1e-300, device=self.device, dtype=torch.double),
        )
        print("\nnorm")
        print(norm.shape)
        print(norm)
        print("")
        print("")
        # back to real dtype
        gw_temp = (expw / norm).type(self.dtype)

        # maximum reference CN for each atom
        maxcn = torch.max(refcn, dim=-1, keepdim=True)[0]

        # prevent division by 0 and small values
        gw = torch.where(
            is_exceptional(gw_temp, self.dtype),
            torch.where(refcn == maxcn, torch.tensor(1.0, **self.dd), zero),
            gw_temp,
        )

        # print("\ngw")
        # print(gw)
        # print("")
        # print("")

        # unsqueeze for reference dimension
        zeff = data.ZEFF.to(self.device)[self.numbers].unsqueeze(-1)
        gam = data.GAM.to(**self.dd)[self.numbers].unsqueeze(-1) * self.gc
        q = q.unsqueeze(-1)

        # charge scaling
        zeta = torch.where(mask, self._zeta(gam, refq + zeff, q + zeff), zero)
        print(zeta.shape)

        if with_dgwdq is False and with_dgwdcn is False:
            return zeta.unsqueeze(-2) * gw

        # DERIVATIVES

        outputs = [zeta * gw]

        if with_dgwdcn is True:

            def _dpow(n: int) -> Tensor:
                return sum(
                    (
                        2 * i * self.wf * dcn * torch.pow(tmp, i * self.wf)
                        for i in range(1, n + 1)
                    ),
                    zero_double,
                )

            wf_1 = torch.where(refc == 1, _dpow(1), zero_double)
            wf_3 = torch.where(refc == 3, _dpow(3), zero_double)
            dexpw = wf_1 + wf_3

            # no mask needed here, already masked in `dexpw`
            dnorm = torch.sum(dexpw, dim=-1, keepdim=True)

            _dgw = (dexpw - expw * dnorm / norm) / norm
            dgw = torch.where(
                is_exceptional(_dgw, self.dtype), zero, _dgw.type(self.dtype)
            )

            outputs.append(zeta * dgw)

        if with_dgwdq is True:
            dzeta = torch.where(
                mask, self._dzeta(gam, refq + zeff, q + zeff), zero
            )

            outputs.append(dzeta * gw)

        return tuple(outputs)  # type: ignore
