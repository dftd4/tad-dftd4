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
from tad_mctc import storch
from tad_mctc.math import einsum

from . import data, reference
from .typing import Literal, Tensor, TensorLike, overload

__all__ = ["D4Model"]


ga_default = 3.0
gc_default = 2.0
wf_default = 6.0


class D4Model(TensorLike):
    """
    The D4 dispersion model.
    """

    numbers: Tensor
    """Atomic numbers of all atoms in the system."""

    ga: float
    """Maximum charge scaling height for partial charge extrapolation."""

    gc: float
    """Charge scaling steepness for partial charge extrapolation."""

    wf: float
    """Weighting factor for coordination number interpolation."""

    ref_charges: Literal["eeq", "gfn2"]
    """Reference charges to use for the model."""

    alpha: Tensor
    """Reference polarizabilities of unique species."""

    __slots__ = ("numbers", "ga", "gc", "wf", "ref_charges", "rc6")

    def __init__(
        self,
        numbers: Tensor,
        ga: float = ga_default,
        gc: float = gc_default,
        wf: float = wf_default,
        ref_charges: Literal["eeq", "gfn2"] = "eeq",
        rc6: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Instantiate `D4Model`.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms in the system.
        ga : float, optional
            Maximum charge scaling height for partial charge extrapolation.
            Defaults to `ga_default`.
        gc : float, optional
            Charge scaling steepness for partial charge extrapolation.
            Defaults to `gc_default`.
        wf : float, optional
            Weighting factor for coordination number interpolation.
            Defaults to `wf_default`.
        ref_charges : Literal["eeq", "gfn2"], optional
            Reference charges to use for the model. Defaults to `"eeq"`.
        rc6 : Tensor | None, optional
            Reference C6 coefficients of unique species. Defaults to `None`.
        device : torch.device | None, optional
            Pytorch device for calculations. Defaults to `None`.
        dtype : torch.dtype | None, optional
            Pytorch dtype for calculations. Defaults to `None`.
        """
        super().__init__(device, dtype)
        self.numbers = numbers

        self.ga = ga
        self.gc = gc
        self.wf = wf
        self.ref_charges = ref_charges

        if rc6 is None:
            self.rc6 = self.get_refc6()

    @property
    def unique(self) -> Tensor:
        """
        Unique species (elements) in molecule(s). Sorted in ascending order.

        Returns
        -------
        Tensor
            Unique species within `D4Model.numbers`.
        """
        return torch.unique(self.numbers)

    @property
    def atom_to_unique(self) -> Tensor:
        """
        Mapping of atoms to unique species.

        Returns
        -------
        Tensor
            Mapping of atoms (`D4Model.numbers`) to unique species.
        """
        return torch.unique(self.numbers, return_inverse=True)[1]

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
        Calculate the weights of the reference system.

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
            Weights for the atomic reference systems. If `with_dgwdq` is `True`,
            also returns the derivative of the weights with respect to the
            partial charges. If `with_dgwdcn` is `True`, also returns the
            derivative of the weights with respect to the coordination numbers.
        """
        if cn is None:
            cn = torch.zeros(self.numbers.shape, **self.dd)
        if q is None:
            q = torch.zeros(self.numbers.shape, **self.dd)

        if self.ref_charges == "eeq":
            from .reference.charge_eeq import clsq as _refq

            refq = _refq.to(**self.dd)[self.numbers]
        elif self.ref_charges == "gfn2":
            from .reference.charge_gfn2 import refq as _refq

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
        tmp = torch.exp(-dcn * dcn)

        # Gaussian weighting function part 2: tmp^(wf * igw)
        # (While the Fortran version just loops over the number of gaussian
        # weights `igw`, we have to use masks and explicitly implement the
        # formulas for exponentiation. Luckily, `igw` only takes on the values
        # 1 and 3.)
        def refc_pow(n: int) -> Tensor:
            return sum(
                (torch.pow(tmp, i * self.wf) for i in range(1, n + 1)),
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

        # unsqueeze for reference dimension
        zeff = data.ZEFF.to(self.device)[self.numbers].unsqueeze(-1)
        gam = data.GAM.to(**self.dd)[self.numbers].unsqueeze(-1) * self.gc
        q = q.unsqueeze(-1)

        # charge scaling
        zeta = torch.where(mask, self._zeta(gam, refq + zeff, q + zeff), zero)

        if with_dgwdq is False and with_dgwdcn is False:
            return zeta * gw

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

    def get_atomic_c6(self, gw: Tensor) -> Tensor:
        """
        Calculate atomic C6 dispersion coefficients.

        Parameters
        ----------
        gw : Tensor
            Weights for the atomic reference systems of shape
            `(..., nat, nref)`.

        Returns
        -------
        Tensor
            C6 coefficients for all atom pairs of shape `(..., nat, nat)`.
        """
        # The default einsum path is fastest if the large tensors comes first.
        # (..., n1, n2, r1, r2) * (..., n1, r1) * (..., n2, r2) -> (..., n1, n2)
        return einsum(
            "...ijab,...ia,...jb->...ij",
            *(self.rc6, gw, gw),
            optimize=[(0, 1), (0, 1)],
        )

        # NOTE: This old version creates large intermediate tensors and builds
        # the full matrix before the sum reduction, requiring a lot of memory.
        #
        # (..., 1, n, 1, r) * (..., n, 1, r, 1) = (..., n, n, r, r)
        # g = gw.unsqueeze(-3).unsqueeze(-2) * gw.unsqueeze(-2).unsqueeze(-1)
        #
        # (..., n, n, r, r) * (..., n, n, r, r) -> (..., n, n)
        # c6 = torch.sum(g * self.rc6, dim=(-2, -1))

    def _zeta(self, gam: Tensor, qref: Tensor, qmod: Tensor) -> Tensor:
        """
        Charge scaling function.

        Parameters
        ----------
        gam : Tensor
            Chemical hardness.
        qref : Tensor
            Reference charges.
        qmod : Tensor
            Modified charges.

        Returns
        -------
        Tensor
            Scaled charges.
        """
        eps = torch.tensor(torch.finfo(self.dtype).eps, **self.dd)
        ga = torch.tensor(self.ga, **self.dd)

        scale = torch.exp(gam * (1.0 - qref / (qmod - eps)))

        return torch.where(
            qmod > 0.0,
            torch.exp(ga * (1.0 - scale)),
            torch.exp(ga),
        )

    def _dzeta(self, gam: Tensor, qref: Tensor, qmod: Tensor) -> Tensor:
        """
        Derivative of charge scaling function with respect to `qmod`.

        Parameters
        ----------
        gam : Tensor
            Chemical hardness.
        qref : Tensor
            Reference charges.
        qmod : Tensor
            Modified charges.

        Returns
        -------
        Tensor
            Derivative of charges.
        """
        eps = torch.tensor(torch.finfo(self.dtype).eps, **self.dd)
        ga = torch.tensor(self.ga, **self.dd)

        scale = torch.exp(gam * (1.0 - qref / (qmod - eps)))
        zeta = torch.exp(ga * (1.0 - scale))

        return torch.where(
            qmod > 0.0,
            -ga * gam * scale * zeta * storch.divide(qref, qmod**2),
            torch.tensor(0.0, **self.dd),
        )

    def get_refc6(self) -> Tensor:
        """
        Calculate reference C6 dispersion coefficients. The reference C6
        coefficients are not weighted by the Gaussian weights yet.

        Returns
        -------
        Tensor
            Reference C6 coefficients.
        """
        zero = torch.tensor(0.0, **self.dd)

        numbers = self.unique
        refsys = reference.refsys.to(self.device)[numbers]
        refascale = reference.refascale.to(**self.dd)[numbers]
        refalpha = reference.refalpha.to(**self.dd)[numbers]
        refscount = reference.refscount.to(**self.dd)[numbers]
        secscale = reference.secscale.to(**self.dd)
        secalpha = reference.secalpha.to(**self.dd)

        if self.ref_charges == "eeq":
            from .reference.charge_eeq import clsh as _refsq

            refsq = _refsq.to(**self.dd)[numbers]
        elif self.ref_charges == "gfn2":
            from .reference.charge_gfn2 import refh as _refsq

            refsq = _refsq.to(**self.dd)[numbers]
        else:
            raise ValueError(f"Unknown reference charges: {self.ref_charges}")

        mask = refsys > 0

        zeff = data.ZEFF.to(self.device)[refsys]
        gam = data.GAM.to(**self.dd)[refsys] * self.gc

        # charge scaling
        zeta = torch.where(
            mask,
            self._zeta(gam, zeff, refsq + zeff),
            zero,
        )

        aiw = secscale[refsys] * secalpha[refsys] * zeta.unsqueeze(-1)
        h = refalpha - refscount.unsqueeze(-1) * aiw
        alpha = refascale.unsqueeze(-1) * h

        # (..., nunique, r, 23) -> (..., n, r, 23)
        a = torch.where(alpha > 0.0, alpha, zero)[self.atom_to_unique]

        # (..., n, r, 23) -> (..., n, n, r, r)
        return trapzd(a)


def trapzd(polarizability: Tensor) -> Tensor:
    """
    Numerical Casimir--Polder integration.

    Parameters
    ----------
    polarizability : Tensor
        Polarizabilities of shape `(..., nat, nref, 23)`

    Returns
    -------
    Tensor
        C6 coefficients.
    """
    thopi = 3.0 / 3.141592653589793238462643383279502884197

    weights = torch.tensor(
        [
            2.4999500000000000e-002,
            4.9999500000000000e-002,
            7.5000000000000010e-002,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1500000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.3500000000000000,
            0.5000000000000000,
            0.7500000000000000,
            1.0000000000000000,
            1.7500000000000000,
            2.5000000000000000,
            1.2500000000000000,
        ],
        device=polarizability.device,
        dtype=polarizability.dtype,
    )

    # NOTE: In the old version, a memory inefficient intermediate tensor was
    # created. The new version uses `einsum` to avoid this.
    #
    # (..., 1, nat, 1, nref, 23) * (..., nat, 1, nref, 1, 23) =
    # (..., nat, nat, nref, nref, 23) -> (..., nat, nat, nref, nref)
    # a = alpha.unsqueeze(-4).unsqueeze(-3) * alpha.unsqueeze(-3).unsqueeze(-2)
    #
    # rc6 = thopi * torch.sum(weights * a, dim=-1)

    return thopi * einsum(
        "w,...iaw,...jbw->...ijab",
        *(weights, polarizability, polarizability),
    )


def is_exceptional(x: Tensor, dtype: torch.dtype) -> Tensor:
    """
    Check if a tensor is exceptional (NaN or too large).

    Parameters
    ----------
    x : Tensor
        Tensor to check.
    dtype : torch.dtype
        Data type of the tensor.

    Returns
    -------
    Tensor
        Boolean tensor indicating exceptional values.
    """
    return torch.isnan(x) | (x > torch.finfo(dtype).max)
