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
Model: Base
===========

This module contains the definition of the base dispersion model for the
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

from abc import abstractmethod

import torch
from tad_mctc import storch
from tad_mctc.math import einsum

from .. import data, reference
from ..typing import Literal, Tensor, TensorLike, overload
from .utils import trapzd

__all__ = ["AlphaModel", "WF_DEFAULT"]


GA_DEFAULT = 3.0
GC_DEFAULT = 2.0
WF_DEFAULT = 6.0


class AlphaModel(TensorLike):
    """
    The D4 dispersion model.
    """

    unique: Tensor
    """Unique elementes in the system."""

    ga: float
    """
    Maximum charge scaling height for partial charge extrapolation.

    :default: :data:`.GA_DEFAULT`
    """

    gc: float
    """
    Charge scaling steepness for partial charge extrapolation.

    :default: :data:`.GC_DEFAULT`
    """

    ref_charges: Literal["eeq", "gfn2"]
    """
    Reference charges to use for the model.

    :default: ``"eeq"``
    """

    _wf: Tensor | None
    """
    Weighting factor for coordination number interpolation.

    :default: ``None`` (model-dependent, set upon instantiation)
    """

    rc6: Tensor
    """
    Reference C6 coefficients of unique species.

    :default: ``None`` (calculated upon instantiation)
    """

    __slots__ = ("unique", "ga", "gc", "_wf", "_alphas", "ref_charges", "rc6")

    def __init__(
        self,
        unique: Tensor,
        ga: float = GA_DEFAULT,
        gc: float = GC_DEFAULT,
        ref_charges: Literal["eeq", "gfn2"] = "eeq",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Instantiate `D4Model`.

        Parameters
        ----------
        unique : Tensor
            Unique elementes in the system.
        ga : float, optional
            Maximum charge scaling height for partial charge extrapolation.
            Defaults to `GA_DEFAULT`.
        gc : float, optional
            Charge scaling steepness for partial charge extrapolation.
            Defaults to `GC_DEFAULT`.
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
        self.unique = unique

        self.ga = ga
        self.gc = gc
        self.ref_charges = ref_charges

        self._wf = None
        self._alphas = None

    ####################
    # Abstract methods #
    ####################

    @property
    def wf(self) -> Tensor:
        """Weighting factor for the Gaussian weights."""
        if self._wf is None:
            self._wf = self._get_wf()
        return self._wf

    @property
    def alphas(self) -> Tensor:
        """Reference polarizabilities."""
        if self._alphas is None:
            self._alphas = self._get_alpha()
        return self._alphas

    @abstractmethod
    def _get_wf(self) -> Tensor:
        """
        Get the weighting factor for the Gaussian weights.

        Returns
        -------
        Tensor
            Weighting factor for the Gaussian weights.
        """

    @abstractmethod
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

    @overload
    @abstractmethod
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        with_dgwdq: Literal[False] = False,
        with_dgwdcn: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    @abstractmethod
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        with_dgwdq: Literal[True] = True,
        with_dgwdcn: Literal[False] = False,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    @abstractmethod
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        with_dgwdq: Literal[False] = False,
        with_dgwdcn: Literal[True] = True,
    ) -> tuple[Tensor, Tensor]: ...

    @overload
    @abstractmethod
    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
        with_dgwdq: Literal[True] = True,
        with_dgwdcn: Literal[True] = True,
    ) -> tuple[Tensor, Tensor, Tensor]: ...

    @abstractmethod
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

    @abstractmethod
    def get_weighted_pols(self, gw: Tensor) -> Tensor:
        """
        Calculate the weighted polarizabilities for each atom and frequency.

        Parameters
        ----------
        gw : Tensor
            Weights for the atomic reference systems of shape
            ``(..., nat, nref)``.

        Returns
        -------
        Tensor
            Weighted polarizabilities of shape ``(..., nat, 23)``.
        """

    ##############
    # Properties #
    ##############

    ##################
    # Public methods #
    ##################

    def get_polarizabilities(self, weights: Tensor) -> Tensor:
        """
        Calculate static polarizabilities for all atoms.

        Parameters
        ----------
        weights : Tensor
            Weights for the atomic reference systems of shape
            ``(..., nat, nref)``.

        Returns
        -------
        Tensor
            Polarizabilities of shape ``(..., nat)``.
        """
        # (..., n, r) * (..., n, r) -> (..., n)
        return einsum("...nr,...nr->...n", weights, self._get_alpha()[..., 0])

    ###################
    # Private methods #
    ###################

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

    def _get_alpha(self) -> Tensor:
        """
        Calculate reference polarizabilities.

        Returns
        -------
        Tensor
            Reference polarizabilities of shape `(..., nat, ref, 23)`.
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
            # pylint: disable=import-outside-toplevel
            from ..reference.charge_eeq import clsh as _refsq

            refsq = _refsq.to(**self.dd)[numbers]
        elif self.ref_charges == "gfn2":
            # pylint: disable=import-outside-toplevel
            from ..reference.charge_gfn2 import refh as _refsq

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

        # (..., nunique, r, 23) -> (..., nunique, r, 23)
        return torch.where(alpha > 0.0, alpha, zero)

    def _get_refc6(self) -> Tensor:
        """
        Calculate reference C6 dispersion coefficients. The reference C6
        coefficients are not weighted by the Gaussian weights yet.

        Returns
        -------
        Tensor
            Reference C6 coefficients of shape ``(..., nat, nat, nref, nref)``.
        """
        # (..., n, r, 23) -> (..., n, n, r, r)
        return trapzd(self._get_alpha())

    ############
    # Printing #
    ############

    def __str__(self) -> str:  # pragma: no cover
        """Return a string representation of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  unique={self.unique},\n"
            f"  ga={self.ga},\n"
            f"  gc={self.gc},\n"
            f"  wf={self.wf},\n"
            f"  ref_charges={self.ref_charges},\n"
            f"  rc6={self.rc6.shape},\n"
            f")"
        )

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation of the model."""
        return str(self)
