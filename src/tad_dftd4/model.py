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

from . import data, params
from ._typing import Tensor, TensorLike

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

    alpha: Tensor
    """Reference polarizabilities of unique species."""

    __slots__ = ("numbers", "ga", "gc", "wf", "alpha")

    def __init__(
        self,
        numbers: Tensor,
        ga: float = ga_default,
        gc: float = gc_default,
        wf: float = wf_default,
        alpha: Tensor | None = None,
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
        alpha : Tensor | None, optional
            Reference polarizabilities of unique species. Defaults to `None`.
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

        if alpha is None:
            self.alpha = self._set_refalpha_eeq()

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

    def weight_references(
        self,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> Tensor:
        """
        Calculate the weights of the reference system.

        Parameters
        ----------
        cn : Tensor | None, optional
            Coordination number of every atom. Defaults to `None` (0).
        q : Tensor | None, optional
            Partial charge of every atom. Defaults to `None` (0).

        Returns
        -------
        Tensor
            Weights for the atomic reference systems.
        """
        if cn is None:
            cn = torch.zeros_like(self.numbers, dtype=self.dtype)
        if q is None:
            q = torch.zeros_like(self.numbers, dtype=self.dtype)

        refc = params.refc[self.numbers].to(self.device)
        refq = params.refq[self.numbers].type(self.dtype).to(self.device)
        mask = refc > 0

        # Due to the exponentiation, `norm` and `expw` may become very small
        # (down to 1e-300). This causes problems for the division by `norm`,
        # since single precision, i.e. `torch.float`, only goes to around 1e-38.
        # Consequently, some values become zero although the actual result
        # should be close to one. The problem does not arise when using `torch.
        # double`. In order to avoid this error, which is also difficult to
        # detect, this part always uses `torch.double`. `params.refcn` is saved
        # with `torch.double`.
        refcn = params.refcn[self.numbers].to(self.device)

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
        expw = torch.where(
            mask,
            torch.where(
                refc == 3,
                torch.pow(tmp, self.wf)
                + torch.pow(tmp, 2 * self.wf)
                + torch.pow(tmp, 3 * self.wf),
                torch.where(
                    refc == 1,
                    torch.pow(tmp, self.wf),
                    tmp,
                ),
            ),
            dcn.new_tensor(0.0),
        )

        # normalize weights
        norm = torch.sum(expw, dim=-1, keepdim=True)
        gw_temp = (expw / norm).type(self.dtype)

        # maximum reference CN for each atom
        maxcn = torch.max(refcn, dim=-1, keepdim=True)[0]

        # prevent division by 0 and small values
        exceptional = (torch.isnan(gw_temp)) | (gw_temp > torch.finfo(self.dtype).max)

        gw = torch.where(
            exceptional,
            torch.where(
                refcn == maxcn,
                gw_temp.new_tensor(1.0),
                gw_temp.new_tensor(0.0),
            ),
            gw_temp,
        )

        # unsqueeze for reference dimension
        zeff = data.zeff[self.numbers].unsqueeze(-1)
        gam = data.gam[self.numbers].unsqueeze(-1) * self.gc
        q = q.unsqueeze(-1)

        # charge scaling
        zeta = torch.where(
            mask,
            self._zeta(gam, refq + zeff, q + zeff),
            gw_temp.new_tensor(0.0),
        )

        return zeta * gw

    def get_atomic_c6(self, gw: Tensor) -> Tensor:
        """
        Calculate atomic C6 dispersion coefficients.

        Parameters
        ----------
        gw : Tensor
            Weights for the atomic reference systems.

        Returns
        -------
        Tensor
            C6 coefficients for all atom pairs.
        """
        alpha = self.alpha[self.atom_to_unique]

        # shape of alpha: (b, nat, nref, 23)
        # (b, 1, nat, 1, nref, 23) * (b, nat, 1, nref, 1, 23) =
        # (b, nat, nat, nref, nref, 23)
        rc6 = trapzd(
            alpha.unsqueeze(-4).unsqueeze(-3) * alpha.unsqueeze(-3).unsqueeze(-2)
        )

        # shape of gw: (batch, natoms, nref)
        # (b, 1, nat, 1, nref)*(b, nat, 1, nref, 1) = (b, nat, nat, nref, nref)
        g = gw.unsqueeze(-3).unsqueeze(-2) * gw.unsqueeze(-2).unsqueeze(-1)

        return torch.sum(g * rc6, dim=(-2, -1))

    def _zeta(self, gam: Tensor, qref: Tensor, qmod: Tensor) -> Tensor:
        """
        charge scaling function.

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
        return torch.where(
            qmod > 0.0,
            torch.exp(self.ga * (1.0 - torch.exp(gam * (1.0 - qref / qmod)))),
            torch.exp(qmod.new_tensor(self.ga)),
        )

    def _set_refalpha_eeq(self) -> Tensor:
        """
        Set the reference polarizibilities for unique species.

        Returns
        -------
        Tensor
            Reference polarizibilities for unique species (not all atoms).
        """
        numbers = self.unique
        refsys = params.refsys[numbers].to(self.device)
        refsq = params.refsq[numbers].type(self.dtype).to(self.device)
        refascale = params.refascale[numbers].type(self.dtype).to(self.device)
        refalpha = params.refalpha[numbers].type(self.dtype).to(self.device)
        refscount = params.refscount[numbers].type(self.dtype).to(self.device)
        secscale = params.secscale.type(self.dtype).to(self.device)
        secalpha = params.secalpha.type(self.dtype).to(self.device)

        mask = refsys > 0

        zeff = data.zeff[refsys].to(self.device)
        gam = data.gam[refsys].type(self.dtype).to(self.device) * self.gc

        aiw = secscale[refsys] * secalpha[refsys]

        # charge scaling
        zeta = torch.where(
            mask,
            self._zeta(gam, zeff, refsq + zeff),
            gam.new_tensor(0.0),
        )

        aiw = secscale[refsys] * secalpha[refsys] * zeta.unsqueeze(-1)
        h = refalpha - refscount.unsqueeze(-1) * aiw
        alpha = refascale.unsqueeze(-1) * h

        return torch.where(alpha > 0.0, alpha, alpha.new_tensor(0.0))


def trapzd(polarizability: Tensor) -> Tensor:
    """
    Numerical Casimir--Polder integration.

    Parameters
    ----------
    polarizability : Tensor
        Polarizabilities.

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
        ]
    )

    return thopi * torch.sum(weights * polarizability, dim=-1)
