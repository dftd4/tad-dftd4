from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import data
from ._typing import Any, Tensor, TensorLike

ref = np.load(Path(Path(__file__).parent, "params.npz"))

# fmt: off
refn = torch.tensor([
  0, 2, 1, 3, 4, 5, 7, 5, 4, 2, 1, 3, 4, 4, 5, 4, 3, 2, 1, 3, 4, 4,
  4, 4, 4, 3, 3, 4, 4, 2, 2, 3, 5, 4, 3, 2, 1, 3, 4, 3, 4, 4, 4, 3,
  3, 4, 3, 2, 2, 4, 5, 4, 3, 2, 1, 3, 4, 3, 1, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 4, 4, 3, 3, 3, 5, 3, 2, 2, 4, 5, 4, 3, 2, 1,
])
# fmt: on

THOPI = 3.0 / 3.141592653589793238462643383279502884197


def load_from_npz(npzfile: Any, name: str, dtype: torch.dtype) -> Tensor:
    """
    Get torch tensor from npz file.

    Parameters
    ----------
    npzfile : Any
        Loaded npz file.
    name : str
        Name of the tensor in the npz file.
    dtype : torch.dtype
        Data type of the tensor.

    Returns
    -------
    Tensor
        Tensor from the npz file.
    """
    name = name.replace("-", "").replace("+", "").lower()
    return torch.from_numpy(npzfile[name]).type(dtype)


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

    return THOPI * torch.sum(weights * polarizability, dim=-1)


ga_default = 3.0
gc_default = 2.0
wf_default = 6.0


class D4Model(TensorLike):
    """
    The D4 dispersion model.
    """

    ga: float = ga_default
    """Maximum charge scaling height for partial charge extrapolation."""

    gc: float = gc_default
    """Charge scaling steepness for partial charge extrapolation."""

    wf: float = wf_default
    """Weighting factor for coordination number interpolation."""

    def __init__(
        self,
        ga: float = ga_default,
        gc: float = gc_default,
        wf: float = wf_default,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.ga = ga
        self.gc = gc
        self.wf = wf

    def weight_references(
        self,
        numbers: Tensor,
        cn: Tensor | None = None,
        q: Tensor | None = None,
    ) -> Tensor:
        if cn is None:
            cn = torch.zeros_like(numbers, dtype=self.dtype)
        if q is None:
            q = torch.zeros_like(numbers, dtype=self.dtype)

        refc = load_from_npz(ref, "refc", torch.int8)[numbers]
        refcn = load_from_npz(ref, "refcn", self.dtype)[numbers]
        refq = load_from_npz(ref, "refq", self.dtype)[numbers]
        mask = refc > 0

        # For vectorization, we reformulate the Gaussian weighting function:
        # exp(-wf * igw * (cn - cn_ref)^2) = [exp(-(cn - cn_ref)^2)]^(wf * igw)

        # Gaussian weighting function part 1: exp(-(cn - cn_ref)^2)
        dcn = cn.unsqueeze(-1) - refcn
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
        # NOTE: Due to the exponentiation, `norm` and `expw` may become very
        # small (down to 1e-300). This causes problems for the division by
        # `norm``, since some values are just zero in Python when using `torch.
        # float`. The problem does not arise when using `torch.double`. In
        # order to avoid errors that are difficult to detect, this part should
        # always use `torch.double`.
        norm = torch.sum(expw, dim=-1, keepdim=True)
        gw_temp = expw / norm

        # maximum reference CN for each atom
        maxcn = torch.max(refcn, dim=-1, keepdim=True)[0]

        # prevent division by 0 and small values
        exceptional = (torch.isnan(gw_temp)) | (gw_temp > torch.finfo(self.dtype).max)

        gw = torch.where(
            exceptional,
            torch.where(
                refcn == maxcn,
                expw.new_tensor(1.0),
                expw.new_tensor(0.0),
            ),
            gw_temp,
        )

        # unsqueeze for reference dimension
        zeff = data.zeff[numbers].unsqueeze(-1)
        gam = data.gam[numbers].unsqueeze(-1) * self.gc
        q = q.unsqueeze(-1)

        # charge scaling
        zeta = torch.where(
            mask,
            self._zeta(gam, refq + zeff, q + zeff),
            expw.new_tensor(0.0),
        )

        return zeta * gw

    def get_atomic_c6(self, numbers: Tensor, gw: Tensor) -> Tensor:
        """
        Calculate atomic dispersion coefficients.
        """

        unique, atom_to_unique = torch.unique(numbers, return_inverse=True)
        alpha = self._set_refalpha_eeq(unique)[atom_to_unique]

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

    def _zeta(self, gam: Tensor, qref: Tensor, qmod: Tensor):
        return torch.where(
            qmod > 0.0,
            torch.exp(self.ga * (1.0 - torch.exp(gam * (1.0 - qref / qmod)))),
            torch.exp(qmod.new_tensor(self.ga)),
        )

    def _set_refalpha_eeq(self, numbers: Tensor):
        refsys = load_from_npz(ref, "refsys", torch.int8)[numbers]
        refsq = load_from_npz(ref, "refsq", torch.float)[numbers]
        refascale = load_from_npz(ref, "refascale", torch.float)[numbers]
        refalpha = load_from_npz(ref, "refalpha", torch.float)
        # FIXME: already store in this format
        refalpha = refalpha.reshape((87, 7, 23))[numbers]

        refscount = load_from_npz(ref, "refscount", torch.float)[numbers]
        secscale = load_from_npz(ref, "secscale", torch.float)
        secalpha = load_from_npz(ref, "secalpha", torch.float)

        # use isys for indexing!
        isys = refsys.type(torch.long)
        mask = refsys > 0

        zeff = data.zeff[isys]
        gam = data.gam[isys] * self.gc

        aiw = secscale[isys] * secalpha[isys]

        # charge scaling
        zeta = torch.where(
            mask,
            self._zeta(gam, zeff, refsq + zeff),
            gam.new_tensor(0.0),
        )

        aiw = secscale[isys] * secalpha[isys] * zeta.unsqueeze(-1)

        h = refalpha - refscount.unsqueeze(-1) * aiw
        alpha = refascale.unsqueeze(-1) * h

        return torch.where(alpha > 0.0, alpha, alpha.new_tensor(0.0))
