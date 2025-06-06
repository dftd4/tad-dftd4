# SPDX-Identifier: CC0-1.0
import tad_mctc as mctc
import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs, real_triples
from tad_mctc.data.molecules import mols

import tad_dftd4 as d4
from tad_dftd4 import data, defaults
from tad_dftd4.typing import DD, Tensor, TensorLike

dd = {"device": torch.device("cpu"), "dtype": torch.float64}

mol = mols["LiH"]
numbers = mol["numbers"].to(dd["device"])
positions = mol["positions"].to(**dd)
charge = torch.tensor(0.0)


class Param(torch.nn.Module):
    def __init__(
        self,
        s6: float = defaults.S6,
        s8: float = defaults.S8,
        s9: float = defaults.S9,
        s10: float = defaults.S10,
        a1: float = defaults.A1,
        a2: float = defaults.A2,
        alp: float = defaults.ALP,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Dummy tensor to get the device and dtype.
        self.register_buffer(
            "dummy", torch.empty(0, device=device, dtype=dtype)
        )

        self.s6 = torch.nn.Parameter(torch.tensor(s6, **dd))
        self.s8 = torch.nn.Parameter(torch.tensor(s8, **dd))
        self.s9 = torch.nn.Parameter(torch.tensor(s9, **dd))
        self.s10 = torch.nn.Parameter(torch.tensor(s10, **dd))
        self.a1 = torch.nn.Parameter(torch.tensor(a1, **dd))
        self.a2 = torch.nn.Parameter(torch.tensor(a2, **dd))
        self.alp = torch.nn.Parameter(torch.tensor(alp, **dd))

    @property
    def device(self) -> torch.device:
        """Returns the device where the first parameter/buffer is located."""
        return self.dummy.device  # type: ignore

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type of the parameters."""
        return self.dummy.dtype  # type: ignore

    @property
    def dd(self) -> DD:
        """Returns the dictionary of device and data type."""
        return {"device": self.device, "dtype": self.dtype}


class TwoBody:
    pass


class ManyBody:
    def get_dispersion(self, numbers: Tensor, positions: Tensor) -> Tensor:
        # Placeholder for the actual implementation
        return torch.tensor(0.0, **dd)


cutoff = d4.cutoff.Cutoff()
model = d4.model.D4Model(numbers, ref_charges="eeq")


print(model)


q = d4.disp.get_eeq_charges(numbers, positions, charge)
cn = d4.ncoord.cn_d4(numbers, positions)

weights = model.weight_references(cn, q)
alpha = model.get_atomic_alpha(weights)

c6_2 = d4.model.utils.trapzd_noref(alpha, alpha)
c6 = model.get_atomic_c6(weights)
diff = c6 - c6_2
print(diff.abs().max())


atm_1 = d4.get_atm_dispersion(numbers, positions, cutoff=cutoff.disp3, c6=c6)

atm_2 = d4.get_exact_atm_dispersion(numbers, positions, cutoff.disp3, alpha)


print(atm_1)
print(atm_2)

print(atm_1.sum() * mctc.units.AU2KCALMOL)
print(atm_2.sum() * mctc.units.AU2KCALMOL)
