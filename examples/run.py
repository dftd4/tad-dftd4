# SPDX-Identifier: CC0-1.0
import tad_mctc as mctc
import torch
from tad_mctc import storch
from tad_mctc.batch import real_pairs, real_triples

import tad_dftd4 as d4

numbers = mctc.convert.symbol_to_number(
    symbols="C C C C N C S H H H H H".split()
)

# coordinates in Bohr
positions = torch.tensor(
    [
        [-2.56745685564671, -0.02509985979910, 0.00000000000000],
        [-1.39177582455797, +2.27696188880014, 0.00000000000000],
        [+1.27784995624894, +2.45107479759386, 0.00000000000000],
        [+2.62801937615793, +0.25927727028120, 0.00000000000000],
        [+1.41097033661123, -1.99890996077412, 0.00000000000000],
        [-1.17186102298849, -2.34220576284180, 0.00000000000000],
        [-2.39505990368378, -5.22635838332362, 0.00000000000000],
        [+2.41961980455457, -3.62158019253045, 0.00000000000000],
        [-2.51744374846065, +3.98181713686746, 0.00000000000000],
        [+2.24269048384775, +4.24389473203647, 0.00000000000000],
        [+4.66488984573956, +0.17907568006409, 0.00000000000000],
        [-4.60044244782237, -0.17794734637413, 0.00000000000000],
    ]
)

charge = torch.tensor(0.0)
param = d4.get_params("tpssh")

cutoff = d4.cutoff.Cutoff()
model = d4.model.D4Model(numbers, ref_charges="eeq")

print(model)


q = d4.disp.get_eeq_charges(numbers, positions, charge)
cn = d4.ncoord.cn_d4(numbers, positions)

weights = model.weight_references(cn, q)


aiw = model._get_alpha()
alpha = mctc.math.einsum("...nr,...nra->...na", weights, aiw)


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
