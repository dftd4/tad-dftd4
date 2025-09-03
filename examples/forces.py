# SPDX-Identifier: CC0-1.0
import tad_mctc as mctc
import torch

import tad_dftd4 as d4

dd: mctc.typing.DD = {"device": torch.device("cpu"), "dtype": torch.float64}

#####################
# System Parameters #
#####################

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
    ],
    **dd
)

# total charge of the system
charge = torch.tensor(0.0)

# TPSSh-D4-ATM parameters
param = d4.get_params(method="d4", functional="tpssh")


#######################
# Analytical Gradient #
#######################

pos = positions.clone().requires_grad_(True)
energy = d4.dftd4(numbers, pos, charge, param)

(grad,) = torch.autograd.grad(energy.sum(), pos)


######################
# Numerical Gradient #
######################

num_grad = torch.zeros(positions.shape, **dd)
step = 1e-5

for i in range(numbers.shape[-1]):
    for j in range(3):
        positions[i, j] += step
        e1 = d4.dftd4(numbers, positions, charge, param).sum()

        positions[i, j] -= 2 * step
        e2 = d4.dftd4(numbers, positions, charge, param).sum()

        positions[i, j] += step
        num_grad[i, j] = (e1 - e2) / (2 * step)


# Check if analytical and numerical gradients match
assert torch.allclose(grad, num_grad, atol=1e-8), "Gradient check failed!"
