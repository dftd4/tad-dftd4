# SPDX-Identifier: CC0-1.0
import tad_mctc as mctc
import torch

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

# total charge of the system
charge = torch.tensor(0.0)

# TPSSh-D4-ATM parameters
param = d4.damping.Param(
    s6=positions.new_tensor(1.0),
    s8=positions.new_tensor(1.85897750),
    s9=positions.new_tensor(1.0),
    a1=positions.new_tensor(0.44286966),
    a2=positions.new_tensor(4.60230534),
)

# parameters can also be obtained using the functional name:
# param = d4.get_params(method="d4", functional="tpssh")

energy1 = d4.dftd4(numbers, positions, charge, param)

# class-based interface
disp = d4.dispersion.DispD4()
energy2 = disp.calculate(numbers, positions, charge, param)

torch.set_printoptions(precision=10)

ref = torch.tensor(
    [
        -0.0020841344,
        -0.0018971195,
        -0.0018107513,
        -0.0018305695,
        -0.0021737693,
        -0.0019484236,
        -0.0022788253,
        -0.0004080658,
        -0.0004261866,
        -0.0004199839,
        -0.0004280768,
        -0.0005108935,
    ]
)
assert torch.allclose(energy1, ref, atol=1e-8), "Energy does not match"
assert torch.allclose(energy2, ref, atol=1e-8), "Energy does not match"


print(energy1)
# tensor([-0.0020841344, -0.0018971195, -0.0018107513, -0.0018305695,
#         -0.0021737693, -0.0019484236, -0.0022788253, -0.0004080658,
#         -0.0004261866, -0.0004199839, -0.0004280768, -0.0005108935])
