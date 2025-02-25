# SPDX-Identifier: CC0-1.0
import tad_mctc as mctc
import torch

import tad_dftd4 as d4

# S22 system 4: formamide dimer
numbers = mctc.batch.pack(
    (
        mctc.convert.symbol_to_number("C C N N H H H H H H O O".split()),
        mctc.convert.symbol_to_number("C O N H H H".split()),
    )
)

# coordinates in Bohr
positions = mctc.batch.pack(
    (
        torch.tensor(
            [
                [-3.81469488143921, +0.09993441402912, 0.00000000000000],
                [+3.81469488143921, -0.09993441402912, 0.00000000000000],
                [-2.66030049324036, -2.15898251533508, 0.00000000000000],
                [+2.66030049324036, +2.15898251533508, 0.00000000000000],
                [-0.73178529739380, -2.28237795829773, 0.00000000000000],
                [-5.89039325714111, -0.02589114569128, 0.00000000000000],
                [-3.71254944801331, -3.73605775833130, 0.00000000000000],
                [+3.71254944801331, +3.73605775833130, 0.00000000000000],
                [+0.73178529739380, +2.28237795829773, 0.00000000000000],
                [+5.89039325714111, +0.02589114569128, 0.00000000000000],
                [-2.74426102638245, +2.16115570068359, 0.00000000000000],
                [+2.74426102638245, -2.16115570068359, 0.00000000000000],
            ]
        ),
        torch.tensor(
            [
                [-0.55569743203406, +1.09030425468557, 0.00000000000000],
                [+0.51473634678469, +3.15152550263611, 0.00000000000000],
                [+0.59869690244446, -1.16861263789477, 0.00000000000000],
                [-0.45355203669134, -2.74568780438064, 0.00000000000000],
                [+2.52721209544999, -1.29200800956867, 0.00000000000000],
                [-2.63139587595376, +0.96447869452240, 0.00000000000000],
            ]
        ),
    )
)

# total charge of both system
charge = torch.tensor([0.0, 0.0])

# TPSSh-D4-ATM parameters
param = {
    "s6": positions.new_tensor(1.0),
    "s8": positions.new_tensor(1.85897750),
    "s9": positions.new_tensor(1.0),
    "a1": positions.new_tensor(0.44286966),
    "a2": positions.new_tensor(4.60230534),
}

# calculate dispersion energy in Hartree
energy = torch.sum(d4.dftd4(numbers, positions, charge, param), -1)
torch.set_printoptions(precision=10)
print(energy)
# tensor([-0.0088341432, -0.0027013607])
print(energy[0] - 2 * energy[1])
# tensor(-0.0034314217)
