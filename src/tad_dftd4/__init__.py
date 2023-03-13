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
Torch autodiff DFT-D4
=====================

Implementation of the DFT-D4 dispersion model in PyTorch.
This module allows to process a single structure or a batch of structures for
the calculation of atom-resolved dispersion energies.

.. note::

   This project is still in early development and the API is subject to change.
   Contributions are welcome, please checkout our
   `contributing guidelines <https://github.com/dftd4/tad-dftd4/blob/main/CONTRIBUTING.md>`_.

Example
-------
>>> import torch
>>> import tad_dftd4 as d4
>>>
>>> # S22 system 4: formamide dimer
>>> numbers = d4.utils.pack((
...     d4.utils.to_number("C C N N H H H H H H O O".split()),
...     d4.utils.to_number("C O N H H H".split()),
... ))
>>>
>>> # coordinates in Bohr
>>> positions = d4.utils.pack((
...     torch.tensor([
...         [-3.81469488143921, +0.09993441402912, 0.00000000000000],
...         [+3.81469488143921, -0.09993441402912, 0.00000000000000],
...         [-2.66030049324036, -2.15898251533508, 0.00000000000000],
...         [+2.66030049324036, +2.15898251533508, 0.00000000000000],
...         [-0.73178529739380, -2.28237795829773, 0.00000000000000],
...         [-5.89039325714111, -0.02589114569128, 0.00000000000000],
...         [-3.71254944801331, -3.73605775833130, 0.00000000000000],
...         [+3.71254944801331, +3.73605775833130, 0.00000000000000],
...         [+0.73178529739380, +2.28237795829773, 0.00000000000000],
...         [+5.89039325714111, +0.02589114569128, 0.00000000000000],
...         [-2.74426102638245, +2.16115570068359, 0.00000000000000],
...         [+2.74426102638245, -2.16115570068359, 0.00000000000000],
...     ]),
...     torch.tensor([
...         [-0.55569743203406, +1.09030425468557, 0.00000000000000],
...         [+0.51473634678469, +3.15152550263611, 0.00000000000000],
...         [+0.59869690244446, -1.16861263789477, 0.00000000000000],
...         [-0.45355203669134, -2.74568780438064, 0.00000000000000],
...         [+2.52721209544999, -1.29200800956867, 0.00000000000000],
...         [-2.63139587595376, +0.96447869452240, 0.00000000000000],
...     ]),
... ))
>>>
>>> # total charge of both systems
>>> charge = torch.tensor([0.0, 0.0])
>>>
>>> # TPSS0-D4-ATM parameters
>>> param = {
...     "s6": positions.new_tensor(1.0),
...     "s8": positions.new_tensor(1.85897750),
...     "s9": positions.new_tensor(1.0),
...     "a1": positions.new_tensor(0.44286966),
...     "a2": positions.new_tensor(4.60230534),
... }
>>>
>>> # calculate dispersion energy in Hartree
>>> energy = torch.sum(d4.dftd4(numbers, positions, charge, param), -1)
>>> torch.set_printoptions(precision=10)
>>> print(energy)
tensor([-0.0088341432, -0.0027013607])
>>> print(energy[0] - 2*energy[1])
tensor(-0.0034314217)
"""
import torch

from . import charges, cutoff, damping, data, disp, model, ncoord, utils
from .__version__ import __version__
from ._typing import CountingFunction, DampingFunction, Tensor
from .disp import dftd4
