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
Test the packing utility functions.
"""

import torch

from tad_dftd4.utils import pack

mol1 = torch.tensor([1, 1])  # H2
mol2 = torch.tensor([8, 1, 1])  # H2O


def test_single_tensor() -> None:
    # dummy test: only give single tensor
    assert (mol1 == pack(mol1)).all()


def test_standard() -> None:
    # standard packing
    ref = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )
    packed = pack([mol1, mol2])
    assert (packed == ref).all()


def test_axis() -> None:
    ref = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    # different axis
    packed = pack([mol1, mol2], axis=-1)
    assert (packed == ref.T).all()


def test_size() -> None:
    ref = torch.tensor(
        [
            [1, 1, 0, 0],  # H2
            [8, 1, 1, 0],  # H2O
        ],
    )

    # one additional column of padding
    packed = pack([mol1, mol2], size=[4])
    assert (packed == ref).all()
