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
Test the utility functions.
"""
import torch

from tad_dftd4.utils import real_atoms, real_pairs, real_triples


def test_real_atoms() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0, 0, 0],  # H2
            [6, 1, 1, 1, 1],  # CH4
        ],
    )
    ref = torch.tensor(
        [
            [True, True, False, False, False],  # H2
            [True, True, True, True, True],  # CH4
        ],
    )
    mask = real_atoms(numbers)
    assert (mask == ref).all()


def test_real_pairs_single() -> None:
    numbers = torch.tensor([6, 1, 1, 1, 1])  # CH4
    size = numbers.shape[0]

    ref = torch.full((size, size), True)
    mask = real_pairs(numbers, diagonal=True)
    assert (mask == ref).all()

    ref *= ~torch.diag_embed(torch.ones(size, dtype=torch.bool))
    mask = real_pairs(numbers, diagonal=False)
    assert (mask == ref).all()


def test_real_pairs_batch() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    ref = torch.tensor(
        [
            [
                [True, True, False],
                [True, True, False],
                [False, False, False],
            ],
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ],
        ]
    )
    mask = real_pairs(numbers, diagonal=True)
    assert (mask == ref).all()

    ref = torch.tensor(
        [
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ],
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ],
        ]
    )
    mask = real_pairs(numbers, diagonal=False)
    assert (mask == ref).all()


def test_real_triples_single() -> None:
    numbers = torch.tensor([8, 1, 1])  # H2O
    size = numbers.shape[0]

    ref = torch.full((size, size, size), True)
    mask = real_triples(numbers, diagonal=True)
    assert (mask == ref).all()

    ref *= ~torch.diag_embed(torch.ones(size, dtype=torch.bool))
    mask = real_pairs(numbers, diagonal=False)
    assert (mask == ref).all()


def test_real_triples_batch() -> None:
    numbers = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )

    ref = torch.tensor(
        [
            [
                [
                    [True, True, False],
                    [True, True, False],
                    [False, False, False],
                ],
                [
                    [True, True, False],
                    [True, True, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
            ],
            [
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
                [
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                ],
            ],
        ]
    )
    mask = real_triples(numbers, diagonal=True)
    assert (mask == ref).all()

    ref = torch.tensor(
        [
            [
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ],
                [
                    [False, True, False],
                    [True, False, False],
                    [False, False, False],
                ],
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                ],
            ],
            [
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ],
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ],
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ],
            ],
        ]
    )
    mask = real_triples(numbers, diagonal=False)
    assert (mask == ref).all()
