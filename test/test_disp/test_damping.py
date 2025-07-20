# This file is part of tad-dftd4.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
General checks for damping functions.
"""

from tad_dftd4.damping import (
    MZeroDamping,
    OptimisedPowerDamping,
    RationalDamping,
    ZeroDamping,
)


def test_eq_same_instance() -> None:
    """Test that a Damping instance is equal to itself."""
    damp = RationalDamping()
    assert damp == damp


def test_eq_same_class() -> None:
    """Test that two instances of the same Damping subclass are equal."""
    damp1 = RationalDamping()
    damp2 = RationalDamping()
    assert damp1 == damp2
    assert not (damp1 != damp2)


def test_neq_different_class() -> None:
    """Test that instances of different Damping subclasses are not equal."""
    damp1 = RationalDamping()
    damp2 = ZeroDamping()
    assert damp1 != damp2
    assert not (damp1 == damp2)


def test_neq_all_subclasses() -> None:
    """
    Test that all defined Damping subclasses are only equal to themselves.
    """
    damping_classes = [
        RationalDamping,
        ZeroDamping,
        MZeroDamping,
        OptimisedPowerDamping,
    ]
    instances = [Cls() for Cls in damping_classes]

    for i, inst1 in enumerate(instances):
        for j, inst2 in enumerate(instances):
            if i == j:
                assert inst1 == inst2
            else:
                assert inst1 != inst2


def test_neq_different_type() -> None:
    """
    Test that a Damping instance is not equal to an object of another type.
    """
    damp = RationalDamping()
    assert damp != "a string"
    assert damp != 123
    assert damp != None
    assert damp != [RationalDamping()]


def test_eq_not_implemented() -> None:
    """
    Test that comparison with an unrelated object returns NotImplemented.
    """

    class Unrelated:
        def __eq__(self, other):
            return NotImplemented

    damp = RationalDamping()
    unrelated = Unrelated()
    # The result of `damp == unrelated` is False because Python falls
    # back to `unrelated == damp`, which returns NotImplemented, and
    # ultimately results in inequality.
    assert (damp == unrelated) is False
