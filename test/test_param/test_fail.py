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
Test reading parameters from TOML file.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tad_dftd4.damping import get_params
from tad_dftd4.damping.parameters import loader as param_loader


def test_unknown_func() -> None:
    with pytest.raises(KeyError):
        get_params(method="d4", variant="d4-eeq-bj", functional="unknown")


def test_unknown_variant() -> None:
    with pytest.raises(KeyError):
        get_params(method="d4", functional="pbe", variant="unknown")


def test_unknown_variant_default() -> None:
    """Unknown variant in the default section."""
    with pytest.raises(KeyError, match="not found in default parameters"):
        get_params(method="d4", functional=None, variant="no-such-variant")


def test_unknown_variant_functional() -> None:
    """Unknown variant for a known functional."""
    with pytest.raises(KeyError, match="not found for functional"):
        get_params(method="d4", functional="pbe", variant="no-such-variant")


def test_missing_toml() -> None:
    """TOML file does not exist (d5 has no TOML yet)."""
    with pytest.raises(FileNotFoundError, match="missing"):
        get_params(method="d5", functional="pbe", variant="x")


def test_default_variant_for_functional() -> None:
    """variant=None with real functional uses default variant."""
    params = get_params(method="d4", functional="pbe", variant=None)
    assert isinstance(params, dict)
    assert "a1" in params


def test_method_missing_in_functional() -> None:
    """Functional entry exists but method key is absent."""
    fake_table = {
        "default": {"d4": ["bj-eeq-atm"]},
        "parameter": {
            "pbe": {
                "reference": {},
                # 'd4' key intentionally absent
            }
        },
    }
    with patch.object(param_loader, "_load", return_value=fake_table):
        with pytest.raises(KeyError, match="Method"):
            get_params(method="d4", functional="pbe", variant="bj-eeq-atm")
