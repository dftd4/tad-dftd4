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

import pytest

from tad_dftd4.damping import get_params


def test_unknown_func() -> None:
    with pytest.raises(KeyError):
        get_params(method="d4", variant="d4-eeq-bj", functional="unknown")


def test_unknown_variant() -> None:
    with pytest.raises(KeyError):
        get_params(method="d4", functional="pbe", variant="unknown")
