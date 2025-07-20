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
Model: Typing
=============

Type annotations for models.
"""
from __future__ import annotations

from typing import Union

from tad_mctc.typing import Literal, TypeAlias

from .d3 import D3Model
from .d4 import D4Model
from .d4s import D4SModel
from .d5 import D5Model

__all__ = ["ModelKey", "ModelInst", "ModelInstD3", "ModelInstD4"]


ModelKey: TypeAlias = Literal["d3", "d4", "d4s", "d5"]

ModelInstD3: TypeAlias = D3Model
ModelInstD4: TypeAlias = Union[D4Model, D4SModel]
ModelInstD5: TypeAlias = D5Model

ModelInst: TypeAlias = Union[ModelInstD3, ModelInstD4, ModelInstD5]
