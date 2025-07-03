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
Damping Parameters
==================

Read damping parameters from toml file. The TOML file is coped from the
Fortran GitHub repository.
- https://github.com/dftd3/s-dftd3/blob/main/assets/parameters.toml
- https://github.com/dftd4/dftd4/blob/main/assets/parameters.toml
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, MutableMapping

from .base import Param

try:
    import tomli as toml
except ImportError as e:
    raise ImportError(
        "A TOML package is required for TOML support. "
        "You can install it via `pip install tomli`."
    ) from e

from .base import DispersionMethod

__all__ = ["get_params"]


@lru_cache(maxsize=None)
def _load(method: DispersionMethod) -> dict[str, Any]:
    """Read <method>.toml exactly once."""
    _BASE = Path(__file__).parent
    toml_path = _BASE / f"{method.value}.toml"

    if not toml_path.is_file():
        raise FileNotFoundError(f"TOML file {toml_path} missing.")

    with toml_path.open("rb") as fp:
        return toml.load(fp)


def get_params(
    *,
    method: DispersionMethod | str,
    functional: str | None,
    variant: str | None = None,
    keep_doi: bool = False,
) -> Param:
    """
    Obtain damping parameters for a given functional.

    Parameters
    ----------
    method : DispersionMethod
        Dispersion method (e.g., D3, D4).
    functional : str
        Functional name, case-insensitive.
    variant : str
        Damping variant (e.g., "bj-eeq-atm").
    keep_doi : bool, optional
        If ``True``, keep the DOI field in the output. Defaults to ``False``.

    Raises
    ------
    KeyError
        If functional or dispersion variant is not found in damping
        parameter file.
    """
    method = DispersionMethod(method)
    table = _load(method)

    if functional in (None, "default"):
        default_section = table["default"]

        variant_section = default_section[method.value]
        if variant not in variant_section:
            raise KeyError(
                f"Variant '{variant}' not found in default parameters for "
                f"method={method.value!r}."
            )

        disp_method_section = default_section["parameter"]
    else:
        func_section = table["parameter"]
        if functional not in func_section:
            raise KeyError(
                f"Functional '{functional!r}' not found in damping parameters."
            )

        disp_method_section = func_section[functional.casefold()]
        if method.value not in disp_method_section:
            raise KeyError(
                f"Method '{method}' not found in damping parameters "
                f"for '{functional!r}'."
            )

    # Get default variant if not specified
    if variant is None:
        variant = table["default"][method.value][0]

    variant_section = disp_method_section[method.value]
    if variant not in variant_section:
        raise KeyError(
            f"Variant '{variant}' not found for functional="
            f"{functional!r}, method={method.value!r}."
        )

    block = variant_section[variant]

    out: MutableMapping[str, float | str] = dict(block)
    if not keep_doi and "doi" in out:
        out.pop("doi")

    return Param(**out)  # type: ignore
