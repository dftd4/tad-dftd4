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

Read damping parameters from toml file. The TOML file is coped from the DFT-D4
Fortran GitHub repository.
(https://github.com/dftd4/dftd4/blob/main/assets/parameters.toml)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from ...typing import Tensor, overload

__all__ = ["get_params", "get_params_default"]


@overload
def get_params(
    func: str,
    variant: Literal["bj-eeq-atm"],
    with_reference: Literal[False],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]: ...


@overload
def get_params(
    func: str,
    variant: Literal["bj-eeq-atm"],
    with_reference: Literal[True],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor | str]: ...


def get_params(
    func: str,
    variant: Literal["bj-eeq-atm"] = "bj-eeq-atm",
    with_reference: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor] | dict[str, Tensor | str]:
    """
    Obtain damping parameters for a given functional.

    Parameters
    ----------
    func : str
        Functional name, case-insensitive.
    variant : Literal["bj-eeq-atm"]
        D4 variant. Only 'bj-eeq-atm' (default D4 model) is supported.
    device : torch.device | None, optional
        Pytorch device for calculations. Defaults to `None`.
    dtype : torch.dtype | None, optional
        Pytorch dtype for calculations. Defaults to `None`.

    Returns
    -------
    dict[str, Tensor]
        Damping parameters for the given functional.

    Raises
    ------
    KeyError
        If functional or D4 variant is not found in damping parameters file.
    """
    # pylint: disable=import-outside-toplevel
    import tomli as toml

    table: dict[str, dict[str, dict[str, dict[str, dict[str, float | str]]]]]
    with open(Path(__file__).parent / "parameters.toml", mode="rb") as f:
        table = toml.load(f)

    func_section = table["parameter"]
    if func not in func_section:
        raise KeyError(
            f"Functional '{func.casefold()}' not found in damping parameters."
        )

    variant_section = func_section[func]["d4"]
    if variant not in variant_section:
        raise KeyError(
            f"Variant '{variant}' not found in damping parameters for '{func}'."
        )

    par_section = variant_section[variant]

    d: dict[str, Tensor | str] = {}
    for k, v in par_section.items():
        if k == "doi":
            if with_reference is False:
                continue
            d[k] = str(v)
        else:
            d[k] = torch.tensor(v, device=device, dtype=dtype)

    return d


def get_params_default(
    variant: Literal[
        "bj-eeq-atm", "d4.bj-eeq-two", "d4.bj-eeq-mbd"
    ] = "bj-eeq-atm",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """
    Obtain default damping parameters and method info.

    Parameters
    ----------
    device : torch.device | None, optional
        Pytorch device for calculations. Defaults to `None`.
    dtype : torch.dtype | None, optional
        Pytorch dtype for calculations. Defaults to `None`.

    Returns
    -------
    dict[str, Tensor]
        Damping parameters for the given functional.
    """
    # pylint: disable=import-outside-toplevel
    import tomli as toml

    table: dict[str, dict[str, dict[str, dict[str, dict[str, float | str]]]]]
    with open(Path(__file__).parent / "parameters.toml", mode="rb") as f:
        table = toml.load(f)

    d = {}
    for k, v in table["default"]["parameter"]["d4"][variant].items():
        if isinstance(v, float):
            d[k] = torch.tensor(v, device=device, dtype=dtype)

    return d
