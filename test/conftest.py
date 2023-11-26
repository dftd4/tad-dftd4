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
Setup for pytest.
"""

import pytest
import torch

# avoid randomness and non-deterministic algorithms
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

torch.set_printoptions(precision=10)

FAST_MODE: bool = True
"""Flag for fast gradient tests."""

DEVICE: torch.device | None = None
"""Name of Device."""


def pytest_addoption(parser: pytest.Parser) -> None:
    """Set up additional command line options."""

    parser.addoption(
        "--cuda",
        action="store_true",
        help="Use GPU as default device.",
    )

    parser.addoption(
        "--detect-anomaly",
        "--da",
        action="store_true",
        help="Enable PyTorch's debug mode for gradient tests.",
    )

    parser.addoption(
        "--jit",
        action="store_true",
        help="Enable JIT during tests (default = False).",
    )

    parser.addoption(
        "--fast",
        action="store_true",
        help="Use `fast_mode` for gradient checks (default = True).",
    )

    parser.addoption(
        "--slow",
        action="store_true",
        help="Do *not* use `fast_mode` for gradient checks (default = False).",
    )

    parser.addoption(
        "--tpo-linewidth",
        action="store",
        default=400,
        type=int,
        help=(
            "The number of characters per line for the purpose of inserting "
            "line breaks (default = 80). Thresholded matrices will ignore "
            "this parameter."
        ),
    )

    parser.addoption(
        "--tpo-precision",
        action="store",
        default=6,
        type=int,
        help=(
            "Number of digits of precision for floating point output " "(default = 4)."
        ),
    )

    parser.addoption(
        "--tpo-threshold",
        action="store",
        default=1000,
        type=int,
        help=(
            "Total number of array elements which trigger summarization "
            "rather than full `repr` (default = 1000)."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Pytest configuration hook."""

    if config.getoption("--detect-anomaly"):
        torch.autograd.anomaly_mode.set_detect_anomaly(True)

    if config.getoption("--jit"):
        torch.jit._state.enable()  # type: ignore # pylint: disable=protected-access
    else:
        torch.jit._state.disable()  # type: ignore # pylint: disable=protected-access

    global FAST_MODE
    if config.getoption("--fast"):
        FAST_MODE = True
    if config.getoption("--slow"):
        FAST_MODE = False

    global DEVICE
    if config.getoption("--cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("No cuda devices available.")

        if FAST_MODE is True:
            FAST_MODE = False

            from warnings import warn

            warn(
                "Fast mode for gradient checks not compatible with GPU "
                "execution. Switching to slow mode. Use the '--slow' flag "
                "for GPU tests ('--cuda') to avoid this warning."
            )

        DEVICE = torch.device("cuda:0")
        torch.use_deterministic_algorithms(False)

        # `torch.set_default_tensor_type` is deprecated since 2.1.0 and version
        # 2.0.0 introduces `torch.set_default_device`
        if torch.__version__ < (2, 0, 0):  # type: ignore
            torch.set_default_tensor_type("torch.cuda.FloatTensor")  # type: ignore
        else:
            torch.set_default_device(DEVICE)  # type: ignore
    else:
        torch.use_deterministic_algorithms(True)
        DEVICE = None

    if config.getoption("--tpo-linewidth"):
        torch.set_printoptions(linewidth=config.getoption("--tpo-linewidth"))

    if config.getoption("--tpo-precision"):
        torch.set_printoptions(precision=config.getoption("--tpo-precision"))

    if config.getoption("--tpo-threshold"):
        torch.set_printoptions(threshold=config.getoption("--tpo-threshold"))

    # register an additional marker
    config.addinivalue_line("markers", "cuda: mark test that require CUDA.")


def pytest_runtest_setup(item: pytest.Function) -> None:
    """Custom marker for tests requiring CUDA."""

    for _ in item.iter_markers(name="cuda"):
        if not torch.cuda.is_available():
            pytest.skip("Torch not compiled with CUDA or no CUDA device available.")
