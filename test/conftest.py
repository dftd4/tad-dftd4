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


def pytest_addoption(parser: pytest.Parser):
    """Set up additional command line options."""

    parser.addoption(
        "--detect-anomaly",
        action="store_true",
        help="Enable more comprehensive gradient tests.",
    )


def pytest_configure(config: pytest.Config):
    """Pytest configuration hook."""

    if config.getoption("--detect-anomaly"):
        torch.autograd.anomaly_mode.set_detect_anomaly(True)

    # register an additional marker
    config.addinivalue_line("markers", "cuda: mark test that require CUDA.")


def pytest_runtest_setup(item: pytest.Function):
    """Custom marker for tests requiring CUDA."""

    for _ in item.iter_markers(name="cuda"):
        if not torch.cuda.is_available():
            pytest.skip(
                "Torch not compiled with CUDA enabled or no CUDA device available."
            )
