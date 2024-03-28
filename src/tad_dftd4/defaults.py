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
Defaults
========

This module defines the default values for all parameters within `tad_dftd4`.
"""

# DFT-D4

D4_CN_CUTOFF = 30.0
"""Coordination number cutoff (30.0)."""

D4_CN_EEQ_CUTOFF = 25.0
"""Coordination number cutoff within EEQ (25.0)."""

D4_CN_EEQ_MAX = 8.0
"""Maximum coordination number (8.0)."""

D4_DISP2_CUTOFF = 60.0
"""Two-body interaction cutoff (60.0)."""

D4_DISP3_CUTOFF = 40.0
"""Three-body interaction cutoff (40.0)."""

D4_KCN = 7.5
"""Steepness of counting function (7.5)."""

D4_K4 = 4.10451
"""Parameter for electronegativity scaling."""

D4_K5 = 19.08857
"""Parameter for electronegativity scaling."""

D4_K6 = 2 * 11.28174**2  # 254.56
"""Parameter for electronegativity scaling."""

# DFT-D4 damping parameters

A1 = 0.4
"""Scaling for the C8 / C6 ratio in the critical radius (0.4)."""

A2 = 5.0
"""Offset parameter for the critical radius (5.0)."""

S6 = 1.0
"""Default scaling of dipole-dipole term (1.0 to retain correct limit)."""

S8 = 1.0
"""Default scaling of dipole-quadrupole term (1.0)."""

S9 = 1.0
"""Default scaling of three-body term (1.0)."""

S10 = 0.0
"""Default scaling of quadrupole-quadrupole term (0.0)."""

RS9 = 4.0 / 3.0
"""Scaling for van-der-Waals radii in damping function (4.0/3.0)."""

ALP = 16.0
"""Exponent of zero damping function (16.0)."""
