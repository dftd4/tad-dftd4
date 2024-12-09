# Torch Autodiff for DFT-D4

<table>
  <tr>
    <td>Compatibility:</td>
    <td>
      <img src="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue.svg" alt="Python Versions"/>
      <img src="https://img.shields.io/badge/PyTorch-%3E=1.11.0-blue.svg" alt="PyTorch Versions"/>
    </td>
  </tr>
  <tr>
    <td>Availability:</td>
    <td>
      <a href="https://github.com/dftd4/tad-dftd4/releases/latest">
        <img src="https://img.shields.io/github/v/release/dftd4/tad-dftd4?color=orange" alt="Release"/>
      </a>
      <a href="https://pypi.org/project/tad-dftd4/">
        <img src="https://img.shields.io/pypi/v/tad-dftd4?color=orange" alt="PyPI"/>
      </a>
      <a href="https://anaconda.org/conda-forge/tad-dftd4">
        <img src="https://img.shields.io/conda/vn/conda-forge/tad-dftd4.svg" alt="Conda Version"/>
      </a>
      <a href="http://www.apache.org/licenses/LICENSE-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="Apache-2.0"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>Status:</td>
    <td>
      <a href="https://github.com/dftd4/tad-dftd4/actions/workflows/ubuntu.yaml">
        <img src="https://github.com/dftd4/tad-dftd4/actions/workflows/ubuntu.yaml/badge.svg" alt="Test Status Ubuntu"/>
      </a>
      <a href="https://github.com/dftd4/tad-dftd4/actions/workflows/macos.yaml">
        <img src="https://github.com/dftd4/tad-dftd4/actions/workflows/macos-x86.yaml/badge.svg" alt="Test Status macOS (x86)"/>
      </a>
      <a href="https://github.com/dftd4/tad-dftd4/actions/workflows/macos-arm.yaml">
        <img src="https://github.com/dftd4/tad-dftd4/actions/workflows/macos-arm.yaml/badge.svg" alt="Test Status macOS (ARM)"/>
      </a>
      <a href="https://github.com/dftd4/tad-dftd4/actions/workflows/windows.yaml">
        <img src="https://github.com/dftd4/tad-dftd4/actions/workflows/windows.yaml/badge.svg" alt="Test Status Windows"/>
      </a>
      <a href="https://github.com/dftd4/tad-dftd4/actions/workflows/release.yaml">
        <img src="https://github.com/dftd4/tad-dftd4/actions/workflows/release.yaml/badge.svg" alt="Build Status"/>
      </a>
      <a href="https://tad-dftd4.readthedocs.io">
        <img src="https://readthedocs.org/projects/tad-dftd4/badge/?version=latest" alt="Documentation Status"/>
      </a>
      <a href="https://results.pre-commit.ci/latest/github/dftd4/tad-dftd4/main">
        <img src="https://results.pre-commit.ci/badge/github/dftd4/tad-dftd4/main.svg" alt="pre-commit.ci Status"/>
      </a>
      <a href="https://codecov.io/gh/dftd4/tad-dftd4">
        <img src="https://codecov.io/gh/dftd4/tad-dftd4/branch/main/graph/badge.svg?token=OGJJnZ6t4G" alt="Coverage"/>
      </a>
    </td>
  </tr>
</table>

<br>

Implementation of the DFT-D4 dispersion model in PyTorch. This module allows to process a single structure or a batch of structures for the calculation of atom-resolved dispersion energies.

If you use this software, please cite the following publication

- M. Friede, C. Hölzer, S. Ehlert, S. Grimme, *J. Chem. Phys.*, **2024**, *161*, 062501. DOI: [10.1063/5.0216715](https://doi.org/10.1063/5.0216715)


For details on the D4 dispersion model, see:

- E. Caldeweyher, C. Bannwarth and S. Grimme, *J. Chem. Phys.*, 2017, 147, 034112. [DOI: 10.1063/1.4993215](https://dx.doi.org/10.1063/1.4993215)
- E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, C. Bannwarth and S. Grimme, *J. Chem. Phys.*, 2019, 150, 154122. [DOI: 10.1063/1.5090222](https://dx.doi.org/10.1063/1.5090222)
- E. Caldeweyher, J.-M. Mewes, S. Ehlert and S. Grimme, *Phys. Chem. Chem. Phys.*, 2020, 22, 8499-8512. [DOI: 10.1039/D0CP00502A](https://doi.org/10.1039/D0CP00502A)


For alternative implementations, also check out:

- [dftd4](https://dftd4.readthedocs.io): Implementation of the DFT-D4 dispersion model in Fortran with Python bindings.
- [cpp-d4](https://github.com/dftd4/cpp-d4): Implementation of the DFT-D4 dispersion model in C++.


## Installation

### pip

*tad-dftd4* can easily be installed with ``pip``.

```sh
pip install tad-dftd4
```

### conda

*tad-dftd4* is also available from ``conda``.

```sh
conda install tad-dftd4
```

### From source

This project is hosted on GitHub at [dftd4/tad-dftd4](https://github.com/dftd4/tad-dftd4).
Obtain the source by cloning the repository with

```sh
git clone https://github.com/dftd4/tad-dftd4
cd tad-dftd4
```

We recommend using a [conda](https://conda.io/) environment to install the package.
You can setup the environment manager using a [mambaforge](https://github.com/conda-forge/miniforge) installer.
Install the required dependencies from the conda-forge channel.

```sh
mamba env create -n torch -f environment.yaml
mamba activate torch
```

Install this project with ``pip`` in the environment

```sh
pip install .
```

The following dependencies are required

- [numpy](https://numpy.org/)
- [tad-mctc](https://github.com/tad-mctc/tad-mctc/)
- [tad-multicharge](https://github.com/tad-mctc/tad-multicharge/)
- [torch](https://pytorch.org/)
- [pytest](https://docs.pytest.org/) (tests only)


## Compatibility

| PyTorch \ Python | 3.8                | 3.9                | 3.10               | 3.11               | 3.12               |
|------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1.11.0           | :white_check_mark: | :white_check_mark: | :x:                | :x:                | :x:                |
| 1.12.1           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:                |
| 1.13.1           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.0.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.1.2            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.2.2            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.3.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.4.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.5.1            | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

Note that only the latest bug fix version is listed, but all preceding bug fix minor versions are supported.
For example, although only version 2.2.2 is listed, version 2.2.0 and 2.2.1 are also supported.

On macOS and Windows, PyTorch<2.0.0 does only support Python<3.11.


## Development

For development, additionally install the following tools in your environment.

```sh
mamba install black covdefaults mypy pre-commit pylint pytest pytest-cov pytest-xdist tox
pip install pytest-random-order
```

With pip, add the option ``-e`` for installing in development mode, and add ``[dev]`` for the development dependencies

```sh
pip install -e .[dev]
```

The pre-commit hooks are initialized by running the following command in the root of the repository.

```sh
pre-commit install
```

For testing all Python environments, simply run `tox`.

```sh
tox
```

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional *posargs*.

```sh
tox -- test
```

## Examples

The following example shows how to calculate the DFT-D4 dispersion energy for a single structure.

```python
import torch
import tad_dftd4 as d4
import tad_mctc as mctc

numbers = mctc.convert.symbol_to_number(symbols="C C C C N C S H H H H H".split())

# coordinates in Bohr
positions = torch.tensor(
    [
        [-2.56745685564671, -0.02509985979910, 0.00000000000000],
        [-1.39177582455797, +2.27696188880014, 0.00000000000000],
        [+1.27784995624894, +2.45107479759386, 0.00000000000000],
        [+2.62801937615793, +0.25927727028120, 0.00000000000000],
        [+1.41097033661123, -1.99890996077412, 0.00000000000000],
        [-1.17186102298849, -2.34220576284180, 0.00000000000000],
        [-2.39505990368378, -5.22635838332362, 0.00000000000000],
        [+2.41961980455457, -3.62158019253045, 0.00000000000000],
        [-2.51744374846065, +3.98181713686746, 0.00000000000000],
        [+2.24269048384775, +4.24389473203647, 0.00000000000000],
        [+4.66488984573956, +0.17907568006409, 0.00000000000000],
        [-4.60044244782237, -0.17794734637413, 0.00000000000000],
    ]
)

# total charge of the system
charge = torch.tensor(0.0)

# TPSSh-D4-ATM parameters
param = {
    "s6": positions.new_tensor(1.0),
    "s8": positions.new_tensor(1.85897750),
    "s9": positions.new_tensor(1.0),
    "a1": positions.new_tensor(0.44286966),
    "a2": positions.new_tensor(4.60230534),
}

# parameters can also be obtained using the functional name:
# param = d4.get_params("tpssh")

energy = d4.dftd4(numbers, positions, charge, param)
torch.set_printoptions(precision=10)
print(energy)
# tensor([-0.0020841344, -0.0018971195, -0.0018107513, -0.0018305695,
#         -0.0021737693, -0.0019484236, -0.0022788253, -0.0004080658,
#         -0.0004261866, -0.0004199839, -0.0004280768, -0.0005108935])
```

The next example shows the calculation of dispersion energies for a batch of structures.

```python

import torch
import tad_dftd4 as d4
import tad_mctc as mctc

# S22 system 4: formamide dimer
numbers = mctc.batch.pack((
    mctc.convert.symbol_to_number("C C N N H H H H H H O O".split()),
    mctc.convert.symbol_to_number("C O N H H H".split()),
))

# coordinates in Bohr
positions = mctc.batch.pack((
    torch.tensor([
        [-3.81469488143921, +0.09993441402912, 0.00000000000000],
        [+3.81469488143921, -0.09993441402912, 0.00000000000000],
        [-2.66030049324036, -2.15898251533508, 0.00000000000000],
        [+2.66030049324036, +2.15898251533508, 0.00000000000000],
        [-0.73178529739380, -2.28237795829773, 0.00000000000000],
        [-5.89039325714111, -0.02589114569128, 0.00000000000000],
        [-3.71254944801331, -3.73605775833130, 0.00000000000000],
        [+3.71254944801331, +3.73605775833130, 0.00000000000000],
        [+0.73178529739380, +2.28237795829773, 0.00000000000000],
        [+5.89039325714111, +0.02589114569128, 0.00000000000000],
        [-2.74426102638245, +2.16115570068359, 0.00000000000000],
        [+2.74426102638245, -2.16115570068359, 0.00000000000000],
    ]),
    torch.tensor([
        [-0.55569743203406, +1.09030425468557, 0.00000000000000],
        [+0.51473634678469, +3.15152550263611, 0.00000000000000],
        [+0.59869690244446, -1.16861263789477, 0.00000000000000],
        [-0.45355203669134, -2.74568780438064, 0.00000000000000],
        [+2.52721209544999, -1.29200800956867, 0.00000000000000],
        [-2.63139587595376, +0.96447869452240, 0.00000000000000],
    ]),
))

# total charge of both system
charge = torch.tensor([0.0, 0.0])

# TPSSh-D4-ATM parameters
param = {
    "s6": positions.new_tensor(1.0),
    "s8": positions.new_tensor(1.85897750),
    "s9": positions.new_tensor(1.0),
    "a1": positions.new_tensor(0.44286966),
    "a2": positions.new_tensor(4.60230534),
}

# calculate dispersion energy in Hartree
energy = torch.sum(d4.dftd4(numbers, positions, charge, param), -1)
torch.set_printoptions(precision=10)
print(energy)
# tensor([-0.0088341432, -0.0027013607])
print(energy[0] - 2*energy[1])
# tensor(-0.0034314217)
```

## Contributing

This is a volunteer open source projects and contributions are always welcome.
Please, take a moment to read the [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
