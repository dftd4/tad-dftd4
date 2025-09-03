Torch autodiff for DFT-D4
=========================

.. image:: https://img.shields.io/badge/python-%3E=3.8-blue.svg
    :target: https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue.svg
    :alt: Python Versions

.. image:: https://img.shields.io/github/v/release/dftd4/tad-dftd4
    :target: https://github.com/dftd4/tad-dftd4/releases/latest
    :alt: Release

.. image:: https://img.shields.io/pypi/v/tad-dftd4
    :target: https://pypi.org/project/tad-dftd4/
    :alt: PyPI

.. image:: https://img.shields.io/conda/vn/conda-forge/tad-dftd4.svg
    :target: https://anaconda.org/conda-forge/tad-dftd4
    :alt: PyPI

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: http://www.apache.org/licenses/LICENSE-2.0
    :alt: Apache-2.0

.. image:: https://github.com/dftd4/tad-dftd4/actions/workflows/ubuntu.yaml/badge.svg
    :target: https://github.com/dftd4/tad-dftd4/actions/workflows/ubuntu.yaml
    :alt: Test Status Ubuntu

.. image:: https://github.com/dftd4/tad-dftd4/actions/workflows/macos-x86.yaml/badge.svg
    :target: https://github.com/dftd4/tad-dftd4/actions/workflows/macos-x86.yaml
    :alt: Test Status macOS (x86)

.. image:: https://github.com/dftd4/tad-dftd4/actions/workflows/macos-arm.yaml/badge.svg
    :target: https://github.com/dftd4/tad-dftd4/actions/workflows/macos-arm.yaml
    :alt: Test Status macOS (ARM)

.. image:: https://github.com/dftd4/tad-dftd4/actions/workflows/windows.yaml/badge.svg
    :target: https://github.com/dftd4/tad-dftd4/actions/workflows/windows.yaml
    :alt: Test Status Windows

.. image:: https://readthedocs.org/projects/tad-dftd4/badge/?version=latest
    :target: https://tad-dftd4.readthedocs.io
    :alt: Documentation Status

.. image:: https://results.pre-commit.ci/badge/github/dftd4/tad-dftd4/main.svg
    :target: https://results.pre-commit.ci/latest/github/dftd4/tad-dftd4/main
    :alt: pre-commit.ci status

.. image:: https://codecov.io/gh/dftd4/tad-dftd4/branch/main/graph/badge.svg?token=OGJJnZ6t4G
    :target: https://codecov.io/gh/dftd4/tad-dftd4
    :alt: Coverage


Implementation of the DFT-D4 dispersion model in PyTorch.
This module allows to process a single structure or a batch of structures for the calculation of atom-resolved dispersion energies.

References
----------

For using DFT-D4 in the automatic differentiation framework:

- \M. Friede, C. Hölzer, S. Ehlert, S. Grimme, *J. Chem. Phys.*, **2024**, *161*, 062501. DOI: `10.1063/5.0216715 <https://doi.org/10.1063/5.0216715>`__


For the DFT-D4 dispersion model:

- \E. Caldeweyher, C. Bannwarth and S. Grimme, *J. Chem. Phys.*, **2017**, *147*, 034112. DOI: `10.1063/1.4993215 <https://dx.doi.org/10.1063/1.4993215>`__

- \E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, C. Bannwarth and S. Grimme, *J. Chem. Phys.*, **2019**, *150*, 154122. DOI: `10.1063/1.5090222 <https://dx.doi.org/10.1063/1.5090222>`__

- \E. Caldeweyher, J.-M. Mewes, S. Ehlert and S. Grimme, *Phys. Chem. Chem. Phys.*, **2020**, *22*, 8499-8512. DOI: `10.1039/D0CP00502A <https://doi.org/10.1039/D0CP00502A>`__


Getting Started
---------------

Check out the :doc:`installation` instructions to get started.

For usage examples, see the :doc:`examples` section.


.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   examples
   disp
   modules/index
