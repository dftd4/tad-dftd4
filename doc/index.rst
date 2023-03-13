Torch autodiff for DFT-D4
=========================

.. image:: https://img.shields.io/badge/python-%3E=3.8-blue.svg
    :target: https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue.svg
    :alt: Python Versions

.. image:: https://img.shields.io/github/v/release/dftd4/tad-dftd4
    :target: https://github.com/dftd4/tad-dftd4/releases/latest
    :alt: Release

.. image:: https://img.shields.io/pypi/v/tad-dftd4
    :target: https://pypi.org/project/tad-dftd4/
    :alt: PyPI

.. image:: https://img.shields.io/badge/License-LGPL_v3-blue.svg
    :target: https://www.gnu.org/licenses/lgpl-3.0
    :alt: LGPL-3.0

.. image:: https://github.com/dftd4/tad-dftd4/actions/workflows/python.yaml/badge.svg
    :target: https://github.com/dftd4/tad-dftd4/actions/workflows/python.yaml
    :alt: CI

.. image:: https://readthedocs.org/projects/tad-dftd4/badge/?version=latest
    :target: https://tad-dftd4.readthedocs.io
    :alt: Documentation Status

.. image:: https://codecov.io/gh/marvinfriede/tad-dftd4/branch/master/graph/badge.svg?token=OGJJnZ6t4G
    :target: https://codecov.io/gh/marvinfriede/tad-dftd4
    :alt: Coverage

.. image:: https://results.pre-commit.ci/badge/github/dftd4/tad-dftd4/main.svg
    :target: https://results.pre-commit.ci/latest/github/dftd4/tad-dftd4/main
    :alt: pre-commit.ci status


Implementation of the DFT-D4 dispersion model in PyTorch.
This module allows to process a single structure or a batch of structures for the calculation of atom-resolved dispersion energies.

References
----------

- \E. Caldeweyher, C. Bannwarth and S. Grimme, *J. Chem. Phys.*, **2017**, *147*, 034112. DOI: `10.1063/1.4993215 <https://dx.doi.org/10.1063/1.4993215>`__

- \E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, C. Bannwarth and S. Grimme, *J. Chem. Phys.*, **2019**, *150*, 154122. DOI: `10.1063/1.5090222 <https://dx.doi.org/10.1063/1.5090222>`__

- \E. Caldeweyher, J.-M. Mewes, S. Ehlert and S. Grimme, *Phys. Chem. Chem. Phys.*, **2020**, *22*, 8499-8512. DOI: `10.1039/D0CP00502A <https://doi.org/10.1039/D0CP00502A>`__


Examples
--------

The following example shows how to calculate the DFT-D3 dispersion energy for a single structure.

.. code:: python

    import torch
    import tad_dftd4 as d4

    numbers = d4.utils.to_number(symbols="C C C C N C S H H H H H".split())

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

    # TPSS0-D4-ATM parameters
    param = {
        "s6": positions.new_tensor(1.0),
        "s8": positions.new_tensor(1.85897750),
        "s9": positions.new_tensor(1.0),
        "a1": positions.new_tensor(0.44286966),
        "a2": positions.new_tensor(4.60230534),
    }

    energy = d4.dftd4(numbers, positions, charge, param)
    torch.set_printoptions(precision=10)
    print(energy)
    # tensor([-0.0020841344, -0.0018971195, -0.0018107513, -0.0018305695,
    #         -0.0021737693, -0.0019484236, -0.0022788253, -0.0004080658,
    #         -0.0004261866, -0.0004199839, -0.0004280768, -0.0005108935])

The next example shows the calculation of dispersion energies for a batch of structures.

.. code:: python

    import torch
    import tad_dftd4 as d4

    # S22 system 4: formamide dimer
    numbers = d4.utils.pack((
        d4.utils.to_number("C C N N H H H H H H O O".split()),
        d4.utils.to_number("C O N H H H".split()),
    ))

    # coordinates in Bohr
    positions = d4.utils.pack((
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

    # TPSS0-D4-ATM parameters
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

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   modules/index
