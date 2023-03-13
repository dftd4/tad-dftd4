Installation
------------

pip
~~~

*tad-dftd4* can easily be installed with ``pip``.

.. code::

    pip install tad-dftd4


From source
~~~~~~~~~~~

This project is hosted on GitHub at `dftd4/tad-dftd4 <https://github.com/dftd4/tad-dftd4>`__.
Obtain the source by cloning the repository with

.. code::

    git clone https://github.com/dftd4/tad-dftd4
    cd tad-dftd4

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code::

    mamba env create -n torch -f environment.yaml
    mamba activate torch

Development
~~~~~~~~~~~

For development, additionally install the following tools in your environment.

.. code::

    mamba install black covdefaults coverage mypy pre-commit pylint tox

With pip, add the option ``-e`` and the development dependencies for installing in development mode.

.. code::

    pip install -e .[dev]

The pre-commit hooks are initialized by running the following command in the root of the repository.

.. code::

    pre-commit install

For testing all Python environments, simply run `tox`.

.. code::

    tox

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional _posargs_.

.. code::

    tox -- test
