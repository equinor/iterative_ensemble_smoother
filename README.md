Iterative Ensemble Smoother
===========================

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![docs](https://readthedocs.org/projects/iterative_ensemble_smoother/badge/?version=latest&style=plastic)](https://iterative_ensemble_smoother.readthedocs.io/)

A library for the iterative ensemble smoother algorithm.
For more information, see [the docs](http://iterative_ensemble_smoother.rtfd.io).

## building

Before building you will have to have a c compiler and python with pip
installed. Depending on your environment you will have to instruct
the conan package manager about which c compiler to use.

```bash
c++ --version # to get information about which c compiler is installed
              # on your system.
pip install conan

# The following sets the compiler settings
# assuming the output of `c++ --version` indicated
# that the installed c compiler was gcc version 11.2
# and writes it to the default profile.
conan profile update "settings.compiler=gcc" default
conan profile update "settings.compiler=11.2" default
```

To build iterative_ensemble_moother from source:

```bash
git clone https://github.com/equinor/iterative_ensemble_moother.git
cd iterative_ensemble_smoother
pip install .
```

### build the documentation

```bash
apt install pandoc # Building the doc requires pandoc
pip install .[doc]
spinx-build -c docs/source/ -b html docs/source/ docs/build/html/
```
