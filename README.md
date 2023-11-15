Iterative Ensemble Smoother
===========================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/equinor/iterative_ensemble_smoother/blob/main/COPYING)
[![Stars](https://img.shields.io/github/stars/equinor/iterative_ensemble_smoother.svg?style=social&label=Star&maxAge=2592000)](https://github.com/equinor/iterative_ensemble_smoother/stargazers)
[![Python](https://img.shields.io/pypi/pyversions/iterative_ensemble_smoother.svg)](https://pypi.org/pypi/iterative_ensemble_smoother)
[![PyPI](https://img.shields.io/pypi/v/iterative_ensemble_smoother.svg)](https://pypi.org/pypi/iterative_ensemble_smoother)
[![Downloads](https://static.pepy.tech/badge/iterative_ensemble_smoother)](https://pepy.tech/project/iterative_ensemble_smoother)
[![Build Status](https://github.com/equinor/iterative_ensemble_smoother/actions/workflows/upload_to_pypi.yml/badge.svg)](https://github.com/equinor/iterative_ensemble_smoother/actions/workflows/main.yml)
[![Precommit: enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![docs](https://readthedocs.org/projects/iterative_ensemble_smoother/badge/?version=latest&style=plastic)](https://iterative-ensemble-smoother.readthedocs.io/)

## About

**iterative_ensemble_smoother** is a Python package that implements the subspace iterative ensemble smoother as described in [evensen2019](https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full).
This algorithm is particularly effective for problems with a large number of parameters (e.g., millions) and a few realizations or samples (e.g., hundreds).

## Installation

**iterative_ensemble_smoother** is on PyPi and can be installed using pip:

```text
pip install iterative_ensemble_smoother
```

If you want to do development, then run:

```text
git clone https://github.com/equinor/iterative_ensemble_smoother.git
cd iterative_ensemble_smoother
<create environment>
pip install --editable '.[doc,dev]'
```

## Usage

**iterative_ensemble_smoother** mainly implements the two classes `SIES` and `ESMDA`.
Check out the examples section to see how to use them.

## Building the documentation

```bash
apt install pandoc # Pandoc is required to build the documentation.
pip install .[doc]
sphinx-build -c docs/source/ -b html docs/source/ docs/build/html/
```
