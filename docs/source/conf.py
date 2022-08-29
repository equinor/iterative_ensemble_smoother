from subprocess import check_output

project = "iterative_ensemble_smoother"
copyright = "2022, Equinor"
author = "Equinor"
release = "0.0.1"


check_output(["jupytext", "Polynomial.py", "-o", "Polynomial.ipynb"])

check_output(["jupytext", "Oscillator.py", "-o", "Oscillator.ipynb"])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "nbsphinx",
    "sphinxcontrib.bibtex",
]
bibtex_bibfiles = ["refs.bib"]
language = "python"
html_theme = "sphinx_rtd_theme"
