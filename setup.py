from glob import glob
from os import path
from pathlib import Path
from subprocess import check_output

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

check_output(["conan", "install", "."])

ext_modules = [
    Pybind11Extension(
        "iterative_ensemble_smoother._ies",
        sorted(glob("src/iterative_ensemble_smoother/*.cpp")),
        cxx_std=17,
        include_dirs=[
            path.join(path.dirname(__file__), "src/iterative_ensemble_smoother/"),
        ],
        extra_compile_args=Path("conanbuildinfo.args").read_text().split(),
    ),
]

setup(cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
