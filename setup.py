"""Build cython extensions.

The full project config can be found in `pyproject.toml`. `setup.py` is still
required to build cython extensions.
"""

import re
import sysconfig

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Enable OpenMP support if available
if re.search("gcc", sysconfig.get_config_var("CC")) is None:
    print("Not using OpenMP")
    omp_args = []
else:
    omp_args = ["-fopenmp"]

# Cython module for fast operations
extensions = [
    Extension(
        "draco.util._fast_tools",
        ["draco/util/_fast_tools.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=omp_args,
        extra_link_args=omp_args,
    ),
    Extension(
        "draco.util.truncate",
        ["draco/util/truncate.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=omp_args,
        extra_link_args=omp_args,
    ),
]

setup(
    name="draco",  # required
    ext_modules=cythonize(extensions),
)
