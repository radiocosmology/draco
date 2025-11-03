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
    OMP_ARGS = []
else:
    OMP_ARGS = ["-fopenmp"]

# Subset of `-ffast-math` compiler flags which should
# preserve IEEE compliance
FAST_MATH_ARGS = ["-O3", "-fno-math-errno", "-fno-trapping-math", "-march=native"]

# Cython module for fast operations
extensions = [
    Extension(
        "draco.util._fast_tools",
        ["draco/util/_fast_tools.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[*FAST_MATH_ARGS, *OMP_ARGS],
        extra_link_args=[*FAST_MATH_ARGS, *OMP_ARGS],
    ),
    Extension(
        "draco.util.truncate",
        ["draco/util/truncate.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=OMP_ARGS,
        extra_link_args=OMP_ARGS,
    ),
]

setup(
    name="draco",  # required
    ext_modules=cythonize(extensions),
)
