"""Build cython extensions.

The full project config can be found in `pyproject.toml`. `setup.py` is still
required to build cython extensions.
"""

import os
import subprocess
import sysconfig
import tempfile

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Subset of `-ffast-math` compiler flags which should
# preserve IEEE compliance
FAST_MATH_ARGS = ["-O3", "-fno-math-errno", "-fno-trapping-math", "-march=native"]


def _compiler_supports_openmp():
    cc = os.environ.get("CC", sysconfig.get_config_var("CC"))
    if cc is None:
        return False

    test_code = r"""
    #include <omp.h>
    int main(void) {
        return omp_get_max_threads();
    }
    """

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "test.c")
        exe = os.path.join(d, "test")
        with open(src, "w") as f:
            f.write(test_code)

        cmd = [cc, src, "-fopenmp", "-o", exe]
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except Exception:  # noqa: BLE001
            return False

    return True


OMP_ARGS = []

if not os.environ.get("DRACO_NO_OPENMP"):
    if _compiler_supports_openmp():
        OMP_ARGS = ["-fopenmp"]
    else:
        cc = os.environ.get("CC", sysconfig.get_config_var("CC"))
        print(
            f"Compiler `{cc}` does not support OpenMP. "
            "If an OpenMP-supporting compiler is available, "
            "add it to your PATH or use `CC=<compiler> pip install ...` "
            "Alternatively, to suppress this warning, set the environment "
            "variable DRACO_NO_OPENMP to any truth-like value."
        )


# Cython module for fast operations
extensions = [
    Extension(
        "draco.util._fast_tools",
        ["draco/util/_fast_tools.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[*FAST_MATH_ARGS, *OMP_ARGS],
        extra_link_args=[*FAST_MATH_ARGS, *OMP_ARGS],
    ),
]

setup(
    name="draco",  # required
    ext_modules=cythonize(extensions),
)
