# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np

import versioneer


# Enable OpenMP support if available
if sys.platform == "darwin":
    compile_args = []
    link_args = []
else:
    compile_args = ["-fopenmp"]
    link_args = ["-fopenmp"]

# Cython module for fast operations
fast_ext = Extension(
    "draco.util._fast_tools",
    ["draco/util/_fast_tools.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

trunc_ext = Extension(
    "draco.util.truncate",
    ["draco/util/truncate.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)


# Load the PEP508 formatted requirements from the requirements.txt file. Needs
# pip version > 19.0
with open("requirements.txt", "r") as fh:
    requires = fh.readlines()

setup(
    name="draco",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    packages=find_packages(),
    ext_modules=cythonize([fast_ext, trunc_ext]),
    install_requires=requires,
    author="Richard Shaw",
    author_email="richard@phas.ubc.ca",
    description="Analysis and simulation tools for driftscan radio interferometers.",
    url="http://github.com/radiocosmology/draco/",
)
