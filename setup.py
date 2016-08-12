import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np

import draco

# Enable OpenMP support if available
if sys.platform == 'darwin':
    compile_args = []
    link_args = []
else:
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']

# Cython module for fast operations
fast_ext = Extension(
    "draco.util._fast_tools",
    ["draco/util/_fast_tools.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name='draco',
    version=draco.__version__,
    license='MIT',

    packages=find_packages(),

    ext_modules=cythonize([fast_ext]),

    install_requires=['Cython>0.18', 'numpy>=1.7', 'scipy>=0.10',
                      'caput>=0.3', 'cora', 'driftscan>=0.2'],

    author="Richard Shaw",
    author_email="richard@phas.ubc.ca",
    description="Analysis and simulation tools for driftscan radio interferometers.",
    url="http://github.com/radiocosmology/draco/",
)
