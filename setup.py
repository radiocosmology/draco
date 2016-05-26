import sys

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np

# Enable OpenMP support if available
if sys.platform == 'darwin':
    compile_args = []
    link_args = []
else:
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']

# Cython module for fast operations
fast_ext = Extension(
    "ch_pipeline.util._fast_tools",
    ["ch_pipeline/util/_fast_tools.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name='ch_pipeline',
    version=0.1,

    packages=find_packages(),
    package_data={ "ch_pipeline": [ "data/*" ] },

    ext_modules=cythonize([fast_ext]),

    author="CHIME collaboration",
    author_email="richard@phas.ubc.ca",
    description="CHIME Pipeline",
    url="http://bitbucket.org/chime/ch_pipeline/",
)
