from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np

# Cython module for fast regridding
rg_ext = Extension(
    "ch_pipeline._regrid_work",
    ["ch_pipeline/_regrid_work.pyx"],
    include_dirs = [np.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name = 'ch_pipeline',
    version = 0.1,

    packages = find_packages(),
    package_data = { "ch_pipeline" : [ "data/*" ] },

    ext_modules = cythonize([rg_ext]),

    author = "CHIME collaboration",
    author_email = "jrs65@cita.utoronto.ca",
    description = "CHIME Pipeline",
    url = "http://bitbucket.org/chime/ch_pipeline/",

    package_data = { "ch_pipeline" : [ "data/*" ] }
)
