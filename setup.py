from setuptools import setup, find_packages
from Cython.Build import cythonize

import numpy as np


setup(
    name = 'ch_pipeline',
    version = 0.1,

    packages = find_packages(),
    package_data = { "ch_pipeline" : [ "data/*" ] },

    include_dirs = [np.get_include()],
    ext_modules = cythonize('ch_pipeline/_regrid_work.pyx'),

    author = "CHIME collaboration",
    author_email = "jrs65@cita.utoronto.ca",
    description = "CHIME Pipeline",
    url = "http://bitbucket.org/chime/ch_pipeline/",

)
