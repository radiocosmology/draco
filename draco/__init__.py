"""draco.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    analysis
    core
    synthesis
    util
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("draco")
except PackageNotFoundError:
    # package is not installed
    pass

del version, PackageNotFoundError
