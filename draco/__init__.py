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

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
