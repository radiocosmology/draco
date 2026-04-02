"""Useful functions for radio interferometry.

NOTE: this is somewhat redundant with stuff in `_fast_tools`,
so eventually they'll get merged.

Interferometry
--------------
- :py:meth:`fringestop_phase`
"""

import numpy as np
from caput.astro.coordinates import spherical


def fringestop_phase(ha, lat, dec, u, v, w=0.0):
    """Return the phase required to fringestop. All angle inputs are radians.

    Note that for a visibility V_{ij} = < E_i E_j^*>, this expects the u, v,
    w coordinates are the components of (d_i - d_j) / lambda.

    Parameters
    ----------
    ha : array_like
        The Hour Angle of the source to fringestop too.
    lat : array_like
        The latitude of the observatory.
    dec : array_like
        The declination of the source.
    u : array_like
        The EW separation in wavelengths (increases to the E)
    v : array_like
        The NS separation in wavelengths (increases to the N)
    w : array_like, optional
        The vertical separation on wavelengths (increases to the sky!)

    Returns
    -------
    phase : np.ndarray
        The phase required to *correct* the fringeing. Shape is
        given by the broadcast of the arguments together.
    """
    phase = -2.0j * np.pi * spherical.projected_distance(ha, lat, dec, u, v, w)

    return np.exp(phase, out=phase)
