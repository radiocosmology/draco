"""CHIME co-ordinate transformations

Functions
=========

- :py:meth:`cirs_radec`
- :py:meth:`star_cirs`
- :py:meth:`object_coords`
- :py:meth:`hadec_to_bmxy`
- :py:meth:`bmxy_to_hadec`
- :py:meth:`range_rate`
- :py:meth:`peak_ra`

"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import skyfield.starlib

from caput.time import skyfield_wrapper, unix_to_skyfield_time

if TYPE_CHECKING:
    from typing import Optional, Tuple, Union
    import skyfield.vectorlib
    import skyfield.jpllib
    import skyfield.timelib
    import skyfield.units
    import caput.time

    SkyfieldSource = Union[
        skyfield.starlib.Star,
        skyfield.vectorlib.VectorSum,
        skyfield.jpllib.ChebyshevPosition,
    ]

del TYPE_CHECKING


def object_coords(
    body: SkyfieldSource,
    obs: Optional[caput.time.Observer],
    date: Optional[float] = None,
    deg: bool = False
) -> Tuple[float, float]:
    """Calculates the RA and DEC of the source.

    Gives the ICRS coordinates if no date is given (=J2000), or if a date is
    specified gives the CIRS coordinates at that epoch.

    This also returns the *apparent* position, including abberation and
    deflection by gravitational lensing. This shifts the positions by up to
    20 arcseconds.

    Parameters
    ----------
    body : skyfield source
        skyfield.starlib.Star or skyfield.vectorlib.VectorSum or
        skyfield.jpllib.ChebyshevPosition body representing the source.
    obs : `caput.time.Observer` or None
        An observer instance to use.
    date : float
        Unix time at which to determine ra of source If None, use Jan 01
        2000.
    deg : bool
        Return RA ascension in degrees if True, radians if false (default).

    Returns
    -------
    ra, dec: float
        Position of the source.
    """

    if date is None:  # No date, get ICRS coords
        if isinstance(body, skyfield.starlib.Star):
            ra, dec = body.ra.radians, body.dec.radians
        else:
            raise ValueError(
                "Body is not fixed, cannot calculate coordinates without a date."
            )

    else:  # Calculate CIRS position with all corrections
        date = unix_to_skyfield_time(date)
        radec = obs.skyfield_obs().at(date).observe(body).apparent().cirs_radec(date)

        ra, dec = radec[0].radians, radec[1].radians

    # If requested, convert to degrees
    if deg:
        ra = np.degrees(ra)
        dec = np.degrees(dec)

    # Return
    return ra, dec


