"""CHIME source objects.

Note: This submodule reads the CHIME radio source catalogues from
disk at import time.

Constants
=========

:const:`source_dictionary`
    The standard CHIME radio source catalogue.  A `dict`
    whose keys are source names and whose values are
    `skyfield.starlib.Star` objects.
:const:`CasA`
    :class:`skyfield.starlib.Star` representing Cassiopeia A.
:const:`CygA`
    :class:`skyfield.starlib.Star` representing Cygnus A.
:const:`TauA`
    :class:`skyfield.starlib.Star` representing Taurus A.
:const:`VirA`
    :class:`skyfield.starlib.Star` representing Virgo A.

Functions
=========

- :py:meth:`get_source_dictionary`
"""

from __future__ import annotations

from caput.time import skyfield_star_from_ra_dec

from .catalogs import load


def get_source_dictionary(*catalogs: str) -> dict:
    """Returns a dictionary containing :class:`skyfield.starlib.Star`
    objects for common radio point sources.  This is useful for
    obtaining the skyfield representation of a source from a string
    containing its name.

    Parameters
    ----------
    catalogs : str
        Catalogue names.  These must be the basename of the json file
        in the `ch_ephem/catalogs` directory.  If multiple catalogues
        are provided, the first catalogue is favoured for any
        overlapping sources.

    Returns
    -------
    src_dict : dict
        Keys are source names.  Values are `skyfield.starlib.Star`
        objects.
    """

    src_dict = {}
    for catalog_name in reversed(catalogs):
        catalog = load(catalog_name)

        for name, info in catalog.items():
            names = info["alternate_names"]
            if name not in names:
                names = [name] + names
            src_dict[name] = skyfield_star_from_ra_dec(
                info["ra"], info["dec"], tuple(names)
            )

    return src_dict


# Common radio point sources
source_dictionary = get_source_dictionary(
    "primary_calibrators_perley2016",
    "specfind_v2_5Jy_vollmer2009",
    "atnf_psrcat",
    "hfb_target_list",
)

# Calibrators
CasA = source_dictionary["CAS_A"]
CygA = source_dictionary["CYG_A"]
TauA = source_dictionary["TAU_A"]
VirA = source_dictionary["VIR_A"]
