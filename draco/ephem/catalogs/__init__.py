"""CHIME ephemeris source catalogues

Functions
=========

- :py:meth:`list`
- :py:meth:`load`
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import glob
import json
import pathlib

if TYPE_CHECKING:
    from collections.abc import Iterable
del TYPE_CHECKING


def list() -> list[str]:
    """List available catalogues.

    Returns
    -------
    catalogs : list
        A list with the names of available catalogs
    """

    cats = sorted(glob.glob("*.json", root_dir=pathlib.Path(__file__).parent))

    # Strip ".json" off the end
    return [cat[:-5] for cat in cats]


def load(name: str) -> Iterable:
    """Read the named catalogue and return the parsed JSON representation.

    Parameters
    ----------
    name : str
        The name of the catalogue to load.  This is the name of a JSON file
        in the `ch_ephem/catalogs` directory, excluding the `.json` suffix.

    Returns
    -------
    catalogue: Iterable
        The parsed catalogue.

    Raises
    ------
    JSONDecodeError:
        An error occurred while trying to parse the catalogue.
    ValueError:
        No catalogue with the given name could be found.
    """
    path = pathlib.Path(__file__).with_name(name + ".json")

    try:
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ValueError(f"No such catalogue: {name}") from e
