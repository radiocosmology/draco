# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from caput import pipeline, memh5

import caput
import h5py
import json
import numpy
import yaml

TAG = "test"


def test_metadata_to_hdf5():
    """Check if metadata is written to HDF5."""

    testconfig = """
    foo: bar
    pipeline:
        save_versions:
            - numpy
            - caput
        tasks:
            - type: draco.util.testing.DummyTask
              params:
                tag: {}
                save: Yes
    """.format(TAG)

    man = pipeline.Manager.from_yaml_str(testconfig)
    man.run()

    # Check HDF5 file for config- and versiondump
    f = h5py.File("{}.h5".format(TAG), "r")
    configdump = f.attrs['config_json']
    versiondump = f.attrs['versions_json']
    assert versiondump == json.dumps({"numpy": numpy.__version__, "caput": caput.__version__})
    assert configdump == json.dumps(yaml.load(testconfig, Loader=yaml.SafeLoader))

    # Do the same using caput.memh5 to make sure it deserializes it
    m = memh5.MemDiskGroup.from_file("{}.h5".format(TAG))
    configdump = m.attrs['config_json']
    versiondump = m.attrs['versions_json']
    assert versiondump == {"numpy": numpy.__version__, "caput": caput.__version__}
    assert configdump == yaml.load(testconfig, Loader=yaml.SafeLoader)
