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
JSON_PREFIX = "!!_memh5_json:"


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
    """.format(
        TAG
    )

    man = pipeline.Manager.from_yaml_str(testconfig)
    man.run()

    # Check HDF5 file for config- and versiondump
    f = h5py.File("{}.h5".format(TAG), "r")
    configdump = f.attrs["config"]
    versiondump = f.attrs["versions"]
    assert versiondump == JSON_PREFIX + json.dumps(
        {"numpy": numpy.__version__, "caput": caput.__version__}
    )
    assert configdump == JSON_PREFIX + json.dumps(
        yaml.load(testconfig, Loader=yaml.SafeLoader)
    )

    # Do the same using caput.memh5 to make sure it deserializes it
    m = memh5.MemDiskGroup.from_file("{}.h5".format(TAG))
    configdump = m.attrs["config"]
    versiondump = m.attrs["versions"]
    assert versiondump == {"numpy": numpy.__version__, "caput": caput.__version__}
    assert configdump == yaml.load(testconfig, Loader=yaml.SafeLoader)


def test_metadata_to_yaml():
    """Check if metadata is written to YAML file."""

    testconfig = """
    foo: bar
    pipeline:
        save_versions:
            - numpy
            - caput
        tasks:
            - type: draco.core.io.SaveModuleVersions
              params:
                root: {0}
            - type: draco.core.io.SaveConfig
              params:
                root: {0}
    """.format(
        TAG
    )

    man = pipeline.Manager.from_yaml_str(testconfig)
    man.run()

    # Check yaml files for config- and versiondump
    yaml_config = open("{}_config.yml".format(TAG), "r")
    yaml_versions = open("{}_versions.yml".format(TAG), "r")
    configdump = yaml_config.read()
    versiondump = yaml_versions.read()
    yaml_config.close()
    yaml_versions.close()

    assert versiondump == yaml.dump(
        {"numpy": numpy.__version__, "caput": caput.__version__}
    )

    # let pyyaml fix the indentation by loading and dumping again
    assert configdump == yaml.dump(yaml.safe_load(testconfig))
