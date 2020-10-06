from caput import pipeline, memh5

import caput
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
    """.format(
        TAG
    )

    man = pipeline.Manager.from_yaml_str(testconfig)
    man.run()

    # Do the same using caput.memh5 to make sure it deserializes it
    with memh5.MemDiskGroup.from_file("{}.h5".format(TAG)) as m:
        configdump = m.history["config"]
        versiondump = m.history["versions"]
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
