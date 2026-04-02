from caput import config, pipeline, memdata

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
    """.format(TAG)

    man = pipeline.Manager.from_yaml_str(testconfig)
    man.run()

    # Do the same using caput.memdata to make sure it deserializes it
    with memdata.MemDiskGroup.from_file("{}.h5".format(TAG)) as m:
        configdump = m.history["config"]
        versiondump = m.history["versions"]
        assert versiondump == {"numpy": numpy.__version__, "caput": caput.__version__}
        assert configdump == yaml.load(testconfig, Loader=config.SafeLineLoader)


def test_metadata_to_yaml():
    """Check if metadata is written to YAML file."""

    testconfig = """
    foo: bar
    pipeline:
        save_versions:
            - numpy
            - caput
        tasks:
            - type: caput.pipeline.tasklib.debug.SaveModuleVersions
              params:
                root: {0}
            - type: caput.pipeline.tasklib.debug.SaveConfig
              params:
                root: {0}
    """.format(TAG)

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
    assert configdump == yaml.dump(yaml.load(testconfig, Loader=config.SafeLineLoader))
