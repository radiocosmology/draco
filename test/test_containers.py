import numpy as np
import pytest

from draco.core import containers

# Run these tests under MPI
pytestmark = pytest.mark.mpi


@pytest.fixture
def ss_container():
    # This is implicitly distributed, so works correctly under MPI
    ss = containers.SiderealStream(
        stack=5, input=3, ra=16, freq=np.linspace(800.0, 750.0, 5)
    )
    ss.attrs["test_attr1"] = "hello"
    ss.vis.attrs["test_attr2"] = "hello2"
    ss.weight.attrs["test_attr3"] = "hello3"
    ss.index_attrs["freq"]["alignment"] = 1

    return ss


def test_attrs_from(ss_container):
    ts = containers.TimeStream(time=10, axes_from=ss_container, attrs_from=ss_container)

    # Check that the attributes have been copied over properly
    assert len(ts.attrs) == 1
    assert ts.attrs["test_attr1"] == "hello"
    assert ts.vis.attrs["test_attr2"] == "hello2"
    assert ts.weight.attrs["test_attr3"] == "hello3"
    assert ts.index_attrs["freq"]["alignment"] == 1

    # Check that the axis attributes on the datasets did not get overwritten
    assert len(ts.vis.attrs) == 2
    assert len(ts.weight.attrs) == 2
    assert tuple(ts.vis.attrs["axis"]) == ("freq", "stack", "time")
    assert tuple(ts.weight.attrs["axis"]) == ("freq", "stack", "time")


def test_copy(ss_container):
    ss_container.vis[:] = np.arange(16)
    ss_container.weight[:] = np.arange(16)

    ss_copy = ss_container.copy(shared=("vis",))

    # Check that the copy has worked
    # Check the dataset values...
    assert (ss_copy.vis[:] == ss_container.vis[:]).all()
    assert (ss_copy.weight[:] == ss_container.weight[:]).all()
    # Check the attributes...
    assert ss_copy.attrs["test_attr1"] == "hello"
    assert ss_copy.vis.attrs["test_attr2"] == "hello2"
    assert ss_copy.weight.attrs["test_attr3"] == "hello3"
    assert ss_copy.index_attrs["freq"]["alignment"] == 1

    # Check the chunking parameters
    assert ss_copy.vis.chunks == ss_container.vis.chunks
    assert ss_copy.vis.compression == ss_container.vis.compression
    assert ss_copy.vis.compression_opts == ss_container.vis.compression_opts

    # Modify the datasets
    ss_container.vis[:] = 1.0
    ss_container.weight[:] = 2.0
    ss_container.vis.attrs["test_attr4"] = "hello4"
    ss_container.weight.attrs["test_attr5"] = "hello5"

    # Check that we see the modified values
    assert (ss_copy.vis[:] == 1.0).all()
    assert (ss_copy.weight[:] == np.arange(16)).all()
    assert ss_copy.vis.attrs["test_attr4"] == "hello4"
    assert "test_attr5" not in ss_copy.weight.attrs

    # These tests only make sense when running across multiple MPI processes
    assert ss_container.vis.distributed_axis == 0
    assert ss_copy.vis.distributed_axis == 0

    current_shape = ss_copy.weight.local_shape

    ss_container.redistribute("ra")

    assert ss_copy.vis.local_shape == ss_container.vis.local_shape
    assert ss_copy.weight.local_shape == current_shape


def test_copy_filter(ss_container):
    """Test copying datasets between container while filtering an axis.

    This test should ensure that slices and selections can be appplied to
    various axes properly, that arguments can be passed correctly, and that
    the function fails as expected when bad arguments are provided.
    """
    new = containers.SiderealStream(axes_from=ss_container, attrs_from=ss_container)

    # Test some selections which should work
    for sel in (slice(None), slice(0, 16, 1), list(range(16)), slice(0, 30)):
        # These should all pass
        containers.copy_datasets_filter(ss_container, new, "ra", {"ra": sel})

    # No selections
    containers.copy_datasets_filter(ss_container, new, [], {})

    # Force redistribution and downselection
    new = containers.SiderealStream(
        axes_from=ss_container, attrs_from=ss_container, ra=5
    )

    new.redistribute("stack")
    ss_container.redistribute("ra")
    containers.copy_datasets_filter(ss_container, new, ("ra",), {"ra": slice(0, 5)})

    new.redistribute("freq")
    ss_container.redistribute("ra")
    containers.copy_datasets_filter(ss_container, new, ["ra"], {"ra": [0, 3, 4, 6, 9]})

    # This should faily due to a mismatch in axis and selection arguments
    with pytest.raises(ValueError):
        containers.copy_datasets_filter(
            ss_container, new, ["freq"], {"ra": slice(None)}
        )

    # Some multi-axis selections
    new = containers.SiderealStream(
        axes_from=ss_container, attrs_from=ss_container, ra=2, freq=2
    )

    # This should pass since there is an axis available for redistribution
    containers.copy_datasets_filter(
        ss_container,
        new,
        selection={"ra": [0, 2], "freq": [0, 4]},
    )

    new = containers.SiderealStream(
        axes_from=ss_container, attrs_from=ss_container, ra=2, freq=2, stack=2
    )
    # This should fail since there is no axis available to redistribute
    with pytest.raises(ValueError):
        containers.copy_datasets_filter(
            ss_container,
            new,
            selection={"ra": [0, 2], "freq": [0, 4], "stack": [0, 2]},
        )
