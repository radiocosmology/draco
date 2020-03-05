import numpy as np
import pytest

from draco.core import containers


@pytest.fixture
def ss_container():

    ss = containers.SiderealStream(
        stack=5, input=3, ra=16, freq=np.linspace(800.0, 750.0, 5)
    )
    ss.attrs["test_attr1"] = "hello"
    ss.vis.attrs["test_attr2"] = "hello2"

    return ss


def test_attrs_from(ss_container):

    ts = containers.TimeStream(time=10, axes_from=ss_container, attrs_from=ss_container)

    # Check that the attributes have been copied over properly
    assert len(ts.attrs) == 1
    assert ts.attrs["test_attr1"] == "hello"
    assert ts.vis.attrs["test_attr2"] == "hello2"

    # Check that the axis attributes on the dataasets did not get overwritten
    assert len(ts.vis.attrs) == 2
    assert len(ts.weight.attrs) == 1
    assert tuple(ts.vis.attrs["axis"]) == ("freq", "stack", "time")
    assert tuple(ts.weight.attrs["axis"]) == ("freq", "stack", "time")
