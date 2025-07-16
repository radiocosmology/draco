import pathlib
import numpy as np
import pytest

from caput import mpiutil, pipeline
from draco.core import containers, io

# Run these tests under MPI
pytestmark = pytest.mark.mpi


@pytest.fixture
def mpi_tmp_path(tmp_path_factory):
    dirname = None
    if mpiutil.rank0:
        dirname = str(tmp_path_factory.mktemp("mpi"))
    dirname = mpiutil.bcast(dirname, root=0)

    return pathlib.Path(dirname)


@pytest.fixture
def ss_container():
    # This is implicitly distributed, so works correctly under MPI
    ss = containers.SiderealStream(
        stack=5, input=3, ra=16, freq=np.linspace(800.0, 750.0, 5)
    )
    ss.vis[:].real = np.arange(
        ss.vis.local_offset[0], ss.vis.local_offset[0] + ss.vis.local_shape[0]
    )[:, np.newaxis, np.newaxis]
    ss.vis[:].imag = np.arange(16)[np.newaxis, np.newaxis, :]
    ss.weight[:] = np.arange(5)[np.newaxis, :, np.newaxis]

    # Set some attributes
    ss.attrs["test_attr1"] = "hello"
    ss.vis.attrs["test_attr2"] = "hello2"
    ss.weight.attrs["test_attr3"] = "hello3"

    return ss


def test_LoadBasicCont_simple(ss_container, mpi_tmp_path):
    fname = str(mpi_tmp_path / "ss.h5")
    ss_container.save(fname)

    task = io.LoadBasicCont()
    task.files = [fname]

    task.setup()
    ss_load = task.next()

    # Check the datasets
    assert (ss_load.vis[:] == ss_container.vis[:]).all()
    assert (ss_load.weight[:] == ss_container.weight[:]).all()

    # Check the attributes...
    assert ss_load.attrs["test_attr1"] == "hello"
    assert ss_load.vis.attrs["test_attr2"] == "hello2"
    assert ss_load.weight.attrs["test_attr3"] == "hello3"

    # As we only put one item into the queue, this should end the iterations
    with pytest.raises(pipeline.PipelineStopIteration):
        task.next()


def test_LoadBasicCont_selection(ss_container, mpi_tmp_path):
    fname = str(mpi_tmp_path / "ss.h5")
    ss_container.save(fname)

    freq_range = [1, 4]
    ra_range = [3, 10]

    task = io.LoadBasicCont()
    task.files = [fname]
    task.selections = {
        "freq_range": freq_range,
        "ra_range": ra_range,
    }

    task.setup()
    ss_load = task.next()
    ss_vis = ss_load.vis[:]
    ss_weight = ss_load.weight[:]

    # Check the datasets (only a maximum of three ranks will have anything)
    nf = freq_range[1] - freq_range[0]
    if ss_load.comm.rank < nf:
        # Check the freq selection
        n, s, e = mpiutil.split_local(nf, comm=ss_load.comm)
        vis_real = np.arange(*freq_range)[s:e][:, np.newaxis, np.newaxis]
        assert (ss_vis.local_array.real == vis_real).all()

        # Check the ra selection
        vis_imag = np.arange(*ra_range)[np.newaxis, np.newaxis, :]
        assert (ss_vis.local_array.imag == vis_imag).all()

        # Check that nothing funky happened on the stack axis
        weight = np.arange(ss_container.vis.shape[1])[np.newaxis, :, np.newaxis]
        assert (ss_weight.local_array == weight).all()

    # Check the attributes...
    assert ss_load.attrs["test_attr1"] == "hello"
    assert ss_load.vis.attrs["test_attr2"] == "hello2"
    assert ss_load.weight.attrs["test_attr3"] == "hello3"

    # As we only put one item into the queue, this should end the iterations
    with pytest.raises(pipeline.PipelineStopIteration):
        task.next()


GLOB_WITH_NO_FILES = "/thjis/directory/better/doesnt/exist/*.h54321"
NUM_FILES = 3


@pytest.fixture
def existing_file_list():
    import os
    import tempfile

    file_list = []
    for _ in range(NUM_FILES):
        f = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        f.write("brr")
        f.flush()
        file_list.append(f.name)
        io._list_or_glob(file_list)
    file_list.append(GLOB_WITH_NO_FILES)
    yield io._list_or_glob(file_list)
    for f in file_list:
        if not "*" in f:
            os.unlink(f)


def test_list_or_glob(existing_file_list):
    assert io._list_or_glob(GLOB_WITH_NO_FILES) == []

    existing_file_list = io._list_or_glob(existing_file_list)

    assert len(existing_file_list) == NUM_FILES
    for f in existing_file_list:
        assert isinstance(f, str)

    with pytest.raises(io.ConfigError):
        io._list_or_glob(1)

    with pytest.raises(io.ConfigError):
        io._list_or_glob("/does/not/exist/for/sure")


def test_list_of_filelists(existing_file_list):
    with pytest.raises(io.ConfigError):
        io._list_of_filelists(GLOB_WITH_NO_FILES)

    existing_file_list = io._list_of_filelists(existing_file_list)
    assert len(existing_file_list) == NUM_FILES
    for f in existing_file_list:
        assert isinstance(f, str)

    with pytest.raises(io.ConfigError):
        io._list_of_filelists([1])

    with pytest.raises(io.ConfigError):
        io._list_of_filelists(["/does/not/exist/for/sure"])

    list_of_lists = [existing_file_list for _ in range(NUM_FILES)]
    list_of_lists = io._list_of_filelists(list_of_lists)
    assert len(list_of_lists) == NUM_FILES * NUM_FILES


def test_list_of_filegroups(existing_file_list):
    with pytest.raises(io.ConfigError):
        io._list_of_filegroups(GLOB_WITH_NO_FILES)

    groups = [
        {"files": existing_file_list, "tag": "test"},
        {"files": existing_file_list},
    ]
    groups = io._list_of_filegroups(groups)
    assert len(groups) == 2
    for i in range(2):
        assert len(groups[i]["files"]) == NUM_FILES
        for f in groups[i]["files"]:
            assert isinstance(f, str)

    with pytest.raises(io.ConfigError):
        io._list_of_filegroups([1])

    with pytest.raises(io.ConfigError):
        io._list_of_filegroups([{"files": ["/does/not/exist/for/sure"]}])
