from draco.core.containers import GainData

from caput import mpiarray, mpiutil

import pytest
import glob
import numpy as np
import os

# Run these tests under MPI
pytestmark = pytest.mark.mpi

comm = mpiutil.world
rank, size = mpiutil.rank, mpiutil.size

len_axis = 8

dset1 = np.arange(len_axis * len_axis * len_axis)
dset1 = dset1.reshape((len_axis, len_axis, len_axis))

dset2 = np.arange(len_axis * len_axis * len_axis)
dset2 = dset2.reshape((len_axis, len_axis, len_axis))

freqs = np.arange(len_axis)
inputs = np.arange(len_axis)
times = np.arange(len_axis)

fsel = slice(5)
isel = slice(1, 4)
tsel = slice(1, 4)


@pytest.fixture
def container_on_disk():
    fname = "tmp_test_memh5_select.h5"
    container = GainData(freq=freqs, input=inputs, time=times)
    container.create_dataset("gain", data=dset1.view())
    container.create_dataset("weight", data=dset2.view())
    container.save(fname)
    yield fname

    # Ensure that all ranks have run their tests before deleting
    if size > 1:
        comm.Barrier()

    # tear down
    file_names = glob.glob(fname + "*")
    if rank == 0:
        for fname in file_names:
            os.remove(fname)


local_from = int(len_axis / size * rank)
local_to = int(len_axis / size * (rank + 1))
global_data1 = np.arange(len_axis * len_axis * len_axis, dtype=np.float32)
local_data1 = global_data1.reshape(len_axis, -1, len_axis)[local_from:local_to]
d_array1 = mpiarray.MPIArray.wrap(local_data1, axis=0)
global_data2 = np.arange(len_axis * len_axis * len_axis, dtype=np.float32)
local_data2 = global_data2.reshape(len_axis, -1, len_axis)[local_from:local_to]
d_array2 = mpiarray.MPIArray.wrap(local_data2, axis=0)


@pytest.fixture
def container_on_disk_distributed():
    fname = "tmp_test_memh5_select_distributed.h5"
    container = GainData(freq=freqs, input=inputs, time=times)
    container.create_dataset("gain", data=d_array1)
    container.create_dataset("weight", data=d_array2)
    container.save(fname)

    # load file and apply selection
    md = GainData.from_file(
        fname, freq_sel=fsel, input_sel=isel, time_sel=tsel, distributed=True
    )
    # save it again
    md.save(fname)
    yield fname

    # Ensure that all ranks have run their tests before deleting
    if size > 1:
        comm.Barrier()

    # tear down
    file_names = glob.glob(fname + "*")
    if rank == 0:
        for fname in file_names:
            os.remove(fname)


def test_H5FileSelect(container_on_disk):
    """Tests that makes hdf5 objects and tests selecting on their axes."""

    m = GainData.from_file(
        container_on_disk, freq_sel=fsel, input_sel=isel, time_sel=tsel
    )
    assert np.all(m["gain"][:] == dset1[(fsel, isel, tsel)])
    assert np.all(m["weight"][:] == dset2[(fsel, isel, tsel)])
    assert np.all(m.index_map["freq"] == freqs[fsel])
    assert np.all(m.index_map["input"] == inputs[isel])
    assert np.all(m.index_map["time"] == times[tsel])


def test_H5FileSelect_distributed(container_on_disk):
    """Load H5 into parallel container while down-selecting axes."""

    m = GainData.from_file(
        container_on_disk,
        freq_sel=fsel,
        input_sel=isel,
        time_sel=tsel,
        distributed=True,
    )
    assert np.all(m["gain"][:] == dset1[(fsel, isel, tsel)])
    assert np.all(m["weight"][:] == dset2[(fsel, isel, tsel)])
    assert np.all(m.index_map["freq"] == freqs[fsel])
    assert np.all(m.index_map["input"] == inputs[isel])
    assert np.all(m.index_map["time"] == times[tsel])


def test_H5FileSelect_distributed_on_disk(container_on_disk_distributed):
    """Load distributed H5 into parallel container while down-selecting axes."""

    if rank == 0:
        md = GainData.from_file(container_on_disk_distributed, distributed=False)

        assert np.all(md["gain"][:] == dset1[(fsel, isel, tsel)])
        assert np.all(md["weight"][:] == dset2[(fsel, isel, tsel)])
        assert np.all(md.index_map["freq"] == freqs[fsel])
        assert np.all(md.index_map["input"] == inputs[isel])
        assert np.all(md.index_map["time"] == times[tsel])


def test_test_H5FileSelect_distributed_on_disk_simple():
    """
    Load distributed H5 into parallel container while down-selecting axes.

    This test does the same as `test_H5FileSelect_distributed_on_disk` but it checks the
    frequencies distributed to each node after selection instead of writing to disk
    before checking.
    """
    if size != 4:
        pytest.skip("This test has to be run with mpirun -np 4")
    len_axis = 8

    local_from = int(len_axis / size * rank)
    local_to = int(len_axis / size * (rank + 1))
    global_data1 = np.arange(len_axis * len_axis * len_axis, dtype=np.int32)
    local_data1 = global_data1.reshape(len_axis, -1, len_axis)[local_from:local_to]
    d_array1 = mpiarray.MPIArray.wrap(local_data1, axis=0)
    global_data2 = np.arange(len_axis * len_axis * len_axis, dtype=np.int32)
    local_data2 = global_data2.reshape(len_axis, -1, len_axis)[local_from:local_to]
    d_array2 = mpiarray.MPIArray.wrap(local_data2, axis=0)

    fname = "tmp_test_memh5_select_distributed_simple.h5"
    container = GainData(freq=freqs, input=inputs, time=times)
    container.create_dataset("gain", data=d_array1)
    container.create_dataset("weight", data=d_array2)
    container.save(fname)

    # load file and apply selection
    fsel = slice(5)
    md = GainData.from_file(fname, freq_sel=fsel, distributed=True)

    # test
    if rank == 0:
        # should hold freq indices 0 and 1
        assert np.all(md["gain"][:] == dset1[(slice(2), slice(None), slice(None))])
        assert np.all(md["weight"][:] == dset2[(slice(2), slice(None), slice(None))])
        assert np.all(md.index_map["freq"] == freqs[fsel])
    else:
        # should hold 1 freq index each
        assert np.all(
            md["weight"][:]
            == dset2[(slice(rank + 1, rank + 2), slice(None), slice(None))]
        )
        assert np.all(
            md["gain"][:]
            == dset1[(slice(rank + 1, rank + 2), slice(None), slice(None))]
        )
        assert np.all(md.index_map["freq"] == freqs[fsel])

    # tear down
    file_names = glob.glob(fname + "*")
    if rank == 0:
        for fname in file_names:
            os.remove(fname)
