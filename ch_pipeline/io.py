"""
=====================================
Tasks for IO (:mod:`~ch_pipeline.io`)
=====================================

.. currentmodule:: ch_pipeline.io

Tasks for calculating IO. Notably a task which will write out the parallel
MPIDataset classes.

Tasks
=====

.. autosummary::
    :toctree: generated/

    LoadFiles
    LoadFilesFromParams
    Save
    Print
    LoadBeamTransfer
"""

import os.path

import numpy as np

from caput import pipeline
from caput import config

from ch_util import andata

from . import task

def _list_of_filelists(files):
    # Take in a list of lists/glob patterns of filenames
    import glob

    f2 = []

    for filelist in files:

        if isinstance(filelist, str):
            filelist = glob.glob(filelist)
        elif isinstance(filelist, list):
            pass
        else:
            raise Exception('Must be list or glob pattern.')
        f2.append(filelist)

    return f2


def _list_or_glob(files):
    # Take in a list of lists/glob patterns of filenames
    import glob

    if isinstance(files, str):
        files = glob.glob(files)
    elif isinstance(files, list):
        pass
    else:
        raise RuntimeError('Must be list or glob pattern.')

    return files


class LoadFilesFromParams(pipeline.TaskBase):
    """Load data from files given in the tasks parameters.

    Attributes
    ----------
    files : glob pattern, or list
        Can either be a glob pattern, or lists of actual files.
    """

    files = config.Property(proptype=_list_or_glob)

    def next(self):
        """Load the given files in turn and pass on.

        Returns
        -------
        cont : subclass of `memh5.BasicCont`
        """

        from caput import memh5


        if len(self.files) == 0:
            raise pipeline.PipelineStopIteration

        # Fetch and remove the first item in the list
        file_ = self.files.pop(0)

        cont = memh5.BasicCont.from_file(file_, distributed=True)

        if 'tag' not in cont.attrs:
            # Get the first part of the actual filename and use it as the tag
            tag = os.path.splitext(os.path.basename(file_))[0]

            cont.attrs['tag'] = tag

        return cont


# Define alias for old code
LoadBasicCont = LoadFilesFromParams


class LoadFiles(LoadFilesFromParams):
    """Load data from files passed into the setup routine.

    File must be a serialised subclass of :class:`memh5.BasicCont`.
    """

    files = None

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
        """
        if not isinstance(files, (list, tuple)):
            raise RuntimeError('Argument must be list of files.')

        self.files = files


class LoadCorrDataFiles(task.SingleTask):
    """Load data from files passed into the setup routine.

    File must be a serialised subclass of :class:`memh5.BasicCont`.
    """

    files = None

    _file_ptr = 0

    freq_start = config.Property(proptype=int, default=None)
    freq_end = config.Property(proptype=int, default=None)

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
        """
        if not isinstance(files, (list, tuple)):
            raise RuntimeError('Argument must be list of files.')

        self.files = files

        # Set up frequency selection
        if self.freq_start is not None:
            self.freq_sel = np.arange(self.freq_start, self.freq_end)
        else:
            self.freq_sel = None


    def process(self):
        """Load in each sidereal day.

        Returns
        -------
        ts : andata.CorrData
            The timestream of each sidereal day.
        """

        from caput import mpiutil

        if len(self.files) == self._file_ptr: # or self._file_ptr > 1:
            raise pipeline.PipelineStopIteration

        # Fetch and remove the first item in the list
        file_ = self.files[self._file_ptr]
        self._file_ptr += 1

        if mpiutil.rank0:
            print "Reading file %i of %i." % (self._file_ptr, len(self.files))

        ts = andata.CorrData.from_acq_h5(file_, distributed=True, freq_sel=self.freq_sel)

        if 'tag' not in ts.attrs:
            # Get the first part of the actual filename and use it as the tag
            tag = os.path.splitext(os.path.basename(file_))[0]

            ts.attrs['tag'] = tag

        # Add a weight dataset if needed
        if 'vis_weight' not in ts.datasets:
            weight_dset = ts.create_dataset('vis_weight', shape=ts.vis.shape, dtype=np.uint8,
                                            distributed=True, distributed_axis=0)
            weight_dset.attrs['axis'] = ts.vis.attrs['axis']

            # Set weight to a reasonable value (128), unless the vis value is
            # zero which presumably came from missing data. NOTE: this may have
            # a small bias
            weight_dset[:] = np.where(ts.vis[:] == 0.0, 0, 128)

        return ts


class Save(pipeline.TaskBase):
    """Save out the input, and pass it on.

    Assumes that the input has a `to_hdf5` method. Appends a *tag* if there is
    a `tag` entry in the attributes, otherwise just uses a count.

    Attributes
    ----------
    root : str
        Root of the file name to output to.
    """

    root = config.Property(proptype=str)

    count = 0

    def next(self, data):
        """Write out the data file.

        Assumes it has an MPIDataset interface.

        Parameters
        ----------
        data : mpidataset.MPIDataset
            Data to write out.
        """

        if 'tag' not in data.attrs:
            tag = self.count
            self.count += 1
        else:
            tag = data.attrs['tag']

        fname = '%s_%s.h5' % (self.root, str(tag))

        data.to_hdf5(fname)

        return data


class Print(pipeline.TaskBase):
    """Stupid module which just prints whatever it gets. Good for debugging.
    """

    def next(self, input_):

        print input_

        return input_


class LoadBeamTransfer(pipeline.TaskBase):
    """Loads a beam transfer manager from disk.

    Attributes
    ----------
    product_directory : str
        Path to the saved Beam Transfer products.
    """

    product_directory = config.Property(proptype=str)

    def setup(self):
        """Load the beam transfer matrices.

        Returns
        -------
        tel : TransitTelescope
            Object describing the telescope.
        bt : BeamTransfer
            BeamTransfer manager.
        feed_info : list, optional
            Optional list providing additional information about each feed.
        """

        import os

        from drift.core import beamtransfer

        if not os.path.exists(self.product_directory):
            raise RuntimeError('BeamTransfers do not exist.')

        bt = beamtransfer.BeamTransfer(self.product_directory)

        tel = bt.telescope

        try:
            feed_info = tel.feed_info
            return tel, bt, feed_info
        except AttributeError:
            return tel, bt
