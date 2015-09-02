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

    Save
    Print
    LoadBasicCont
    LoadBeamTransfer
"""

from caput import pipeline
from caput import config
from caput import mpiutil

import containers


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


class LoadFiles(pipeline.TaskBase):
    """Load data ifrom specified files.

    Attributes
    ----------
    files : glob pattern
        List of sets of files to take in. Can either be lists of actual files,
        or glob patterns. Example: [ 'dir1/*.h5', ['dir2/a.h5', 'dir2/b.h5']].

    Examples
    --------
    This can be configured from a pipeline file like this:

    .. code-block:: yaml

        pipeline :
            tasks:
            -   type:   ch_pipeline.io.LoadFiles
                out:    ts
                params:
                    files:
                        -   "dir1/*.h5"
                        -
                            -   "dir2/a.h5"
                            -   "dir2/b.h5"

    Each set is fed through the pipeline individually. That is, using the
    above example, the first call to `next`, creates a timestream from all the
    `*.h5` files in `dir1`, and then passes it on. The second, and final
    `next` call returns a timestream from `dir1/a.h5` and `dir2/b.h5`.
    """

    files = config.Property(proptype=_list_of_filelists)

    def next(self):
        """Load in each set of files.

        Returns
        -------
        ts : containers.TimeStream
            The timestream of each set of files.
        """

        if len(self.files) == 0:
            raise pipeline.PipelineStopIteration

        files = self.files.pop(0)

        if mpiutil.rank0:
            print "Starting read of [%i files]" % len(files)

        ts = containers.TimeStream.from_acq_files(sorted(files))  # Ensure file list if sorted
        ts.attrs['tag'] = 'meh'

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

    def next(self, input):

        print input

        return input


class LoadBasicCont(pipeline.TaskBase):
    """Load a series of files from disk and pass them through the pipeline.

    Uses the ability of :class:`memh5.BasicCont` to return the data in the class
    it was saved in.

    Attributes
    ----------
    files : list
        A list containing the paths of files to load.
    """

    files = config.Property(proptype=list)

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
        file = self.files.pop(0)

        cont = memh5.BasicCont.from_file(file, distributed=True)

        return cont


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
