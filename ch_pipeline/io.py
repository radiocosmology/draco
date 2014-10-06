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

    FilesFromDatasetSpec
    SaveOutput
"""
import os

from caput import pipeline
from caput import config
from caput import mpiutil

import containers


def _files_from_spec(inst_name, timerange, archive_root=None):
    # Get a list of files in a dataset from an instrument name and timerange.

    files = None

    if mpiutil.rank0:

        from ch_util import data_index as di

        # Get instrument
        inst_obj = di.ArchiveInst.select().where(di.ArchiveInst.name == inst_name).get()

        # Ensure timerange is a list
        if not isinstance(timerange, list):
            timerange = [timerange]

        # Find the earliest and latest times
        earliest = min([ tr['start'] for tr in timerange ])
        latest   = max([ tr['end']   for tr in timerange ])

        # Create a finder object limited to the relevant time
        fi = di.Finder()

        # Set the archive_root
        if archive_root is not None:
            fi.archive_root = archive_root

        # Set the time range that encapsulates all the intervals
        fi.set_time_range(earliest, latest)

        # Add in all the time ranges
        for ti in timerange:
            fi.include_time_interval(ti['start'], ti['end'])

        # Only include the required instrument
        fi.filter_acqs(di.ArchiveAcq.inst == inst_obj)

        # Pull out the results and extract all the files
        results = fi.get_results()
        files = [ fname for result in results for fname in result[0] ]
        files.sort()

    files = mpiutil.world.bcast(files, root=0)

    return files


class FilesFromDatasetSpec(pipeline.TaskBase):
    """Create a list of files to process in a dataset.

    Attributes
    ----------
    dataset_file : str
        YAML file containing dataset specification. If not specified, use the
        one contained within the ch_pipeline repository.
    dataset_name : str
        Name of dataset to use.
    archive_root : str
        Root of archive to add to file paths.
    """

    dataset_file = config.Property(proptype=str, default='')
    dataset_name = config.Property(proptype=str, default='')
    archive_root = config.Property(proptype=str, default='')

    def setup(self):

        import yaml

        # Set to default datasets file
        if self.dataset_file == '':
            self.dataset_file = os.path.dirname(__file__) + '/data/datasets.yaml'

        # Check existense and read yaml datasets file
        if not os.path.exists(self.dataset_file):
            raise Exception("Dataset file not found.")

        with open(self.dataset_file, 'r') as f:
            dconf = yaml.safe_load(f)

        dsets = dconf['datasets']

        # Find the correct dataset
        dset = None
        for ds in dsets:
            if ds['name'] == self.dataset_name:
                dset = ds
                break

        # Raise exception if it's not found
        if dset is None:
            raise Exception("Dataset %s not found in %s." % (self.dataset_name, self.dataset_file))

        # Create a list of files
        files = _files_from_spec(dset['instrument'], dset['timerange'], self.archive_root)

        return files


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


class SaveOutput(pipeline.TaskBase):
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


class PrintInput(pipeline.TaskBase):
    """Stupid module which just prints whatever it gets. Good for debugging.
    """

    def next(self, input):

        print input

        return input
