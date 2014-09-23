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


def _files_from_spec(inst_name, timerange, archive_root=None):
    # Get a list of files in a dataset from an instrument name and timerange.

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
