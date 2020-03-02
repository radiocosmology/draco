"""An improved base task implementing easy (and explicit) saving of outputs.

Tasks
=====

.. autosummary::
    :toctree:

    SingleTask
    ReturnLastInputOnFinish
    ReturnFirstInputOnFinish
    Delete
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
from past.builtins import basestring

# === End Python 2/3 compatibility

import os
import logging

import numpy as np

from caput import pipeline, config, memh5


class MPILogFilter(logging.Filter):
    """Filter log entries by MPI rank.

    Also this will optionally add MPI rank information, and add an elapsed time
    entry.

    Parameters
    ----------
    add_mpi_info : boolean, optional
        Add MPI rank/size info to log records that don't already have it.
    level_rank0 : int
        Log level for messages from rank=0.
    level_all : int
        Log level for messages from all other ranks.
    """

    def __init__(
        self, add_mpi_info=True, level_rank0=logging.INFO, level_all=logging.WARN
    ):

        from mpi4py import MPI

        self.add_mpi_info = add_mpi_info

        self.level_rank0 = level_rank0
        self.level_all = level_all

        self.comm = MPI.COMM_WORLD

    def filter(self, record):

        # Add MPI info if desired
        try:
            record.mpi_rank
        except AttributeError:
            if self.add_mpi_info:
                record.mpi_rank = self.comm.rank
                record.mpi_size = self.comm.size

        # Add a new field with the elapsed time in seconds (as a float)
        record.elapsedTime = record.relativeCreated * 1e-3

        # Return whether we should filter the record or not.
        return (record.mpi_rank == 0 and record.levelno >= self.level_rank0) or (
            record.mpi_rank > 0 and record.levelno >= self.level_all
        )


def _log_level(x):
    """Interpret the input as a logging level.

    Parameters
    ----------
    x : int or str
        Explicit integer logging level or one of 'DEBUG', 'INFO', 'WARN',
        'ERROR' or 'CRITICAL'.

    Returns
    -------
    level : int
    """

    level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "WARNING": logging.WARN,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if isinstance(x, int):
        return x
    elif isinstance(x, basestring) and x in level_dict:
        return level_dict[x.upper()]
    else:
        raise ValueError("Logging level %s not understood" % repr(x))


class SetMPILogging(pipeline.TaskBase):
    """A task used to configure MPI aware logging.

    Attributes
    ----------
    level_rank0, level_all : int or str
        Log level for rank=0, and other ranks respectively.
    """

    level_rank0 = config.Property(proptype=_log_level, default=logging.INFO)
    level_all = config.Property(proptype=_log_level, default=logging.WARN)

    def __init__(self):

        from mpi4py import MPI
        import math

        logging.captureWarnings(True)

        rank_length = int(math.log10(MPI.COMM_WORLD.size)) + 1

        mpi_fmt = "[MPI %%(mpi_rank)%id/%%(mpi_size)%id]" % (rank_length, rank_length)
        filt = MPILogFilter(level_all=self.level_all, level_rank0=self.level_rank0)

        # This uses the fact that caput.pipeline.Manager has already
        # attempted to set up the logging. We just override the level, and
        # insert our custom filter
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        ch = root_logger.handlers[0]
        ch.setLevel(logging.DEBUG)
        ch.addFilter(filt)

        formatter = logging.Formatter(
            "%(elapsedTime)8.1fs "
            + mpi_fmt
            + " - %(levelname)-8s %(name)s: %(message)s"
        )

        ch.setFormatter(formatter)


class LoggedTask(pipeline.TaskBase):
    """A task with logger support.
    """

    log_level = config.Property(proptype=_log_level, default=None)

    def __init__(self):

        # Get the logger for this task
        self._log = logging.getLogger("%s.%s" % (__name__, self.__class__.__name__))

        # Set the log level for this task if specified
        if self.log_level is not None:
            self.log.setLevel(self.log_level)

    @property
    def log(self):
        """The logger object for this task.
        """
        return self._log


class MPITask(pipeline.TaskBase):
    """Base class for MPI using tasks. Just ensures that the task gets a `comm`
    attribute.
    """

    comm = None

    def __init__(self):

        from mpi4py import MPI

        # Set default communicator
        self.comm = MPI.COMM_WORLD


class _AddRankLogAdapter(logging.LoggerAdapter):
    """Add the rank of the logging process to a log message.

    Attributes
    ----------
    calling_obj : object
        An object with a `comm` property that will be queried for the rank.
    """

    calling_obj = None

    def process(self, msg, kwargs):

        if "extra" not in kwargs:
            kwargs["extra"] = {}

        kwargs["extra"]["mpi_rank"] = self.calling_obj.comm.rank
        kwargs["extra"]["mpi_size"] = self.calling_obj.comm.size

        return msg, kwargs


class MPILoggedTask(MPITask, LoggedTask):
    """A task base that has MPI aware logging.
    """

    def __init__(self):

        # Initialise the base classes
        MPITask.__init__(self)
        LoggedTask.__init__(self)

        # Replace the logger with a LogAdapter instance that adds MPI process
        # information
        logadapter = _AddRankLogAdapter(self._log, None)
        logadapter.calling_obj = self
        self._log = logadapter


class SingleTask(MPILoggedTask, pipeline.BasicContMixin):
    """Process a task with at most one input and output.

    Both input and output are expected to be :class:`memh5.BasicCont` objects.
    This class allows writing of the output when requested.

    Tasks inheriting from this class should override `process` and optionally
    :meth:`setup` or :meth:`finish`. They should not override :meth:`next`.

    If the value of :attr:`input_root` is anything other than the string "None"
    then the input will be read (using :meth:`read_input`) from the file
    ``self.input_root + self.input_filename``.  If the input is specified both as
    a filename and as a product key in the pipeline configuration, an error
    will be raised upon initialization.


    If the value of :attr:`output_root` is anything other than the string
    "None" then the output will be written (using :meth:`write_output`) to the
    file ``self.output_root + self.output_filename``.

    Attributes
    ----------
    save : bool
        Whether to save the output to disk or not.
    output_name : string
        A python format string used to construct the filename. Valid identifiers are:
          - `count`: an integer giving which iteration of the task is this.
          - `tag`: a string identifier for the output derived from the
                   containers `tag` attribute. If that attribute is not present
                   `count` is used instead.
          - `key`: the name of the output key.
          - `task`: the (unqualified) name of the task.
          - `output_root`: the value of the output root argument. This is deprecated
                           and is just used for legacy support. The default value of
                           `output_name` means the previous behaviour works.
    output_root : string
        Pipeline settable parameter giving the first part of the output path.
        Deprecated in favour of `output_name`.
    nan_check : bool
        Check the output for NaNs (and infs) logging if they are present.
    nan_dump : bool
        If NaN's are found, dump the container to disk.
    nan_skip : bool
        If NaN's are found, don't pass on the output.
    versions : dict
        Keys are module names (str) and values are their version strings. This is
        attached to output metadata.
    pipeline_config : dict
        Global pipeline configuration. This is attached to output metadata.

    Methods
    -------
    next
    setup
    process
    finish
    read_input
    cast_input
    write_output

    """

    save = config.Property(default=False, proptype=bool)

    output_root = config.Property(default="", proptype=str)
    output_name = config.Property(default="{output_root}{tag}.h5", proptype=str)

    nan_check = config.Property(default=True, proptype=bool)
    nan_skip = config.Property(default=True, proptype=bool)
    nan_dump = config.Property(default=True, proptype=bool)

    # Metadata to get attached to the output
    versions = config.Property(default={}, proptype=dict)
    pipeline_config = config.Property(default={}, proptype=dict)

    _count = 0

    done = False
    _no_input = False

    def __init__(self):
        """Checks inputs and outputs and stuff."""

        super(SingleTask, self).__init__()

        import inspect

        # Inspect the `process` method to see how many arguments it takes.
        pro_argspec = inspect.getargspec(self.process)
        n_args = len(pro_argspec.args) - 1

        if pro_argspec.varargs or pro_argspec.keywords or pro_argspec.defaults:
            msg = (
                "`process` method may not have variable length or optional"
                " arguments."
            )
            raise pipeline.PipelineConfigError(msg)

        if n_args == 0:
            self._no_input = True
        else:
            self._no_input = False

    def next(self, *input):
        """Should not need to override. Implement `process` instead."""

        self.log.info("Starting next for task %s" % self.__class__.__name__)

        self.comm.Barrier()

        # This should only be called once.
        try:
            if self.done:
                raise pipeline.PipelineStopIteration()
        except AttributeError:
            self.done = True

        # Process input and fetch ouput
        if self._no_input:
            if len(input) > 0:
                # This should never happen.  Just here to catch bugs.
                raise RuntimeError("Somehow `input` was set.")
            output = self.process()
        else:
            output = self.process(*input)

        # Return immediately if output is None to skip writing phase.
        if output is None:
            return

        # Set a tag in output if needed
        if "tag" not in output.attrs and len(input) > 0 and "tag" in input[0].attrs:
            output.attrs["tag"] = input[0].attrs["tag"]

        # Check for NaN's etc
        output = self._nan_process_output(output)

        # Write the output if needed
        self._save_output(output)

        # Increment internal counter
        self._count = self._count + 1

        self.log.info("Leaving next for task %s" % self.__class__.__name__)

        # Return the output for the next task
        return output

    def finish(self):
        """Should not need to override. Implement `process_finish` instead."""

        self.log.info("Starting finish for task %s" % self.__class__.__name__)

        try:
            output = self.process_finish()

            # Check for NaN's etc
            output = self._nan_process_output(output)

            # Write the output if needed
            self._save_output(output)

            self.log.info("Leaving finish for task %s" % self.__class__.__name__)

            return output

        except AttributeError:
            self.log.info("No finish for task %s" % self.__class__.__name__)
            pass

    def _save_output(self, output):
        # Routine to write output if needed.
        if self.save and output is not None:

            # add metadata to output
            metadata = {"versions": self.versions, "config": self.pipeline_config}
            for key, value in metadata.items():
                # NOTE: this will overwrite any existing metadata keys. This is probably the right
                # thing to do for now, but really we should be using the `history` functionality for
                # this.
                output.attrs[key] = value

            # Create a tag for the output file name
            tag = output.attrs["tag"] if "tag" in output.attrs else self._count

            # Construct the filename
            name_parts = {
                "tag": tag,
                "count": self._count,
                "task": self.__class__.__name__,
                "key": self._out_keys[0] if self._out_keys else "",
                "output_root": self.output_root,
            }
            outfile = self.output_name.format(**name_parts)

            # Expand any variables in the path
            outfile = os.path.expanduser(outfile)
            outfile = os.path.expandvars(outfile)

            self.log.debug("Writing output %s to disk.", outfile)
            self.write_output(outfile, output)

    def _nan_process_output(self, output):
        # Process the output to check for NaN's
        # Returns the output or, None if it should be skipped

        if self.nan_check:
            nan_found = self._nan_check_walk(output)

            if nan_found and self.nan_dump:

                # Construct the filename
                tag = output.attrs["tag"] if "tag" in output.attrs else self._count
                outfile = "nandump_" + self.__class__.__name__ + "_" + str(tag) + ".h5"
                self.log.debug("NaN found. Dumping %s", outfile)
                self.write_output(outfile, output)

            if nan_found and self.nan_skip:
                self.log.debug("NaN found. Skipping output.")
                return None

        return output

    def _nan_check_walk(self, cont):
        # Walk through a memh5 container and check for NaN's and Inf's.
        # Logs any issues found and returns True if there were any found.
        from mpi4py import MPI

        if isinstance(cont, memh5.MemDiskGroup):
            cont = cont._data

        stack = [cont]
        found = False

        # Walk over the container tree...
        while stack:
            n = stack.pop()

            # Check the dataset for non-finite numbers
            if isinstance(n, memh5.MemDataset):

                # Try to test for NaN's and infs. This will fail for compound datatypes...
                arr = n[:]
                try:
                    is_nan = np.isnan(arr)
                    is_inf = np.isinf(arr)
                    arr = n[:]
                except TypeError:
                    continue

                if is_nan.any():
                    self.log.info(
                        "NaN's found in dataset %s [%i of %i elements]",
                        n.name,
                        is_nan.sum(),
                        arr.size,
                    )
                    found = True

                if is_inf.any():
                    self.log.info(
                        "Inf's found in dataset %s [%i of %i elements]",
                        n.name,
                        is_inf.sum(),
                        arr.size,
                    )
                    found = True

            elif isinstance(n, (memh5.MemGroup, memh5.MemDiskGroup)):
                for item in n.values():
                    stack.append(item)

        # All ranks need to know if any rank found a NaN/Inf
        found = self.comm.allreduce(found, op=MPI.MAX)

        return found


class ReturnLastInputOnFinish(SingleTask):
    """Workaround for `caput.pipeline` issues.

    This caches its input on every call to `process` and then returns
    the last one for a finish call.
    """

    x = None

    def process(self, x):
        """Take a reference to the input.

        Parameters
        ----------
        x : object
        """
        self.x = x

    def process_finish(self):
        """Return the last input to process.

        Returns
        -------
        x : object
            Last input to process.
        """
        return self.x


class ReturnFirstInputOnFinish(SingleTask):
    """Workaround for `caput.pipeline` issues.

    This caches its input on the first call to `process` and
    then returns it for a finish call.
    """

    x = None

    def process(self, x):
        """Take a reference to the input.

        Parameters
        ----------
        x : object
        """
        if self.x is None:
            self.x = x

    def process_finish(self):
        """Return the last input to process.

        Returns
        -------
        x : object
            Last input to process.
        """
        return self.x


class Delete(SingleTask):
    """Delete pipeline products to free memory."""

    def process(self, x):
        """Delete the input and collect garbage.

        Parameters
        ----------
        x : object
            The object to be deleted.
        """
        import gc

        self.log.info("Deleting %s" % type(x))
        del x
        gc.collect()

        return None
