import os

from caput import pipeline, config


class SingleTask(pipeline.TaskBase, pipeline.BasicContMixin):
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
    output_root : string
        Pipeline settable parameter giving the first part of the output path.
        If set to 'None' no output is written.

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
    output_root = config.Property(default='', proptype=str)

    _count = 0

    done = False
    _no_input = False

    def __init__(self):
        """Checks inputs and outputs and stuff."""

        import inspect

        # Inspect the `process` method to see how many arguments it takes.
        pro_argspec = inspect.getargspec(self.process)
        n_args = len(pro_argspec.args) - 1

        #if n_args  > 1:
        #    msg = ("`process` method takes more than 1 argument, which is not"
        #           " allowed.")
        #    raise PipelineConfigError(msg)

        if pro_argspec.varargs or pro_argspec.keywords or pro_argspec.defaults:
            msg = ("`process` method may not have variable length or optional"
                   " arguments.")
            raise PipelineConfigError(msg)
        
        if n_args == 0:
            self._no_input = True
        else:
            self._no_input = False


    def next(self, *input):
        """Should not need to override. Implement `process` instead."""

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
        if 'tag' not in output.attrs and len(input) > 0 and 'tag' in input[0].attrs:
            output.attrs['tag'] = input[0].attrs['tag']

        # Write the output if needed
        self._save_output(output)

        # Increment internal counter
        self._count = self._count + 1

        # Return the output for the next task
        return output

    def finish(self):
        """Should not need to override. Implement `process_finish` instead."""

        try:
            output = self.process_finish()
            
            # Write the output if needed
            self._save_output(output)

            return output

        except AttributeError:
            pass

    def _save_output(self, output):
        # Routine to write output if needed.

        if self.save and output is not None:

            # Create a tag for the output file name
            tag = output.attrs['tag'] if 'tag' in output.attrs else self._count

            # Construct the filename
            outfile = self.output_root + str(tag) + '.h5'

            # Expand any variables in the path
            outfile = os.path.expanduser(outfile)
            outfile = os.path.expandvars(outfile)

            self.write_output(outfile, output)


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
