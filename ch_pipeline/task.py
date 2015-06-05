import os

from caput import pipeline, config


class SingleTask(pipeline._OneAndOne, pipeline.BasicContMixin):
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

    def next(self, input=None):
        """Should not need to override."""

        # This should only be called once.
        try:
            if self.done:
                raise pipeline.PipelineStopIteration()
        except AttributeError:
            self.done = True

        # Process input and fetch ouput
        if self._no_input:
            if input is not None:
                # This should never happen.  Just here to catch bugs.
                raise RuntimeError("Somehow `input` was set.")
            output = self.process()
        else:
            if input is not None:
                input = self.cast_input(input)
            output = self.process(input)

        # Set a tag in output if needed
        if 'tag' in input.attrs and 'tag' not in output.attrs:
            output.attrs['tag'] = input.attrs['tag']

        # Write output if needed.
        if self.save and output is not None:

            # Create a tag for the output file name
            tag = output.attrs['tag'] if 'tag' in output.attrs else self._count

            # Construct the filename
            outfile = self.output_root + str(tag) + '.h5'

            # Expand any variables in the path
            outfile = os.path.expanduser(outfile)
            outfile = os.path.expandvars(outfile)
            # logger.info("%s writing data to file %s." %
            #             (self.__class__.__name__, output_filename))

            outdir = os.path.dirname(outfile)

            # Make directory if required
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            self.write_output(outfile, output)

        # Increment internal counter
        self._count = self._count + 1

        # Return the output for the next task
        return output
