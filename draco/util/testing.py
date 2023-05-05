"""draco test utils."""

from caput import config, memh5, pipeline

from draco.core.task import SingleTask


class DummyTask(SingleTask):
    """Produce an empty data stream for testing.

    Attributes
    ----------
    total_len : int
        Length of output data stream. Default: 1.
    tag : str
        What to use as a tag for the produced data.
    """

    total_len = config.Property(default=1, proptype=int)
    tag = config.Property(proptype=str)

    def process(self):
        """Produce an empty stream and pass on.

        Returns
        -------
        cont : subclass of `memh5.BasicCont`
            Empty data stream.
        """
        if self.total_len == 0:
            raise pipeline.PipelineStopIteration

        self.log.debug(f"Producing test data '{self.tag}'...")

        cont = memh5.BasicCont()

        if "tag" not in cont.attrs:
            cont.attrs["tag"] = self.tag

        self.total_len -= 1
        return cont
