"""draco test utils."""
from typing import Tuple

import numpy as np
from caput import config, memh5, pipeline

from ..core.task import SingleTask
from . import random


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


def mock_freq_data(
    freq: np.ndarray,
    ntime: int,
    delaycut: float,
    noise: float = 0.0,
    bad_freq: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Make mock delay data with a constant delay spectrum up to a specified cut.

    Parameters
    ----------
    freq
        Frequencies of each channel (in MHz).
    ntime
        Number of independent time samples.
    delaycut
        Cutoff in us.
    noise
        RMS noise level in the data.
    bad_freq
        A list of bad frequencies to mask out.

    Return
    ------
    data
        The 2D data array [freq, time].
    weights
        The 2D weights data [freq, time].
    """
    nfreq = len(freq)
    ndelay = nfreq

    df = np.abs(freq[1] - freq[0])

    delays = np.fft.fftfreq(ndelay, df)
    dspec = np.where(np.abs(delays) < delaycut, 1.0, 0.0)

    # Construct a set of delay spectra
    delay_spectra = random.complex_normal((ntime, ndelay))
    delay_spectra *= dspec**0.5

    # Transform to get frequency spectra
    data = np.fft.fft(delay_spectra, axis=-1).T.copy()

    if noise > 0:
        data += noise * random.complex_normal(data.shape)

    weights = np.empty_like(data)
    weights[:] = 1.0 / noise**2

    if bad_freq:
        data[bad_freq] = 0.0
        weights[bad_freq] = 0.0

    return data, weights
