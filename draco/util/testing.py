"""draco test utils."""

from typing import List, Optional, Tuple, Union

import numpy as np
from caput import config, memh5, pipeline

from ..core.containers import SiderealStream
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
    ndata: Optional[int] = None,
    noise: float = 0.0,
    bad_freq: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
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
    ndata
        Number of correlated data sets. If not set (i.e. `None`) then do no add a
        dataset axis.
    noise
        RMS noise level in the data.
    bad_freq
        A list of bad frequencies to mask out.
    rng
        The random number generator to use.

    Return
    ------
    data
        The 2D/3D data array [dataset, freq, time]. If ndata is `None` then the dataset
        axis is dropped.
    weights
        The 2D weights data [freq, time].
    """
    nfreq = len(freq)
    ndelay = nfreq

    df = np.abs(freq[1] - freq[0])

    delays = np.fft.fftfreq(ndelay, df)
    dspec = np.where(np.abs(delays) < delaycut, 1.0, 0.0)

    # Construct a set of delay spectra
    delay_spectra = random.complex_normal(size=(ntime, ndelay), rng=rng)
    delay_spectra *= dspec**0.5

    # Generate the noise realisation
    outshape = (nfreq, ntime)
    if ndata is not None:
        outshape = (ndata, *outshape)
    data = noise * random.complex_normal(size=outshape, rng=rng)

    # Transform to get frequency spectra
    data += np.fft.fft(delay_spectra, axis=-1).T

    weights = np.empty(data.shape, dtype=np.float64)
    weights[:] = 1.0 / noise**2

    if bad_freq:
        data[..., bad_freq, :] = 0.0
        weights[..., bad_freq, :] = 0.0

    return data, weights


class RandomFreqData(random.RandomTask):
    """Generate a random sidereal stream with structure in delay.

    Attributes
    ----------
    num_realisation
        How many to generate in subsequent process calls.
    num_correlated
        The number of correlated realisations output per cycle.
    num_ra
        The number of RA samples in the output.
    num_base
        The number of baselines in the output.
    freq_start, freq_end
        The start and end frequencies.
    num_freq
        The number of frequency channels.
    delay_cut
        The maximum delay in the data in us.
    noise
        The RMS noise level.
    """

    num_realisation = config.Property(proptype=int, default=1)
    num_correlated = config.Property(proptype=int, default=None)

    num_ra = config.Property(proptype=int)
    num_base = config.Property(proptype=int)

    freq_start = config.Property(proptype=float, default=800.0)
    freq_end = config.Property(proptype=float, default=400.0)
    num_freq = config.Property(proptype=int, default=1024)

    delay_cut = config.Property(proptype=float, default=0.2)
    noise = config.Property(proptype=float, default=1e-5)

    def next(self) -> Union[SiderealStream, List[SiderealStream]]:
        """Generate correlated sidereal streams.

        Returns
        -------
        streams
            Either a single stream (if num_correlated=None), or a list of correlated
            streams.
        """
        if self.num_realisation == 0:
            raise pipeline.PipelineStopIteration()

        # Construct the frequency axis
        freq = np.linspace(
            self.freq_start,
            self.freq_end,
            self.num_freq,
            endpoint=False,
        )

        streams = []

        # Construct all the sidereal streams
        for ii in range(self.num_correlated or 1):
            stream = SiderealStream(
                input=5,  # Probably should be something smarter
                freq=freq,
                ra=self.num_ra,
                stack=self.num_base,
            )
            stream.redistribute("stack")
            ssv = stream.vis[:].local_array
            ssw = stream.weight[:].local_array

            streams.append((stream, ssv, ssw))

        # Iterate over baselines and construct correlated realisations for each, and
        # then insert them into each of the sidereal streams
        for ii in range(ssv.shape[1]):
            d, w = mock_freq_data(
                freq,
                self.num_ra,
                self.delay_cut,
                ndata=(self.num_correlated or 1),
                noise=self.noise,
            )

            for jj, (_, ssv, ssw) in enumerate(streams):
                ssv[:, ii] = d[jj]
                ssw[:, ii] = w[jj]

        self.num_realisation -= 1

        # Don't return a list of streams if num_correlated is None
        if self.num_correlated is None:
            return streams[0][0]

        return [s for s, *_ in streams]
