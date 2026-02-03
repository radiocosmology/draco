"""Add the effects of instrumental noise into the simulation.

This is separated out into multiple tasks. The first, :class:`ReceiverTemperature`
adds in the effects of instrumental noise bias into the data. The second,
:class:`SampleNoise`, takes a timestream which is assumed to be the expected (or
average) value and returns an observed time stream. The :class: `GaussianNoise`
adds in the effects of a Gaussian distributed noise into visibility data.
The :class: `GaussianNoiseDataset` replaces visibility data with Gaussian distributed noise,
using the variance of the noise estimate in the existing data.
"""

import numpy as np
from caput import config
from caput.astro.constants import STELLAR_S
from caput.pipeline import exceptions, tasklib

from ..core import containers, io
from ..util import random, tools


class ReceiverTemperature(tasklib.base.ContainerTask):
    """Add a basic receiver temperature term into the data.

    This class adds in an uncorrelated, frequency and time independent receiver
    noise temperature to the data. As it is uncorrelated this will only affect
    the auto-correlations. Note this only adds in the offset to the visibility,
    to add the corresponding random fluctuations to subsequently use the
    :class:`SampleNoise` task.

    Attributes
    ----------
    recv_temp : float
        The receiver temperature in Kelvin.
    """

    recv_temp = config.Property(proptype=float, default=0.0)

    def process(self, data):
        """Iterate over the products to find the auto-correlations and add the noise into them."""
        for pi, prod in enumerate(data.prodstack):
            # Great an auto!
            if prod[0] == prod[1]:
                data.vis[:, pi] += self.recv_temp

        return data


class GaussianNoiseDataset(tasklib.base.ContainerTask, tasklib.random.RandomTask):
    """Generates a Gaussian distributed noise dataset using the noise estimates of an existing dataset.

    Attributes
    ----------
    dataset : string
        The dataset to fill with gaussian noise. If set to 'vis', will ensure
        autos are real. If not set, will look for a default dataset in a list
        of known containers.
    """

    dataset = config.Property(proptype=str, default=None)
    in_place = config.Property(proptype=bool, default=True)

    def process(self, data):
        """Generates a Gaussian distributed noise dataset given the provided dataset's visibility weights.

        Parameters
        ----------
        data : :class:`VisContainer`
            Any dataset which contains a vis and weight attribute.
            Note the modification is done in place.

        Returns
        -------
        data_noise : same as parameter `data`
            The previous dataset with the visibility replaced with
            a Gaussian distributed noise realisation.

        """
        if self.dataset is None:
            if isinstance(data, containers.DataWeightContainer):
                dataset_name = data._data_dset_name
            else:
                raise ValueError(
                    f"No default dataset known for {type(data)} container."
                )
        else:
            dataset_name = self.dataset

        if dataset_name not in data:
            raise config.CaputConfigError(
                f"Dataset '{dataset_name}' does not exist in container {type(data)}."
            )

        # Distribute in something other than `stack`
        data.redistribute("freq")

        # If requested, create a new output container
        if not self.in_place:
            out = data.copy()
            out.redistribute("freq")
        else:
            out = data

        # Replace visibilities with noise
        dset = out[dataset_name][:].local_array
        weight = data.weight[:].local_array
        if np.iscomplexobj(dset):
            random.complex_normal(
                scale=tools.invert_no_zero(weight) ** 0.5,
                out=dset,
                rng=self.rng,
            )
        else:
            self.rng.standard_normal(out=dset)
            dset *= tools.invert_no_zero(weight) ** 0.5

        # We need to loop to ensure the autos are real and have the correct variance
        if dataset_name == "vis":
            for si, prod in enumerate(data.prodstack):
                if prod[0] == prod[1]:
                    # This is an auto-correlation
                    dset[:, si].real *= 2**0.5
                    dset[:, si].imag = 0.0

        return out


class MultipleNoiseRealizationsMixin:
    """Generates multiple noise realizations with the same underlying statistics.

    This is a non-functional class mixin that must be combined with a class whose
    process method generates a single noise realization.

    Attributes
    ----------
    niter : int
        Number of noise realizations to generate.
    """

    niter = config.Property(proptype=int, default=1)
    in_place = False

    def setup(self, data1, data2=None):
        """Save the data as a class attribute.

        If multiple input containers are provided, the class alternates between
        them when generating the noise realization. This enables cross power
        spectrum analysis.

        Parameters
        ----------
        data1 : :class:`VisContainer`
            Any dataset which contains a vis and weight attribute.
        data2 : :class:`VisContainer`
            Any dataset which contains a vis and weight attribute.
        """
        self.data = [data1]
        if data2 is not None:
            self.data.append(data2)

    def process(self):
        """Generate a noise realization.

        The variance will be set to the inverse of the
        weight dataset of the container provided on setup.
        """
        if self._count == self.niter:
            raise exceptions.PipelineStopIteration

        return super().process(self.data[self._count % len(self.data)])


class MultipleGaussianNoiseDatasets(
    MultipleNoiseRealizationsMixin, GaussianNoiseDataset
):
    """Generates multiple Gaussian distributed noise datasets."""


class GaussianNoise(tasklib.base.ContainerTask, tasklib.random.RandomTask):
    """Add Gaussian distributed noise to a visibility dataset.

    Note that this is an approximation to the actual noise distribution good only
    when T_recv >> T_sky and delta_time * delta_freq >> 1.

    Attributes
    ----------
    ndays : float
        Multiplies the number of samples in each measurement.
    set_weights : bool
        Set the weights to the appropriate values.
    add_noise : bool
        Add Gaussian noise to the visibilities. By default this is True, but it may be
        desirable to only set the weights.
    recv_temp : bool
        The temperature of the noise to add.
    """

    recv_temp = config.Property(proptype=float, default=50.0)
    ndays = config.Property(proptype=float, default=733.0)
    set_weights = config.Property(proptype=bool, default=True)
    add_noise = config.Property(proptype=bool, default=True)

    def setup(self, manager=None):
        """Set the telescope instance if a manager object is given.

        This is used to simulate noise for visibilities that are stacked
        over redundant baselines.

        Parameters
        ----------
        manager : manager.ProductManager, optional
            The telescope/manager used to set the `redundancy`. If not set,
            `redundancy` is derived from the data.
        """
        if manager is not None:
            self.telescope = io.get_telescope(manager)
        else:
            self.telescope = None

    def process(self, data):
        """Generate a noisy dataset.

        Parameters
        ----------
        data : :class:`containers.SiderealStream` or :class:`containers.TimeStream`
            The expected (i.e. noiseless) visibility dataset. Note the modification
            is done in place.

        Returns
        -------
        data_noise : same as parameter `data`
            The sampled (i.e. noisy) visibility dataset.
        """
        data.redistribute("freq")

        visdata = data.vis[:]

        # Get the time interval
        if isinstance(data, containers.SiderealStream):
            dt = 240 * (data.ra[1] - data.ra[0]) * STELLAR_S
            ntime = len(data.ra)
        else:
            dt = data.time[1] - data.time[0]
            ntime = len(data.time)

        # TODO: this assumes uniform channels
        df = data.index_map["freq"]["width"][0] * 1e6
        nfreq = data.vis.local_shape[0]
        nprod = len(data.prodstack)
        ninput = len(data.index_map["input"])

        # Consider if this data is stacked over redundant baselines or not.
        if (self.telescope is not None) and (nprod == self.telescope.nbase):
            redundancy = self.telescope.redundancy
        elif nprod == ninput * (ninput + 1) / 2:
            redundancy = np.ones(nprod)
        else:
            raise ValueError("Unexpected number of products")

        # Calculate the number of samples, this is a 1D array for the prod axis.
        nsamp = int(self.ndays * dt * df) * redundancy
        std = self.recv_temp / np.sqrt(nsamp)

        if self.add_noise:
            noise = random.complex_normal(
                size=(nfreq, nprod, ntime),
                scale=std[np.newaxis, :, np.newaxis],
                rng=self.rng,
            )

            # Iterate over the products to find the auto-correlations and add the noise
            for pi, prod in enumerate(data.prodstack):
                # Auto: multiply by sqrt(2) because auto has twice the variance
                if prod[0] == prod[1]:
                    visdata[:, pi].real += np.sqrt(2) * noise[:, pi].real

                else:
                    visdata[:, pi] += noise[:, pi]

        # Construct and set the correct weights in place
        if self.set_weights:
            for lfi, fi in visdata.enumerate(0):
                data.weight[fi] = 1.0 / std[:, np.newaxis] ** 2

        return data


class SampleNoise(tasklib.base.ContainerTask, tasklib.random.RandomTask):
    """Add properly distributed noise to a visibility dataset.

    This task draws properly (complex Wishart) distributed samples from an input
    visibility dataset which is assumed to represent the expectation.

    See http://link.springer.com/article/10.1007%2Fs10440-010-9599-x for a
    discussion of the Bartlett decomposition for complex Wishart distributed
    quantities.

    Attributes
    ----------
    sample_frac : float
        Multiplies the number of samples in each measurement. For instance this
        could be a duty cycle if the correlator was not keeping up, or could be
        larger than one if multiple measurements were combined.
    set_weights : bool
        Set the weights to the appropriate values.
    """

    sample_frac = config.Property(proptype=float, default=1.0)
    seed = config.Property(proptype=int, default=None)
    set_weights = config.Property(proptype=bool, default=True)

    def process(self, data_exp):
        """Generate a noisy dataset.

        Parameters
        ----------
        data_exp : :class:`containers.SiderealStream` or :class:`containers.TimeStream`
            The expected (i.e. noiseless) visibility dataset. Must be the full
            triangle. Make sure you have added an instrumental noise bias if you
            want instrumental noise.

        Returns
        -------
        data_samp : same as parameter `data_exp`
            The sampled (i.e. noisy) visibility dataset.
        """
        from ..util import _fast_tools

        data_exp.redistribute("freq")

        nfeed = len(data_exp.index_map["input"])

        # Get a reference to the base MPIArray. Attempting to do this in the
        # loop fails if not all ranks enter the loop (as there is an implied MPI
        # Barrier)
        vis_data = data_exp.vis[:]

        # Get the time interval
        if isinstance(data_exp, containers.SiderealStream):
            dt = 240 * (data_exp.ra[1] - data_exp.ra[0]) * STELLAR_S
        else:
            dt = data_exp.time[1] - data_exp.time[0]

        # Iterate over frequencies
        for lfi, fi in vis_data.enumerate(0):
            # Get the frequency interval
            df = data_exp.index_map["freq"]["width"][fi] * 1e6

            # Calculate the number of samples
            nsamp = int(self.sample_frac * dt * df)

            # Iterate over time
            for lti, ti in vis_data.enumerate(2):
                # Unpack visibilites into full matrix
                vis_utv = vis_data[lfi, :, lti].view(np.ndarray).copy()
                vis_mat = np.zeros((nfeed, nfeed), dtype=vis_utv.dtype)
                _fast_tools._unpack_product_array_fast(
                    vis_utv, vis_mat, np.arange(nfeed), nfeed
                )

                vis_samp = random.complex_wishart(vis_mat, nsamp, rng=self.rng) / nsamp

                vis_data[lfi, :, lti] = vis_samp[np.triu_indices(nfeed)]

            # Construct and set the correct weights in place
            if self.set_weights:
                autos = tools.extract_diagonal(vis_data[lfi], axis=0).real
                weight_fac = nsamp**0.5 / autos
                tools.apply_gain(
                    data_exp.weight[fi][np.newaxis, ...],
                    weight_fac[np.newaxis, ...],
                    out=data_exp.weight[fi][np.newaxis, ...],
                )

        return data_exp


class FreqCorrelatedNoise(tasklib.base.ContainerTask, tasklib.random.RandomTask):
    """Generate frequency-correlated noise using Cholesky factors.

    This task uses precomputed Cholesky decompositions of the frequency-frequency
    noise covariance (stored in a FreqNoiseModel container) to simulate
    a noise realization into a VisGridStream container.

    Attributes
    ----------
    save_redundancy : bool
        If True, save the redundancy of each visibility.  Default is False.
    """

    save_redundancy = config.Property(proptype=bool, default=False)

    def process(self, noise_model):
        """Simulate noise into a VisGridStream container.

        Parameters
        ----------
        noise_model : containers.FreqNoiseModel
            Input noise model containing Cholesky factors and baseline redundancy.

        Returns
        -------
        out : containers.VisGridStream
            Container filled with a frequency-correlated noise realization.
        """
        noise_model.redistribute("ra")

        out = containers.VisGridStream(
            axes_from=noise_model,
            attrs_from=noise_model,
            distributed=noise_model.distributed,
            comm=noise_model.comm,
        )
        out.redistribute("ra")

        if self.save_redundancy:
            out.add_dataset("redundancy")
            out.redundancy[:] = noise_model.redundancy[:][..., np.newaxis]

        # Compute the redundancy factor that determines how the noise depends
        # on north-south baseline distance
        redundancy = noise_model.redundancy[:]
        inv_sqrt_redundancy = tools.invert_no_zero(np.sqrt(redundancy))

        # Dereference datasets
        L = noise_model.freq_cov[:].local_array
        weight = noise_model.weight[:].local_array

        ovis = out.vis[:].local_array
        oweight = out.weight[:].local_array

        npol, nfreq, new, nns, nra = ovis.shape

        # Loop over pol and ew to reduce memory usage
        for pp in range(npol):
            for ee in range(new):
                # Generate a set of complex random numbers with unit standard deviation
                z = np.empty((nra, nfreq, nns), dtype=ovis.dtype)

                random.complex_normal(
                    scale=1.0,
                    out=z,
                    rng=self.rng,
                )

                # Multiply by the Cholesky decomposition of the freq-freq covariance matrix
                # to introduce the desired correlation as a function of frequency, then divide
                # by the square root of the redundancy of the baseline.
                sz = np.matmul(L[pp, ee], z) * inv_sqrt_redundancy[pp, ee]

                ovis[pp, :, ee] = sz.transpose(1, 2, 0)

                oweight[pp, :, ee] = (
                    weight[pp, :, ee, np.newaxis, :] * redundancy[pp, ee, :, np.newaxis]
                )

        # Ensure that the visibility matrix is Hermitian
        nyp = nns // 2 + 1
        slc_pos = slice(1, nyp)
        slc_neg = slice(-1, -nyp, -1)

        pconjmap = np.unique(
            [pj + pi for pi, pj in out.index_map["pol"]], return_inverse=True
        )[1]

        for pi, po in enumerate(pconjmap):
            ovis[po, :, 0, slc_neg, :] = ovis[pi, :, 0, slc_pos, :].conj()
            if pi == po:
                ovis[po, :, 0, 0, :] = ovis[pi, :, 0, 0, :].real * 2**0.5

        return out


class MultipleFreqCorrelatedNoise(MultipleNoiseRealizationsMixin, FreqCorrelatedNoise):
    """Generates multiple realizations of frequency-correlated noise."""
