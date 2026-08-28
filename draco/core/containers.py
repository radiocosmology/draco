"""Distributed containers for holding various types of analysis data.

Containers
==========
- :py:class:`Map`
- :py:class:`SiderealStream`
- :py:class:`SystemSensitivity`
- :py:class:`RFIMask`
- :py:class:`TimeStream`
- :py:class:`GridBeam`
- :py:class:`TrackBeam`
- :py:class:`MModes`
- :py:class:`SVDModes`
- :py:class:`KLModes`
- :py:class:`VisGridStream`
- :py:class:`HybridVisStream`
- :py:class:`HybridVisMModes`
- :py:class:`RingMap`
- :py:class:`RingMapMask`
- :py:class:`RingMapTaper`
- :py:class:`CommonModeGainData`
- :py:class:`CommonModeSiderealGainData`
- :py:class:`GainData`
- :py:class:`SiderealGainData`
- :py:class:`StaticGainData`
- :py:class:`DelayCutoff`
- :py:class:`DelaySpectrum`
- :py:class:`Powerspectrum2D`
- :py:class:`SVDSpectrum`
- :py:class:`FrequencyStack`
- :py:class:`FrequencyStackByPol`
- :py:class:`MockFrequencyStack`
- :py:class:`MockFrequencyStackByPol`
- :py:class:`SourceCatalog`
- :py:class:`SpectroscopicCatalog`
- :py:class:`FormedBeam`
- :py:class:`FormedBeamHA`
- :py:class:`FormedBeamMask`
- :py:class:`FormedBeamHAMask`
- :py:class:`LocalizedRFIMask`
- :py:class:`LocalizedSiderealRFIMask`

Container Base Classes
----------------------
- :py:class:`TODContainer`
- :py:class:`VisContainer`
- :py:class:`SampleVarianceContainer`
- :py:class:`FreqContainer`
- :py:class:`SiderealContainer`
- :py:class:`MContainer`

Helper Routines
---------------
These routines are designed to be replaced by other packages trying to insert
their own custom container types.

- :py:meth:`empty_timestream`
"""

from typing import ClassVar

import numpy as np
from caput import memdata
from caput.containers import (
    COMPRESSION,
    COMPRESSION_OPTS,
    ContainerPrototype,
    DataWeightContainer,
    TableSpec,
    tod,
)
from cora.core.containers import (
    CosmologyContainer,
    HealpixContainer,
)
from cora.core.containers import (
    Map as _CoraMap,
)

from ..util import tools


class TODContainer(ContainerPrototype, tod.TOData):
    """A pipeline container for time ordered data.

    This works like a normal :class:`ContainerPrototype` container, with the added
    ability to be concatenated, and treated like a a :class:`tod.TOData`
    instance.
    """

    _axes = ("time",)


class VisBase(DataWeightContainer):
    """A very basic class for visibility data.

    For better support for input/prod/stack structured data use `VisContainer`.
    """

    _data_dset_name = "vis"
    _weight_dset_name = "vis_weight"

    @property
    def vis(self):
        """The visibility like dataset."""
        return self.datasets["vis"]


class VisContainer(VisBase):
    """A base container for holding a visibility dataset.

    This works like a :class:`ContainerPrototype` container, with the
    ability to create visibility specific axes, if they are not
    passed as a kwargs parameter.

    Additionally this container has visibility specific defined properties
    such as 'vis', 'weight', 'freq', 'input', 'prod', 'stack',
    'prodstack', 'conjugate'.

    Parameters
    ----------
    axes_from : `caput.containers.ContainerPrototype`, optional
        Another container to copy axis definitions from. Must be supplied as
        keyword argument.
    attrs_from : `caput.containers.ContainerPrototype`, optional
        Another container to copy attributes from. Must be supplied as keyword
        argument. This applies to attributes in default datasets too.
    kwargs : dict
        Should contain entries for all other axes.
    """

    _axes = ("input", "prod", "stack")

    def __init__(self, *args, **kwargs):
        # Resolve product map
        prod = None
        if "prod" in kwargs:
            prod = kwargs["prod"]
        elif ("axes_from" in kwargs) and ("prod" in kwargs["axes_from"].index_map):
            prod = kwargs["axes_from"].index_map["prod"]

        # Resolve input map
        inputs = None
        if "input" in kwargs:
            inputs = kwargs["input"]
        elif ("axes_from" in kwargs) and ("input" in kwargs["axes_from"].index_map):
            inputs = kwargs["axes_from"].index_map["input"]

        # Resolve stack map
        stack = None
        if "stack" in kwargs:
            stack = kwargs["stack"]
        elif ("axes_from" in kwargs) and ("stack" in kwargs["axes_from"].index_map):
            stack = kwargs["axes_from"].index_map["stack"]

        # Automatically construct product map from inputs if not given
        if prod is None and inputs is not None:
            nfeed = inputs if isinstance(inputs, int) else len(inputs)
            kwargs["prod"] = np.array(
                [[fi, fj] for fi in range(nfeed) for fj in range(fi, nfeed)]
            )

        if stack is None and prod is not None:
            stack = np.empty_like(prod, dtype=[("prod", "<u4"), ("conjugate", "u1")])
            stack["prod"][:] = np.arange(len(prod))
            stack["conjugate"] = 0
            kwargs["stack"] = stack

        # Call initializer from `ContainerPrototype`
        super().__init__(*args, **kwargs)

        # `axes_from` can be provided via the `copy_from` argument, so
        # need to update kwargs to account
        if ("axes_from" not in kwargs) and ("copy_from" in kwargs):
            kwargs["axes_from"] = kwargs.pop("copy_from")

        reverse_map_stack = None
        # Create reverse map
        if "reverse_map_stack" in kwargs:
            # If axis is an integer, turn into an arange as a default definition
            if isinstance(kwargs["reverse_map_stack"], int):
                reverse_map_stack = np.arange(kwargs["reverse_map_stack"])
            else:
                reverse_map_stack = kwargs["reverse_map_stack"]
        # If not set in the arguments copy from another object if set
        elif ("axes_from" in kwargs) and ("stack" in kwargs["axes_from"].reverse_map):
            reverse_map_stack = kwargs["axes_from"].reverse_map["stack"]

        # Set the reverse_map['stack'] if we have a definition,
        # otherwise do NOT throw an error, errors are thrown in
        # classes that actually need a reverse stack
        if reverse_map_stack is not None:
            self.create_reverse_map("stack", reverse_map_stack)

    @property
    def input(self):
        """The correlated inputs."""
        return self.index_map["input"]

    @property
    def prod(self):
        """All the pairwise products that are represented in the data."""
        return self.index_map["prod"]

    @property
    def stack(self):
        """The stacks definition as an index (and conjugation) of a member product."""
        return self.index_map["stack"]

    @property
    def prodstack(self):
        """A pair of input indices representative of those in the stack.

        Note, these are correctly conjugated on return, and so calculations
        of the baseline and polarisation can be done without additionally
        looking up the stack conjugation.
        """
        if not self.is_stacked:
            return self.prod

        t = self.index_map["prod"][:][self.index_map["stack"]["prod"]]

        prodmap = t.copy()
        conj = self.stack["conjugate"]
        prodmap["input_a"] = np.where(conj, t["input_b"], t["input_a"])
        prodmap["input_b"] = np.where(conj, t["input_a"], t["input_b"])

        return prodmap

    @property
    def is_stacked(self):
        """Test if the data has been stacked or not."""
        return len(self.stack) != len(self.prod)


class SampleVarianceContainer(ContainerPrototype):
    """Base container for holding the sample variance over observations.

    This works like :class:`ContainerPrototype` but provides additional capabilities
    for containers that may be used to hold the sample mean and variance over
    complex-valued observations.  These capabilities include automatic definition
    of the component axis, properties for accessing standard datasets, properties
    that rotate the sample variance into common bases, and a `sample_weight` property
    that provides an equivalent to the `weight` dataset that is determined from the
    sample variance over observations.

    Subclasses must include a `sample_variance` and `nsample` dataset
    in there `_dataset_spec` dictionary.  They must also specify a
    `_mean` property that returns the dataset containing the mean over observations.
    """

    _axes = ("component",)

    def __init__(self, *args, **kwargs):
        # Set component axis to default real-imaginary basis if not already provided
        if "component" not in kwargs:
            kwargs["component"] = np.array(
                [("real", "real"), ("real", "imag"), ("imag", "imag")],
                dtype=[("component_a", "<U8"), ("component_b", "<U8")],
            )

        super().__init__(*args, **kwargs)

    @property
    def component(self):
        """Get the component axis."""
        return self.index_map["component"]

    @property
    def sample_variance(self):
        """Convenience access to the sample variance dataset.

        Returns
        -------
        C: np.ndarray[ncomponent, ...]
            The variance over the dimension that was stacked
            (e.g., sidereal days, holographic observations)
            in the default real-imaginary basis. The array is packed
            into upper-triangle format such that the component axis
            contains [('real', 'real'), ('real', 'imag'), ('imag', 'imag')].
        """
        if "sample_variance" in self.datasets:
            return self.datasets["sample_variance"]

        raise KeyError("Dataset 'sample_variance' not initialised.")

    @property
    def sample_variance_iq(self):
        """Rotate the sample variance to the in-phase/quadrature basis.

        Returns
        -------
        C: np.ndarray[ncomponent, ...]
            The `sample_variance` dataset in the in-phase/quadrature basis,
            packed into upper triangle format such that the component axis
            contains [('I', 'I'), ('I', 'Q'), ('Q', 'Q')].
        """
        C = self.sample_variance[:].view(np.ndarray)

        # Construct rotation coefficients from average vis angle
        phi = np.angle(self._mean[:].view(np.ndarray))
        cc = np.cos(phi) ** 2
        cs = np.cos(phi) * np.sin(phi)
        ss = np.sin(phi) ** 2

        # Rotate the covariance matrix from real-imag to in-phase/quadrature
        Cphi = np.zeros_like(C)
        Cphi[0] = cc * C[0] + 2 * cs * C[1] + ss * C[2]
        Cphi[1] = -cs * C[0] + (cc - ss) * C[1] + cs * C[2]
        Cphi[2] = ss * C[0] - 2 * cs * C[1] + cc * C[2]

        return Cphi

    @property
    def sample_variance_amp_phase(self):
        """Calculate the amplitude/phase covariance.

        This interpretation is only valid if the fractional
        variations in the amplitude and phase are small.

        Returns
        -------
        C: np.ndarray[ncomponent, ...]
            The observed amplitude/phase covariance matrix, packed
            into upper triangle format such that the component axis
            contains [('amp', 'amp'), ('amp', 'phase'), ('phase', 'phase')].
        """
        # Rotate to in-phase/quadrature basis and then
        # normalize by squared amplitude to convert to
        # fractional units (amplitude) and radians (phase).
        return self.sample_variance_iq * tools.invert_no_zero(
            np.abs(self._mean[:][np.newaxis, ...]) ** 2
        )

    @property
    def nsample(self):
        """Get the nsample dataset if it exists."""
        if "nsample" in self.datasets:
            return self.datasets["nsample"]

        raise KeyError("Dataset 'nsample' not initialised.")

    @property
    def sample_weight(self):
        """Calculate a weight from the sample variance.

        Returns
        -------
        weight: np.ndarray[...]
            The trace of the `sample_variance` dataset is used
            as an estimate of the total variance and divided by the
            `nsample` dataset to yield the uncertainty on the mean.
            The inverse of this quantity is returned, and can be compared
            directly to the `weight` dataset.
        """
        C = self.sample_variance[:].view(np.ndarray)
        nsample = self.nsample[:].view(np.ndarray)

        return nsample * tools.invert_no_zero(C[0] + C[2])


class FreqContainer(ContainerPrototype):
    """A pipeline container for data with a frequency axis.

    This works like a normal :class:`ContainerPrototype` container, but already has a freq
    axis defined, and specific properties for dealing with frequencies.
    """

    _axes = ("freq",)

    @property
    def freq(self):
        """The physical frequency associated with each entry of the time axis.

        By convention this property should return the frequency in MHz at the centre
        of each of frequency channel.
        """
        try:
            return self.index_map["freq"][:]["centre"]
        # Need to check for both types as different numpy versions return
        # different exceptions.
        except (IndexError, ValueError):
            return self.index_map["freq"][:]


class SiderealContainer(ContainerPrototype):
    """A pipeline container for data with an RA axis.

    This works like a normal :class:`ContainerPrototype` container, but already has an RA
    axis defined, and specific properties for dealing with this axis.

    Note that Right Ascension is a fairly ambiguous term. What is typically meant
    here is the Local Stellar Angle, which is the transiting RA in CIRS coordinates.
    This is similar to J2000/ICRS with the minimal amount of coordinate rotation to
    account for the polar axis precession.

    Parameters
    ----------
    ra : array or int, optional
        Either the explicit locations of samples of the RA axis, or if passed an
        integer interpret this as a number of samples dividing the full sidereal day
        and create an axis accordingly.
    """

    _axes = ("ra",)

    def __init__(self, ra=None, *args, **kwargs):
        # Allow the passing of a number of samples for the RA axis
        if ra is not None:
            if isinstance(ra, int):
                ra = np.linspace(0.0, 360.0, ra, endpoint=False)
            kwargs["ra"] = ra

        super().__init__(*args, **kwargs)

    @property
    def ra(self):
        """The RA in degrees associated with each sample of the RA axis."""
        return self.index_map["ra"][:]


class MContainer(ContainerPrototype):
    """Container for holding m-mode type data.

    Note this container will have an `msign` axis even though not all m-mode based
    data needs one. As always this is not an issue, datasets that don't need it are
    not required to list it in their `axes` list.

    Parameters
    ----------
    mmax : integer, optional
        Largest m to be held.
    oddra : bool, optional
        Does this MContainer come from an underlying odd number of RA points. This
        determines if the largest negative m is filled or not (it is for odd=True, not
        for odd=False). Default is odd=False.
    """

    _axes = ("m", "msign")

    def __init__(
        self, mmax: int | None = None, oddra: bool | None = None, *args, **kwargs
    ):
        # Set up axes from passed arguments
        if mmax is not None:
            kwargs["m"] = mmax + 1

        # Ensure the sign axis is set correctly
        kwargs["msign"] = np.array(["+", "-"])

        super().__init__(*args, **kwargs)

        # Set oddra, prioritising an explicit keyword argument over anything else
        if oddra is not None:
            self.attrs["oddra"] = oddra
        elif "oddra" not in self.attrs:
            self.attrs["oddra"] = False

    @property
    def mmax(self) -> int:
        """The maximum m stored."""
        return int(self.index_map["m"][-1])

    @property
    def oddra(self) -> bool:
        """Whether this represents an odd or even number of RA points."""
        return self.attrs["oddra"]


class Map(FreqContainer, _CoraMap):
    """Container for holding multi-frequency sky maps.

    The maps are packed in format `[freq, pol, pixel]` where the polarisations
    are Stokes I, Q, U and V, and the pixel dimension stores a Healpix map.

    This is just an extension of the `Map` class in `cora` with a modified
    frequency index map.

    Parameters
    ----------
    nside : int
        The nside of the Healpix maps.
    polarisation : bool, optional
        If `True` all Stokes parameters are stored, if `False` only Stokes I is
        stored.
    """


class SiderealStream(
    FreqContainer, VisContainer, SiderealContainer, SampleVarianceContainer
):
    """A container for holding a visibility dataset in sidereal time.

    Parameters
    ----------
    ra : int
        The number of points to divide the RA axis up into.
    """

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (32, 512, 2048),
            "truncate": {
                "weight_dataset": "vis_weight",
            },
        },
        "vis_weight": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (32, 512, 2048),
            "truncate": True,
        },
        "input_flags": {
            "axes": ["input", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "gain": {
            "axes": ["freq", "input", "ra"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "sample_variance": {
            "axes": ["component", "freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (1, 32, 512, 2048),
            "truncate": True,
        },
        "nsample": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.uint16,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (32, 512, 2048),
        },
        "effective_ra": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (32, 512, 2048),
            "truncate": True,
        },
    }

    @property
    def gain(self):
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def input_flags(self):
        """Get the input_flags dataset."""
        return self.datasets["input_flags"]

    @property
    def _mean(self):
        """Get the vis dataset."""
        return self.datasets["vis"]

    @property
    def effective_ra(self):
        """Get the effective_ra dataset if it exists, None otherwise."""
        if "effective_ra" in self.datasets:
            return self.datasets["effective_ra"]

        raise KeyError("Dataset 'effective_ra' not initialised.")


class SystemSensitivity(FreqContainer, TODContainer):
    """A container for holding the total system sensitivity.

    This should be averaged/collapsed in the stack/prod axis
    to provide an overall summary of the system sensitivity.
    Two datasets are available: the measured noise from the
    visibility weights and the radiometric estimate of the
    noise from the autocorrelations.
    """

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
        "measured": {
            "axes": ["freq", "pol", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
        },
        "radiometer": {
            "axes": ["freq", "pol", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
        },
        "weight": {
            "axes": ["freq", "pol", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
        },
        "frac_lost": {
            "axes": ["freq", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
        },
    }

    @property
    def measured(self):
        """Get the measured noise dataset."""
        return self.datasets["measured"]

    @property
    def radiometer(self):
        """Get the radiometer estimate dataset."""
        return self.datasets["radiometer"]

    @property
    def weight(self):
        """Get the weight dataset."""
        return self.datasets["weight"]

    @property
    def frac_lost(self):
        """Get the frac_lost dataset."""
        return self.datasets["frac_lost"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]


class RFIMask(FreqContainer, TODContainer):
    """A container for holding an RFI mask for a timestream.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.
    """

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]


class RFIMaskByPol(RFIMask):
    """A container for holding a polarisation-dependent RFI mask as a function of time.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.
    """

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["pol", "freq", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "freq",
        }
    }

    @property
    def pol(self):
        """Get the pol index map."""
        return self.index_map["pol"]


class SiderealRFIMask(FreqContainer, SiderealContainer):
    """A container for holding an RFI mask for a sidereal stream.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.
    """

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "ra"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]


class SiderealRFIMaskByPol(SiderealRFIMask):
    """A container for holding a polarisation-dependent RFI mask as a function of RA.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.
    """

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["pol", "freq", "ra"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "freq",
        }
    }

    @property
    def pol(self):
        """Get the pol index map."""
        return self.index_map["pol"]


class BaselineMask(FreqContainer, TODContainer):
    """A container for holding a baseline-dependent mask for a timestream.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.

    Unlike RFIMask, this is distributed by default.
    """

    _axes = ("stack",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "stack", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]

    @property
    def stack(self):
        """The stack definition as an index (and conjugation) of a member product."""
        return self.index_map["stack"]


class SiderealBaselineMask(FreqContainer, SiderealContainer):
    """A container for holding a baseline-dependent mask for a sidereal stream.

    The mask is `True` for contaminated samples that should be excluded, and
    `False` for clean samples.

    Unlike SiderealRFIMask, this is distributed by default.
    """

    _axes = ("stack",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "stack", "ra"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]

    @property
    def stack(self):
        """The stack definition as an index (and conjugation) of a member product."""
        return self.index_map["stack"]


class TimeStream(FreqContainer, VisContainer, TODContainer):
    """A container for holding a visibility dataset in time.

    This should look similar enough to the CHIME
    :class:`~ch_util.andata.CorrData` container that they can be used
    interchangably in most cases.
    """

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["freq", "stack", "time"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (16, 256, 1024),
            "truncate": {
                "weight_dataset": "vis_weight",
            },
        },
        "vis_weight": {
            "axes": ["freq", "stack", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (16, 256, 1024),
            "truncate": True,
        },
        "input_flags": {
            "axes": ["input", "time"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": False,
        },
        "gain": {
            "axes": ["freq", "input", "time"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (16, 256, 1024),
        },
    }

    @property
    def gain(self):
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def input_flags(self):
        """Get the input_flags dataset."""
        return self.datasets["input_flags"]


class GridBeam(FreqContainer, DataWeightContainer):
    """Generic container for representing a 2D beam on a rectangular grid."""

    _axes = ("pol", "input", "theta", "phi")

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["freq", "pol", "input", "theta", "phi"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "pol", "input", "theta", "phi"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "quality": {
            "axes": ["freq", "pol", "input", "theta", "phi"],
            "dtype": np.uint8,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "gain": {
            "axes": ["freq", "input"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    _data_dset_name = "beam"
    _weight_dset_name = "weight"

    def __init__(self, coords="celestial", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs["coords"] = coords

    @property
    def beam(self):
        """Get the beam dataset."""
        return self.datasets["beam"]

    @property
    def quality(self):
        """Get the quality dataset."""
        return self.datasets["quality"]

    @property
    def gain(self):
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def coords(self):
        """Get the coordinates attribute."""
        return self.attrs["coords"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]

    @property
    def theta(self):
        """Get the theta axis."""
        return self.index_map["theta"]

    @property
    def phi(self):
        """Get the phi axis."""
        return self.index_map["phi"]


class HEALPixBeam(FreqContainer, HealpixContainer, DataWeightContainer):
    """Container for representing the spherical 2-d beam in a HEALPix grid.

    Parameters
    ----------
    ordering : {"nested", "ring"}
        The HEALPix ordering scheme used for the beam map.
    coords : {"celestial", "galactic", "telescope"}
        The coordinate system that the beam map is defined on.
    """

    _axes = ("pol", "input")

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["freq", "pol", "input", "pixel"],
            "dtype": [("Et", np.complex64), ("Ep", np.complex64)],
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "pol", "input", "pixel"],
            "dtype": [("Et", np.float32), ("Ep", np.float32)],
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    _data_dset_name = "beam"
    _weight_dset_name = "weight"

    def __init__(self, coords="unknown", ordering="unknown", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs["coords"] = coords
        self.attrs["ordering"] = ordering

    @property
    def beam(self):
        """Get the beam dataset."""
        return self.datasets["beam"]

    @property
    def ordering(self):
        """Get the ordering attribute."""
        return self.attrs["ordering"]

    @property
    def coords(self):
        """Get the coordinate attribute."""
        return self.attrs["coords"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]

    @property
    def nside(self):
        """Get the nsides of the map."""
        return int(np.sqrt(len(self.index_map["pixel"]) / 12))


class TrackBeam(FreqContainer, SampleVarianceContainer, DataWeightContainer):
    """Container for a sequence of beam samples at arbitrary locations on the sphere.

    The axis of the beam samples is 'pix', defined by the numpy.dtype
    [('theta', np.float32), ('phi', np.float32)].
    """

    _axes = ("pol", "input", "pix")

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["freq", "pol", "input", "pix"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 2, 64, 128),
            "truncate": {
                "weight_dataset": "weight",
            },
        },
        "weight": {
            "axes": ["freq", "pol", "input", "pix"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 2, 64, 128),
            "truncate": True,
        },
        "sample_variance": {
            "axes": ["component", "freq", "pol", "input", "pix"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (3, 64, 2, 64, 128),
            "truncate": True,
        },
        "nsample": {
            "axes": ["freq", "pol", "input", "pix"],
            "dtype": np.uint8,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 2, 64, 128),
        },
    }

    _data_dset_name = "beam"
    _weight_dset_name = "weight"

    def __init__(
        self,
        theta=None,
        phi=None,
        coords="celestial",
        track_type="drift",
        *args,
        **kwargs,
    ):
        if theta is not None and phi is not None:
            if len(theta) != len(phi):
                raise RuntimeError(
                    "theta and phi axes must have same length: "
                    f"({len(theta)} != {len(phi)})"
                )

            pix = np.zeros(
                len(theta), dtype=[("theta", np.float32), ("phi", np.float32)]
            )
            pix["theta"] = theta
            pix["phi"] = phi
            kwargs["pix"] = pix
        elif (theta is None) != (phi is None):
            raise RuntimeError("Both theta and phi coordinates must be specified.")

        super().__init__(*args, **kwargs)

        self.attrs["coords"] = coords
        self.attrs["track_type"] = track_type

    @property
    def beam(self):
        """Get the beam dataset."""
        return self.datasets["beam"]

    @property
    def gain(self):
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def coords(self):
        """Get the coordinates attribute."""
        return self.attrs["coords"]

    @property
    def track_type(self):
        """Get the track type attribute."""
        return self.attrs["track_type"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]

    @property
    def pix(self):
        """Get the pix axis."""
        return self.index_map["pix"]

    @property
    def _mean(self):
        """Get the beam dataset."""
        return self.datasets["beam"]


class MModes(FreqContainer, VisContainer, MContainer):
    """Parallel container for holding m-mode data.

    Attributes
    ----------
    vis : mpidataset.MPIArray
        Visibility array.
    weight : mpidataset.MPIArray
        Array of weights for each point.
    """

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["m", "msign", "freq", "stack"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
        "vis_weight": {
            "axes": ["m", "msign", "freq", "stack"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
    }


class SVDModes(MContainer, VisBase):
    """Parallel container for holding SVD m-mode data.

    Parameters
    ----------
    mmax : integer, optional
        Largest m to be held.
    """

    _axes = ("mode",)

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["m", "mode"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
        "vis_weight": {
            "axes": ["m", "mode"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
        "nmode": {
            "axes": ["m"],
            "dtype": np.int32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        },
    }

    @property
    def nmode(self):
        """Get the nmode dataset."""
        return self.datasets["nmode"]


class KLModes(SVDModes):
    """Parallel container for holding KL filtered m-mode data.

    Parameters
    ----------
    mmax : integer, optional
        Largest m to be held.
    """

    pass


class VisGridStream(FreqContainer, SiderealContainer, VisBase):
    """Visibilities gridded into a 2D array.

    Only makes sense for an array which is a cartesian grid.
    """

    _axes = ("pol", "ew", "ns")

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 64, 1, 64, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": {
                "weight_dataset": "weight",
            },
        },
        "vis_weight": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 64, 1, 64, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "redundancy": {
            "axes": ["pol", "ew", "ns", "ra"],
            "dtype": np.int32,
            "initialise": False,
            "distributed": False,
            "chunks": (1, 64, 1, 64, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    @property
    def redundancy(self):
        """Get the redundancy dataset."""
        if "redundancy" in self.datasets:
            return self.datasets["redundancy"]

        raise KeyError("Dataset 'redundancy' not initialised.")


class FilterFreqContainer(ContainerPrototype):
    """Base container for data that has undergone filtering along the frequency axis.

    This container behaves like a standard `ContainerPrototype`, but is tailored for
    datasets where a frequency-domain filter has been applied. It defines a
    `freq_sum` axis by default and provides additional logic to manage mutually
    exclusive filter and frequency covariance datasets.

    Expected Datasets:
    ------------------
    - `filter` or `complex_filter`
      Real or complex filter applied in the frequency domain.
    - `freq_cov` or `complex_freq_cov`:
      Real or complex covariance matrix in the frequency domain.
    """

    _axes = ("freq_sum",)

    def __init__(self, *args, **kwargs):
        for ax in ["freq_sum"]:
            if ax not in kwargs:
                if "axes_from" in kwargs and ax in kwargs["axes_from"].index_map:
                    kwargs[ax] = kwargs["axes_from"].index_map[ax]
                elif "freq" in kwargs:
                    kwargs[ax] = kwargs["freq"]
                elif "axes_from" in kwargs and "freq" in kwargs["axes_from"].index_map:
                    kwargs[ax] = kwargs["axes_from"].index_map["freq"]
            else:
                raise RuntimeError(f"Must provide {ax} or freq axis.")

        super().__init__(*args, **kwargs)

    def add_dataset(self, name):
        """Ensure that multiple filters and covariances are not created."""
        if name == "filter" and "complex_filter" in self.datasets:
            raise RuntimeError(
                "Requesting creation of real-valued filter but "
                "complex filter already exists."
            )
        if name == "complex_filter" and "filter" in self.datasets:
            raise RuntimeError(
                "Requesting creation of complex-valued filter but "
                "real filter already exists."
            )
        if name == "freq_cov" and "complex_freq_cov" in self.datasets:
            raise RuntimeError(
                "Requesting creation of real-valued freq_cov but "
                "complex_freq_cov already exists."
            )
        if name == "complex_freq_cov" and "freq_cov" in self.datasets:
            raise RuntimeError(
                "Requesting creation of complex_freq_cov but "
                "real-valued freq_cov already exists."
            )
        return super().add_dataset(name)

    @property
    def filter(self):
        """Return the filter dataset, if available."""
        if "filter" in self.datasets:
            return self.datasets["filter"]
        if "complex_filter" in self.datasets:
            return self.datasets["complex_filter"]

        raise KeyError("Dataset 'filter' not initialised.")

    @property
    def freq_cov(self):
        """Return the freq_cov dataset, if available."""
        if "freq_cov" in self.datasets:
            return self.datasets["freq_cov"]
        if "complex_freq_cov" in self.datasets:
            return self.datasets["complex_freq_cov"]

        raise KeyError("Dataset 'freq_cov' not initialised.")

    @property
    def swapped_freq_cov_axis(self):
        """Return the axis names of the freq_cov dataset with freq <--> freq_sum.

        This is useful for broadcasting the weight dataset when propagating the
        covariance matrix through linear operators.
        """
        swap = {"freq": "freq_sum", "freq_sum": "freq"}
        return np.array([swap.get(ax, ax) for ax in self.freq_cov.attrs["axis"]])


class HybridVisStream(FilterFreqContainer, FreqContainer, SiderealContainer, VisBase):
    """Visibilities beamformed only in the NS direction.

    This container has visibilities beam formed only in the NS direction to give a
    grid in elevation.
    """

    _axes = ("pol", "ew", "el")

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["pol", "freq", "ew", "el", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 1, 512, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "dirty_beam": {
            "axes": ["pol", "freq", "ew", "el", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 1, 512, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "vis_weight": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "elevation_vis_weight": {
            "axes": ["pol", "freq", "ew", "el", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 4, 512, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "effective_ra": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "nsample": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "filter": {
            "axes": ["pol", "freq", "freq_sum", "ew", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 96, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_filter": {
            "axes": ["pol", "freq", "freq_sum", "ew", "ra"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 96, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "freq_cov": {
            "axes": ["pol", "freq", "freq_sum", "ew", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 96, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_freq_cov": {
            "axes": ["pol", "freq", "freq_sum", "ew", "ra"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 96, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    def add_dataset(self, name):
        """Override base class to deal with elevation dependent vis weight."""
        if name == "vis_weight" and "elevation_vis_weight" in self.datasets:
            raise RuntimeError(
                "Requesting creation of elevation-independent weights but "
                "elevation-dependent weights already exist."
            )
        if name == "elevation_vis_weight":
            if "vis_weight" in self.datasets:
                raise RuntimeError(
                    "Requesting creation of elevation-dependent weights but "
                    "elevation-independent weights already exist."
                )
            # Make this the default weight dataset
            self._weight_dset_name = "elevation_vis_weight"
        return super().add_dataset(name)

    @property
    def dirty_beam(self):
        """Not useful at this stage, but it's needed to propagate onward."""
        return self.datasets["dirty_beam"]

    @property
    def effective_ra(self):
        """Get the effective_ra dataset if it exists, None otherwise."""
        if "effective_ra" in self.datasets:
            return self.datasets["effective_ra"]

        raise KeyError("Dataset 'effective_ra' not initialised.")

    @property
    def nsample(self):
        """Get the nsample dataset if it exists, None otherwise."""
        if "nsample" in self.datasets:
            return self.datasets["nsample"]

        raise KeyError("Dataset 'nsample' not initialised.")

    @property
    def pol(self):
        """Get the polarisation index map."""
        return self.index_map["pol"]

    @property
    def ew(self):
        """Get the east-west baseline index map."""
        return self.index_map["ew"]


class HybridVisMModes(FreqContainer, MContainer, VisBase):
    """Visibilities beamformed in the NS direction and m-mode transformed in RA.

    This container has visibilities beam formed only in the NS direction to give a
    grid in elevation.
    """

    _axes = ("pol", "ew", "el")

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["m", "msign", "pol", "freq", "ew", "el"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "vis_weight": {
            "axes": ["m", "msign", "pol", "freq", "ew"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class RingMap(
    FilterFreqContainer, FreqContainer, SiderealContainer, DataWeightContainer
):
    """Container for holding multifrequency ring maps.

    The maps are packed in format `[freq, pol, ra, EW beam, el]` where
    the polarisations are Stokes I, Q, U and V.

    Parameters
    ----------
    nside : int
        The nside of the Healpix maps.
    polarisation : bool, optional
        If `True` all Stokes parameters are stored, if `False` only Stokes I is
        stored.
    """

    _axes = ("pol", "beam", "el")

    _dataset_spec: ClassVar = {
        "map": {
            "axes": ["beam", "pol", "freq", "ra", "el"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 1, 32, 512, 512),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": {
                "weight_dataset": "weight",
            },
        },
        "weight": {
            "axes": ["pol", "freq", "ra", "el"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 512, 512),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "dirty_beam": {
            "axes": ["beam", "pol", "freq", "ra", "el"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 1, 32, 512, 512),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "dirty_beam_power": {
            "axes": ["beam", "pol", "freq", "el"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 1, 512, 512),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "rms": {
            "axes": ["pol", "freq", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 512, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "truncate": True,
        },
        "filter": {
            "axes": ["pol", "freq", "freq_sum", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 32, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_filter": {
            "axes": ["pol", "freq", "freq_sum", "ra"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 32, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "freq_cov": {
            "axes": ["pol", "freq", "freq_sum", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 32, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_freq_cov": {
            "axes": ["pol", "freq", "freq_sum", "ra"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (1, 32, 32, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    _data_dset_name = "map"
    _weight_dset_name = "weight"

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def el(self):
        """Get the el axis."""
        return self.index_map["el"]

    @property
    def map(self):
        """Get the map dataset."""
        return self.datasets["map"]

    @property
    def rms(self):
        """Get the rms dataset."""
        return self.datasets["rms"]

    @property
    def dirty_beam(self):
        """Get the dirty_beam dataset."""
        return self.datasets["dirty_beam"]

    @property
    def dirty_beam_power(self):
        """Get the dirty_beam_power dataset."""
        return self.datasets["dirty_beam_power"]


class RingMapMask(FreqContainer, SiderealContainer):
    """Mask bad ringmap pixels."""

    _axes = ("pol", "el")

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["pol", "freq", "ra", "el"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]


class RingMapTaper(FreqContainer, SiderealContainer):
    """Container for a smooth transition from good to bad ringmap pixels."""

    _axes = ("pol", "el")

    _dataset_spec: ClassVar = {
        "taper": {
            "axes": ["pol", "freq", "ra", "el"],
            "dtype": float,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def taper(self):
        """Get the mask dataset."""
        return self.datasets["taper"]

    @property
    def weight(self):
        """Map weight to taper so that it can easily be updated with masks."""
        return self.datasets["taper"]


class FreqNoiseModel(FilterFreqContainer, FreqContainer, SiderealContainer):
    """Container storing Cholesky factors of frequency-frequency noise covariance.

    This container is intended for generating noise realizations in visibility space
    that include the desired correlations as a function of freq and el.  This is
    typically used to populate a VisGridStream container with synthetic noise having
    a specific spectral structure.
    """

    _axes = ("pol", "ew", "ns")

    _dataset_spec: ClassVar = {
        "redundancy": {
            "axes": ["pol", "ew", "ns"],
            "dtype": np.int32,
            "initialise": True,
            "distributed": False,
            "chunks": (1, 1, 128),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "weight": {
            "axes": ["pol", "freq", "ew", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "chunks": (1, 64, 1, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "freq_cov": {
            "axes": ["pol", "ew", "ra", "freq", "freq_sum"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 1, 2048, 64, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "complex_freq_cov": {
            "axes": ["pol", "ew", "ra", "freq", "freq_sum"],
            "dtype": np.complex128,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 1, 2048, 64, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    @property
    def redundancy(self):
        """Get the redundancy dataset."""
        return self.datasets["redundancy"]

    @property
    def weight(self):
        """Get the weight dataset."""
        return self.datasets["weight"]


class GainDataBase(DataWeightContainer):
    """A container interface for gain-like data.

    To support the previous behaviour of gain type data the weight dataset is optional,
    and returns None if it is not present.
    """

    _data_dset_name = "gain"
    _weight_dset_name = "weight"

    @property
    def gain(self) -> memdata.MemDataset:
        """Get the gain dataset."""
        return self.datasets["gain"]

    @property
    def weight(self) -> memdata.MemDataset | None:
        """The weights for each data point.

        Returns None is no weight dataset exists.
        """
        try:
            return super().weight
        except KeyError:
            return None


class CommonModeGainData(FreqContainer, TODContainer, GainDataBase):
    """Parallel container for holding gain data common to all inputs."""

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "time"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "time"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class CommonModeSiderealGainData(FreqContainer, SiderealContainer, GainDataBase):
    """Parallel container for holding sidereal gain data common to all inputs."""

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "ra"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class GainData(FreqContainer, TODContainer, GainDataBase):
    """Parallel container for holding gain data."""

    _axes = ("input",)

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "input", "time"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "input", "time"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "update_id": {
            "axes": ["time"],
            "dtype": np.dtype("<U64"),
            "initialise": False,
            "distributed": False,
        },
    }

    @property
    def update_id(self):
        """Get the update id dataset if it exists."""
        try:
            return self.datasets["update_id"]
        except KeyError:
            return None

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]


class VisCrosstalkGain(FreqContainer, SiderealContainer):
    """Joint visibility gain and crosstalk estimates."""

    _axes = ("pol", "stack")

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "gain_weight": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "crosstalk": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "crosstalk_weight": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class VisFocalCrosstalkGain(FreqContainer, SiderealContainer):
    """Joint visibility gain, crosstalk and foal expansion estimates."""

    _axes = ("pol", "stack")

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "gain_weight": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "crosstalk": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "crosstalk_weight": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "focalexpansion": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "focalexpansion_weight": {
            "axes": ["freq", "stack", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class VisCrosstalkGainGrid(FreqContainer, SiderealContainer):
    """Joint visibility gain and crosstalk estimates.

    These estimates have been transformed into the visibility grid order.
    """

    _axes = ("pol", "ew", "ns")

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "gain_weight": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "crosstalk": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "crosstalk_weight": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class VisFocalCrosstalkGainGrid(FreqContainer, SiderealContainer):
    """Joint visibility gain, crosstalk and focal expansion estimates.

    These estimates have been transformed into the visibility grid order.
    """

    _axes = ("pol", "ew", "ns")

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "gain_weight": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "crosstalk": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "crosstalk_weight": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "focalexpansion": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "focalexpansion_weight": {
            "axes": ["pol", "freq", "ew", "ns", "ra"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }


class SiderealGainData(FreqContainer, SiderealContainer, GainDataBase):
    """Parallel container for holding sidereal gain data."""

    _axes = ("input",)

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "input", "ra"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "input", "ra"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]


class StaticGainData(FreqContainer, GainDataBase):
    """Parallel container for holding static gain data (i.e. non time varying)."""

    _axes = ("input",)

    _dataset_spec: ClassVar = {
        "gain": {
            "axes": ["freq", "input"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["freq", "input"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def input(self):
        """Get the input axis."""
        return self.index_map["input"]


class DelayCutoff(ContainerPrototype):
    """Container for a delay cutoff."""

    _axes = ("pol", "el")

    _dataset_spec: ClassVar = {
        "cutoff": {
            "axes": ["pol", "el"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
            "distributed_axis": "el",
        }
    }

    @property
    def cutoff(self):
        """Get the cutoff dataset."""
        return self.datasets["cutoff"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]

    @property
    def el(self):
        """Get the el axis."""
        return self.index_map["el"]


class DelayContainer(ContainerPrototype):
    """A container with a delay axis."""

    _axes = ("delay",)

    @property
    def delay(self) -> np.ndarray:
        """The delay axis in microseconds."""
        return self.index_map["delay"]


class DelaySpectrum(DelayContainer):
    """Container for a delay power spectrum.

    Notes
    -----
    A note about definitions: for a dataset with a frequency axis, the corresponding
    delay spectrum is the result of Fourier transforming in frequency, while the delay
    power spectrum is obtained by taking the squared magnitude of each element of the
    delay spectrum, and then usually averaging over some other axis. Our unfortunate
    convention is to store a delay power spectrum in a `DelaySpectrum` container, and
    store a delay spectrum in a :py:class:`~draco.core.containers.DelayTransform`
    container.
    """

    _axes = ("baseline", "sample")

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["baseline", "delay"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
        },
        "spectrum_samples": {
            "axes": ["sample", "baseline", "delay"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
        },
        "spectrum_mask": {
            "axes": ["baseline"],
            "dtype": bool,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
        },
    }

    def __init__(self, *args, weight_boost=1.0, sample=1, **kwargs):
        super().__init__(*args, sample=sample, **kwargs)
        self.attrs["weight_boost"] = weight_boost

    @property
    def spectrum(self):
        """Get the spectrum dataset."""
        return self.datasets["spectrum"]

    @property
    def weight_boost(self):
        """Get the weight boost factor.

        If set, this factor was used to set the assumed noise when computing the
        spectrum.
        """
        return self.attrs["weight_boost"]

    @property
    def freq(self):
        """Get the frequency axis of the input data."""
        return self.attrs["freq"]


class DelayTransform(DelayContainer):
    """Container for a delay spectrum.

    Notes
    -----
    See the docstring for :py:class:`~draco.core.containers.DelaySpectrum` for a
    description of the difference between `DelayTransform` and `DelaySpectrum`.
    """

    _axes = ("baseline", "sample")

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["baseline", "sample", "delay"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (512, 2048, 32),
            "truncate": True,
        },
        "weight": {
            "axes": ["baseline", "sample", "delay"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (512, 2048, 32),
            "truncate": True,
        },
        "spectrum_mask": {
            "axes": ["baseline", "sample"],
            "dtype": bool,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
        },
    }

    def __init__(self, weight_boost=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs["weight_boost"] = weight_boost

    @property
    def spectrum(self):
        """Get the spectrum dataset."""
        return self.datasets["spectrum"]

    @property
    def weight(self):
        """Get the spectrum dataset."""
        return self.datasets["weight"]

    @property
    def weight_boost(self):
        """Get the weight boost factor.

        If set, this factor was used to set the assumed noise when computing the
        spectrum.
        """
        return self.attrs["weight_boost"]

    @property
    def freq(self):
        """Get the frequency axis of the input data."""
        return self.attrs["freq"]


class DelayTransformOperator(DelayContainer, FreqContainer, SiderealContainer):
    """Wiener filter that transforms each pixel from frequency to delay."""

    _axes = ("pol", "el")

    _dataset_spec: ClassVar = {
        "filter": {
            "axes": ["pol", "ra", "el", "delay", "freq"],
            "dtype": np.complex64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "el",
        }
    }

    @property
    def filter(self):
        """Get the filter dataset."""
        return self.datasets["filter"]


class Fourier3DContainer(CosmologyContainer, DelayContainer):
    """A base container with Fourier axes, (pol,delay,u,v)."""

    _axes = ("pol", "u", "v")

    _dataset_spec: ClassVar = {
        "kx": {
            "axes": ["u"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "ky": {
            "axes": ["v"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "kpara": {
            "axes": ["delay"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "uv_mask": {
            "axes": ["u", "v"],
            "dtype": bool,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def kx(self):
        """Get the kx axis."""
        return self.datasets["kx"]

    @property
    def ky(self):
        """Get the ky axis."""
        return self.datasets["ky"]

    @property
    def kpara(self):
        """Get the k_parallel axis."""
        return self.datasets["kpara"]

    @property
    def uv_mask(self):
        """Get the uv-domain mask."""
        return self.datasets["uv_mask"]

    @property
    def redshift(self):
        """Get the redshift attrs."""
        return self.attrs["redshift"]

    @property
    def freq_center(self):
        """Get the central frequency attrs."""
        return self.attrs["freq_center"]


class SpatialDelayCube(Fourier3DContainer):
    """Container for a data in (pol,delays,u,v) domain."""

    _dataset_spec: ClassVar = {
        "vis": {
            "axes": ["pol", "delay", "u", "v"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "delay",
        },
    }

    @property
    def vis(self):
        """Get the spatial data cube."""
        return self.datasets["vis"]


class PowerSpectrum3D(Fourier3DContainer):
    """Container for a 3D power spectrum."""

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["pol", "delay", "u", "v"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "delay",
        }
    }

    @property
    def spectrum(self):
        """Get the 3D power spectrum."""
        return self.datasets["spectrum"]

    @property
    def ps_norm(self):
        """Get the power spectrum normalizaiton attrs."""
        return self.attrs["ps_norm"]


class PowerSpectrum2D(CosmologyContainer):
    """Container for a 2D cylindrically averaged  power spectrum."""

    _axes = ("pol", "delay", "uv_dist")

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["pol", "delay", "uv_dist"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "delay",
        },
        "weight": {
            "axes": ["pol", "delay", "uv_dist"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
        "neff": {
            "axes": ["pol", "delay", "uv_dist"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "delay",
        },
        "mask": {
            "axes": ["pol", "delay", "uv_dist"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
        },
        "kpara": {
            "axes": ["delay"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "kperp": {
            "axes": ["uv_dist"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def spectrum(self):
        """Get the 2D power spectrum dataset."""
        return self.datasets["spectrum"]

    @property
    def weight(self):
        """Get the 2D weight dataset."""
        return self.datasets["weight"]

    @property
    def neff(self):
        """Get the effective number of modes dataset."""
        return self.datasets["neff"]

    @property
    def mask(self):
        """Get the 2D signal window dataset."""
        return self.datasets["mask"]

    @property
    def kpara(self):
        """Get the k_parallel axis."""
        return self.datasets["kpara"]

    @property
    def kperp(self):
        """Get the kprep axis."""
        return self.datasets["kperp"]

    @property
    def delay_cut(self):
        """Get the delay cutoff value."""
        return self.attrs["delay_cut"]


class PowerSpectrum1D(CosmologyContainer):
    """Container for a 1D power spectrum."""

    _axes = ("pol", "k")

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["pol", "k"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
        },
        "samp_var": {
            "axes": ["pol", "k"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
        "var": {
            "axes": ["pol", "k"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
        "neff": {
            "axes": ["pol", "k"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
        "k1D": {
            "axes": ["pol", "k"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
        },
    }

    @property
    def spectrum(self):
        """Get the 1D power spectrum dataset."""
        return self.datasets["spectrum"]

    @property
    def samp_var(self):
        """Get the 1D power spectrum error dataset."""
        return self.datasets["samp_var"]

    @property
    def var(self):
        """Get the 1D power spectrum var dataset."""
        return self.datasets["var"]

    @property
    def neff(self):
        """Get the 1D power spectrum var dataset."""
        return self.datasets["neff"]

    @property
    def k1D(self):
        """Get the k1D dataset."""
        return self.datasets["k1D"]


class WaveletSpectrum(FreqContainer, DelayContainer, DataWeightContainer):
    """Container for a wavelet power spectrum."""

    _axes = ("baseline",)

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["baseline", "delay", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
        },
        "weight": {
            "axes": ["baseline", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
        },
    }
    _data_dset_name = "spectrum"
    _weight_dset_name = "weight"

    @property
    def spectrum(self):
        """The wavelet spectrum."""
        return self.datasets["spectrum"]


class DelayCrossSpectrum(DelaySpectrum):
    """Container for a delay cross power spectra."""

    _axes = ("dataset",)

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["dataset", "dataset", "baseline", "delay"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "baseline",
        },
        "spectrum_samples": {
            "axes": ["sample", "dataset", "dataset", "baseline", "delay"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "baseline",
        },
    }

    @property
    def spectrum(self):
        """Get the spectrum dataset."""
        return self.datasets["spectrum"]


class Powerspectrum2D(ContainerPrototype):
    """Container for a 2D cartesian power spectrum.

    Generally you should set the standard attributes `z_start` and `z_end` with
    the redshift range included in the power spectrum estimate, and the `type`
    attribute with a description of the estimator type. Suggested valued for
    `type` are:

    `unwindowed`
        The standard unbiased quadratic estimator.

    `minimum_variance`
        The minimum variance, but highly correlated, estimator. Just a rescaled
        version of the q-estimator.

    `uncorrelated`
        The uncorrelated estimator using the root of the Fisher matrix.

    Parameters
    ----------
    kpar_edges, kperp_edges : np.ndarray
        Array of the power spectrum bin boundaries.
    """

    _axes = ("kperp", "kpar")

    _dataset_spec: ClassVar = {
        "powerspectrum": {
            "axes": ["kperp", "kpar"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "C_inv": {
            "axes": ["kperp", "kpar", "kperp", "kpar"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    def __init__(self, kperp_edges=None, kpar_edges=None, *args, **kwargs):
        # Construct the kperp axis from the bin edges
        if kperp_edges is not None:
            centre = 0.5 * (kperp_edges[1:] + kperp_edges[:-1])
            width = kperp_edges[1:] - kperp_edges[:-1]

            kwargs["kperp"] = np.rec.fromarrays(
                [centre, width], names=["centre", "width"]
            ).view(np.ndarray)

        # Construct the kpar axis from the bin edges
        if kpar_edges is not None:
            centre = 0.5 * (kpar_edges[1:] + kpar_edges[:-1])
            width = kpar_edges[1:] - kpar_edges[:-1]

            kwargs["kpar"] = np.rec.fromarrays(
                [centre, width], names=["centre", "width"]
            ).view(np.ndarray)

        super().__init__(*args, **kwargs)

    @property
    def powerspectrum(self):
        """Get the powerspectrum dataset."""
        return self.datasets["powerspectrum"]

    @property
    def C_inv(self):
        """Get the C inverse dataset."""
        return self.datasets["C_inv"]


class SVDSpectrum(ContainerPrototype):
    """Container for an m-mode SVD spectrum."""

    _axes = ("m", "singularvalue")

    _dataset_spec: ClassVar = {
        "spectrum": {
            "axes": ["m", "singularvalue"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "m",
        }
    }

    @property
    def spectrum(self):
        """Get the spectrum dataset."""
        return self.datasets["spectrum"]


class FrequencyStack(FreqContainer, DataWeightContainer):
    """Container for a frequency stack.

    In general used to hold the product of `draco.analysis.SourceStack`
    The stacked signal of frequency slices of the data in the direction
    of sources of interest.
    """

    _dataset_spec: ClassVar = {
        "stack": {
            "axes": ["freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    _data_dset_name = "stack"
    _weight_dset_name = "weight"

    @property
    def stack(self):
        """Get the stack dataset."""
        return self.datasets["stack"]


class FrequencyStackByPol(FrequencyStack):
    """Container for a frequency stack split by polarisation."""

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
        "stack": {
            "axes": ["pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]


class MockFrequencyStack(FrequencyStack):
    """Container for holding a frequency stack for multiple mock catalogs.

    Adds a `mock` axis as the first dimension of each dataset.
    """

    _axes = ("mock",)

    _dataset_spec: ClassVar = {
        "stack": {
            "axes": ["mock", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["mock", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }


class MockFrequencyStackByPol(FrequencyStackByPol):
    """Container for holding a frequency stack split by pol for multiple mock catalogs.

    Adds a `mock` axis as the first dimension of each dataset.
    """

    _axes = ("mock",)

    _dataset_spec: ClassVar = {
        "stack": {
            "axes": ["mock", "pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["mock", "pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }


class Stack3D(FreqContainer, DataWeightContainer):
    """Container for a 3D frequency stack."""

    _axes = ("pol", "delta_ra", "delta_dec")

    _dataset_spec: ClassVar = {
        "stack": {
            "axes": ["pol", "delta_ra", "delta_dec", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "weight": {
            "axes": ["pol", "delta_ra", "delta_dec", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    _data_dset_name = "stack"
    _weight_dset_name = "weight"

    @property
    def stack(self):
        """Get the stack dataset."""
        return self.datasets["stack"]


class SourceCatalog(TableSpec):
    """A basic container for holding astronomical source catalogs.

    Notes
    -----
    The `ra` and `dec` coordinates should be ICRS.
    """

    _table_spec: ClassVar = {
        "position": {
            "columns": [["ra", np.float64], ["dec", np.float64]],
            "axis": "object_id",
        }
    }


class SpectroscopicCatalog(SourceCatalog):
    """A container for spectroscopic catalogs."""

    _table_spec: ClassVar = {
        "redshift": {
            "columns": [["z", np.float64], ["z_error", np.float64]],
            "axis": "object_id",
        }
    }


class FormedBeam(FreqContainer, DataWeightContainer):
    """Container for formed beams."""

    _axes = ("object_id", "pol")

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "position": {
            "axes": ["object_id"],
            "dtype": np.dtype([("ra", np.float64), ("dec", np.float64)]),
            "initialise": True,
            "distributed": False,
        },
        "redshift": {
            "axes": ["object_id"],
            "dtype": np.dtype([("z", np.float64), ("z_error", np.float64)]),
            "initialise": False,
            "distributed": False,
        },
    }

    _data_dset_name = "beam"
    _weight_dset_name = "weight"

    @property
    def beam(self):
        """Get the beam dataset."""
        return self.datasets["beam"]

    @property
    def position(self):
        """Get the position dataset."""
        return self.datasets["position"]

    @property
    def redshift(self):
        """Get the redshift dataset."""
        if "redshift" in self.datasets:
            return self.datasets["redshift"]

        raise KeyError("Dataset 'redshift' not initialised.")

    @property
    def frequency(self):
        """Get the frequency axis."""
        return self.index_map["freq"]

    @property
    def id(self):
        """Get the object id axis."""
        return self.index_map["object_id"]

    @property
    def pol(self):
        """Get the pol axis."""
        return self.index_map["pol"]


class FormedBeamHA(FormedBeam):
    """Container for formed beams.

    These have not been collapsed in the hour angle (HA) axis.
    """

    _axes = ("ha",)

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["object_id", "pol", "freq", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (32, 4, 128, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "weight": {
            "axes": ["object_id", "pol", "freq", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (32, 4, 128, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "object_ha": {
            "axes": ["object_id", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def ha(self):
        """Get the hour angle dataset."""
        return self.datasets["object_ha"]


class FormedBeamHAEW(FormedBeamHA):
    """Container for formed beams constructed from a HybridVisStream.

    These have not been collapsed along the hour angle (ha) or
    east west baseline (ew) axis.
    """

    _axes = ("ew",)

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["object_id", "pol", "freq", "ew", "ha"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (8, 4, 128, 4, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "weight": {
            "axes": ["object_id", "pol", "freq", "ew", "ha"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
            "chunks": (8, 4, 128, 4, 64),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "object_ha": {
            "axes": ["object_id", "ha"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def ew(self):
        """Get the ew index map."""
        return self.index_map["ew"]


class FitFormedBeam(FormedBeam):
    """Container for formed beams fit to a primary beam model versus hour angle."""

    _dataset_spec: ClassVar = {
        "background": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight_background": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "corr_background_beam": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def background(self):
        """Get the background dataset."""
        return self.datasets["background"]

    @property
    def weight_background(self):
        """Get the weight_background dataset."""
        return self.datasets["weight_background"]

    @property
    def corr_background_beam(self):
        """Get the corr_background_beam dataset."""
        return self.datasets["corr_background_beam"]


class FitFormedBeamEW(FitFormedBeam):
    """Container for formed beams fit to a primary beam model versus hour angle.

    These have not been collapsed along the east west baseline (ew) axis.
    """

    _axes = ("ew",)

    _dataset_spec: ClassVar = {
        "beam": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "background": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "weight_background": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "corr_background_beam": {
            "axes": ["object_id", "pol", "freq", "ew"],
            "dtype": np.float32,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
    }

    @property
    def ew(self):
        """Get the ew index map."""
        return self.index_map["ew"]


class FormedBeamMask(FreqContainer):
    """Mask bad formed beams."""

    _axes = ("object_id", "pol")

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["object_id", "pol", "freq"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]


class FormedBeamHAMask(FormedBeamMask):
    """Mask bad formed beams as a function of hour angle."""

    _axes = ("ha",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["object_id", "pol", "freq", "ha"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }


def empty_timestream(**kwargs):
    """Create a new timestream container.

    This indirect call exists so it can be replaced to return custom timestream
    types.

    Parameters
    ----------
    kwargs : optional
        Arguments to pass to the timestream constructor.

    Returns
    -------
    ts : TimeStream
    """
    return TimeStream(**kwargs)


class LocalizedRFIMask(FreqContainer, TODContainer):
    """Container for an RFI mask for each freq, el, and time sample.

    The data frac_rfi stores information about the proportion of subdata
    that detected RFI, which is used to generate the mask.
    """

    _axes = ("el",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "el", "time"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "frac_rfi": {
            "axes": ["freq", "el", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 128, 512),
            "truncate": True,
        },
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]

    @property
    def frac_rfi(self):
        """Get the frac_rfi dataset."""
        return self.datasets["frac_rfi"]

    @property
    def el(self):
        """Get the el axis."""
        return self.index_map["el"]


class LocalizedSiderealRFIMask(FreqContainer, SiderealContainer):
    """Container for an RFI mask for each freq, ra, and el sample.

    The data frac_rfi stores information about the proportion of subdata
    that detected RFI, which is used to generate the mask.
    """

    _axes = ("el",)

    _dataset_spec: ClassVar = {
        "mask": {
            "axes": ["freq", "ra", "el"],
            "dtype": bool,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "frac_rfi": {
            "axes": ["freq", "ra", "el"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (64, 512, 128),
            "truncate": True,
        },
    }

    @property
    def mask(self):
        """Get the mask dataset."""
        return self.datasets["mask"]

    @property
    def frac_rfi(self):
        """Get the frac_rfi dataset."""
        return self.datasets["frac_rfi"]

    @property
    def el(self):
        """Get the el axis."""
        return self.index_map["el"]


class VisBandpassWindow(FreqContainer):
    """Container for bandpass gains and their window estimated by running bandpass HyFoReS on hybrid beam-formed visibilities."""

    _axes = ("pol",)

    # TODO: check if np.complex128 is required
    _dataset_spec: ClassVar = {
        "bandpass": {
            "axes": ["pol", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
        "window": {
            "axes": ["pol", "freq", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def bandpass(self):
        """Get the bandpass dataset."""
        return self.datasets["bandpass"]

    @property
    def window(self):
        """Get the window dataset."""
        return self.datasets["window"]


class VisBandpassCompensate(FreqContainer):
    """Container for window-compensated bandpass gains."""

    _axes = ("pol",)

    # TODO: redefine the second axis for sval
    _dataset_spec: ClassVar = {
        "comp_bandpass": {
            "axes": ["pol", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
        "sval": {
            "axes": ["pol", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def comp_bandpass(self):
        """Get the comp_bandpass dataset."""
        return self.datasets["comp_bandpass"]

    @property
    def sval(self):
        """Get the sval dataset."""
        return self.datasets["sval"]


class VisBandpassWindowBaseline(VisBandpassWindow):
    """Container for bandpass gains and their window estimated by running bandpass HyFoReS on hybrid beam-formed visibilities."""

    _axes = ("ew",)

    _dataset_spec: ClassVar = {
        "bandpass": {
            "axes": ["pol", "ew", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
        "window": {
            "axes": ["pol", "ew", "freq", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def bandpass(self):
        """Get the bandpass dataset."""
        return self.datasets["bandpass"]

    @property
    def window(self):
        """Get the window dataset."""
        return self.datasets["window"]


class VisBandpassCompensateBaseline(VisBandpassCompensate):
    """Container for window-compensated bandpass gains."""

    _axes = ("ew",)

    _dataset_spec: ClassVar = {
        "comp_bandpass": {
            "axes": ["pol", "ew", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
        "sval": {
            "axes": ["pol", "ew", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def comp_bandpass(self):
        """Get the comp_bandpass dataset."""
        return self.datasets["comp_bandpass"]

    @property
    def sval(self):
        """Get the sval dataset."""
        return self.datasets["sval"]


class VisBandpassWindowBaselineRA(SiderealContainer, VisBandpassWindowBaseline):
    """Container for bandpass gains and their window estimated by running bandpass HyFoReS on hybrid beam-formed visibilities."""

    _dataset_spec: ClassVar = {
        "bandpass": {
            "axes": ["pol", "ew", "ra", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 4, 2048, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "window": {
            "axes": ["pol", "ew", "ra", "freq", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 4, 2048, 32, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    @property
    def bandpass(self):
        """Get the bandpass dataset."""
        return self.datasets["bandpass"]

    @property
    def window(self):
        """Get the window dataset."""
        return self.datasets["window"]


class VisBandpassCompensateBaselineRA(SiderealContainer, VisBandpassCompensateBaseline):
    """Container for window-compensated bandpass gains."""

    _dataset_spec: ClassVar = {
        "comp_bandpass": {
            "axes": ["pol", "ew", "ra", "freq"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 4, 2048, 32),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
        "rank": {
            "axes": ["pol", "ew", "ra"],
            "dtype": np.complex128,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ra",
            "chunks": (1, 4, 2048),
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
        },
    }

    @property
    def comp_bandpass(self):
        """Get the comp_bandpass dataset."""
        return self.datasets["comp_bandpass"]

    @property
    def rank(self):
        """Get the sval dataset."""
        return self.datasets["rank"]


class HorizonLimit(ContainerPrototype):
    """Container holding the altitude of the horizon as a function of azimuth."""

    _axes = ("azimuth",)

    _dataset_spec: ClassVar = {
        "altitude": {
            "axes": ["azimuth"],
            "dtype": float,
            "initialise": True,
            "distributed": False,
        }
    }

    def get_horizon_limit(self, az):
        """Interpolate the horizon altitude at a given azimuth.

        Parameters
        ----------
        az : float or ndarray
            Azimuth angle in degrees at which to evaluate the horizon limit.

        Returns
        -------
        alt : float or ndarray
            Interpolated horizon altitude(s) in degrees.
        """
        return np.interp(az, self.azimuth, self.altitude, period=360.0)

    @property
    def azimuth(self):
        """Get the index map containing the azimuth angle (in degrees)."""
        return self.index_map["azimuth"]

    @property
    def altitude(self):
        """Get the dataset containing the altitude (in degrees)."""
        return self.datasets["altitude"]
