"""Tasks to do data interpolation/inpainting."""

import numpy as np
from caput import config, mpiutil
from cora.util import units

from ..core import io, task
from ..util import dpss


class DPSSInpaint(task.SingleTask):
    """Fill data gaps using DPSS inpainting.

    Discrete prolate spheroidal sequence (DPSS) inpainting involves
    projecting a partially-masked data series onto a basis which
    maximally concentrates spectral power within a defined window.
    This basis, called the nth order discrete prolate spheroidal
    sequence or Slepian sequence consists of the large eigenvectors
    of a covariance matrix defined as a sum of `sinc` functions,
    which represent top-hats in the spectral inverse of the data.

    This class is fully-functional, but only supports applying a
    constant cutoff.

    Attributes
    ----------
    axis : str
        Name of the axis over which to inpaint. Only one-dimensional
        inpainting is currently supported. Must be either "freq" or
        "ra". Default is `freq`.
    iter_axes : list[str]
        List of independent axes over which to iterate. This can
        include axes not in the dataset, but at least one of these
        axes should be present. Default is ["stack", "el"].
    centres : list
        List of top-hat window centres. If all windows are centred
        about zero, the covariance matrix will be real, which provides
        significant performance improvements.
    halfwidths : list
        List of window half-widths. Must be the same length as `centres`.
    snr_cov : float
        Wiener filter inverse signal covariance. Default is 1.0e-3.
    flag_above_cutoff : bool
        Re-flag gaps in the data above the width specified by
        cutoff_frac * fs / max(halfwidths), where fs is the
        sample rate. Default is True.
    cutoff_frac : float
        Fraction of the cutoff used when re-flagging inpainted
        samples. Default is 1.0.
    copy : bool
        If true, copy the container instead of inpainting in-place.
    """

    axis = config.enum(["freq", "ra"], default="freq")
    iter_axes = config.Property(proptype=list, default=["stack", "el"])
    centres = config.Property(proptype=list)
    halfwidths = config.Property(proptype=list)
    snr_cov = config.Property(proptype=float, default=1.0e-3)
    flag_above_cutoff = config.Property(proptype=bool, default=True)
    cutoff_frac = config.Property(proptype=float, default=1.0)
    copy = config.Property(proptype=bool, default=True)

    def setup(self, mask=None):
        """Use an optional mask dataset.

        Parameters
        ----------
        mask : containers.RFIMask, optional
            Container used to select samples to inpaint. If
            not provided, inpaint samples where the data
            weights are zero.
        """
        self.mask = mask

    def process(self, data):
        """Inpaint visibility data.

        Parameters
        ----------
        data : containers.VisContainer
            Container with a visibility dataset

        Returns
        -------
        data : containers.VisContainer
            Input container with masked values filled
        """
        try:
            # Get the axis samples
            samples = getattr(data, self.axis)
        except AttributeError as exc:
            raise ValueError(f"Could not get axis `{self.axis}`.") from exc

        # Redistribute over an independent axis
        data.redistribute(self.iter_axes)
        # Set the local selection over the distributed axis
        self._set_sel(data)

        vinp, winp = self.inpaint(data.vis, data.weight, samples)

        # Make the output container
        if self.copy:
            # Trying to copy a dataset while distributed
            # over a non-contiguous axis (time/ra in this case)
            # hangs almost indefinitely. We need to distribute
            # over a different axis before copying.
            data.redistribute("freq")
            out = data.copy()
            out.redistribute(self.iter_axes)
        else:
            out = data

        out.vis[:].local_array[:] = vinp
        out.weight[:].local_array[:] = winp

        return out

    def inpaint(self, vis, weight, samples):
        """Inpaint visibilities using a wiener filter.

        Use a single sequence for the entire dataset.
        """
        # Move the iteration and interpolation axes
        # to the front and flatten the other axes
        vobs, vaxind = _flatten_axes(vis, (*self.iter_axes, self.axis))
        wobs, waxind = _flatten_axes(weight, (*self.iter_axes, self.axis))

        if self.mask is not None:
            mobs, _ = _flatten_axes(self.mask.mask, (*self.iter_axes, self.axis))
            # Invert the mask to avoid doing it every loop
            mobs = ~mobs

        # Pre-allocate the full output array
        vinp = np.zeros_like(vobs)
        winp = np.zeros_like(wobs)

        # Construct the covariance matrix and get dpss modes
        modes, amap, cutoff = self._get_basis(samples)

        # Iterate over the variable axis
        for ii in range(vobs.shape[0]):
            # Get the correct basis for each slice
            A = modes[amap[ii]]

            # Get a selection for data to keep
            M = wobs[ii] > 0
            W = mobs if self.mask is not None else M

            vinp[ii], winp[ii] = dpss.inpaint(vobs[ii], wobs[ii], A, W, self.snr_cov)

            # Re-flag gaps above the cutoff width
            if self.flag_above_cutoff:
                winp[ii] *= dpss.flag_above_cutoff(M, cutoff)

        # Reshape and move the interpolation axis back
        vinp = _inv_move_front(vinp, vaxind, vis.local_shape)
        winp = _inv_move_front(winp, waxind, weight.local_shape)

        return vinp, winp

    def _set_sel(self, data):
        """Extract selection along local axis."""
        self._local_sel = data.vis[:].local_bounds

    def _get_basis(self, samples):
        """Make the DPSS basis.

        Returns a list of bases and a map.
        """
        # Construct the covariance matrix and get dpss modes
        cov = dpss.make_covariance(samples, self.halfwidths, self.centres)
        modes = dpss.get_basis(cov)
        # All iterations map to the same basis
        amap = [0] * (self._local_sel.stop - self._local_sel.start)

        # Flagging cutoff
        fs = 1 / np.median(abs(np.diff(samples)))
        cutoff = self.cutoff_frac * fs / np.max(self.halfwidths)

        return [modes], amap, cutoff


class DPSSInpaintBaseline(DPSSInpaint):
    """Inpaint with baseline-dependent cut.

    This is a non-functional base class which provides functionality
    for selecting the correct baselines and making a set of unique
    basis functions.

    Users should override the `_get_cuts` method to make baseline-
    dependent cuts along the desired axis.

    Attributes
    ----------
    telescope_orientation : one of ('NS', 'EW', 'none')
        Determines if the baseline-dependent delay cut is based on the north-south
        component, the east-west component or the full baseline length. For
        cylindrical telescopes oriented in the NS direction (like CHIME) use 'NS'.
        The default is 'NS'.
    """

    telescope_orientation = config.enum(["NS", "EW", "none"], default="NS")

    def setup(self, telescope, mask=None):
        """Load a telescope object.

        Parameters
        ----------
        telescope : TransitTelescope
            Telescope object with baseline information.
        mask : containers.RFIMask, optional
            Container used to select samples to inpaint. If
            not provided, inpaint samples where the data
            weights are zero.
        """
        self.telescope = io.get_telescope(telescope)
        # Pass the mask to the parent class
        super().setup(mask)

    def _set_sel(self, data):
        """Set the local baselines."""
        prod = data.prodstack
        sel = self.telescope.feedmap[(prod["input_a"], prod["input_b"])]

        self._baselines = self.telescope.baselines[sel]

    def _get_basis(self, samples):
        """Make the DPSS basis for each unique delay cut.

        Returns a list of bases and a map.
        """
        # Get cutoffs for each baseline
        cuts = self._get_baseline_cuts()

        # Compute covariances for each unique baseline and
        # map to each individual baseline.
        cuts, amap = np.unique(cuts, return_inverse=True)

        modes = []

        for ii, cut in enumerate(cuts):
            self.log.debug(
                f"Making unique covariance {ii+1}/{len(cuts)} with cut={cut}."
            )
            cov = dpss.make_covariance(samples, cut, 0.0)
            modes.append(dpss.get_basis(cov))

        # Flagging cutoff
        fs = 1 / np.median(abs(np.diff(samples)))
        cutoff = self.cutoff_frac * fs / np.max(cuts)
        # Need the mask to be the same for all baselines,
        # so use the most aggressive masking
        cutoff = mpiutil.allreduce(cutoff, op=mpiutil.MIN)

        return modes, amap, cutoff

    def _get_baseline_cuts(self):
        """Get an array of cutoffs for each baseline."""
        raise NotImplementedError()


class DPSSInpaintDelay(DPSSInpaintBaseline):
    """Inpaint with baseline-dependent delay cut.

    Attributes
    ----------
    axis : str
        Name of axis over which to inpaint. `freq` is the only
        accepted argument.
    za_cut : float
        Sine of the maximum zenith angle included in baseline-dependent delay
        filtering. Default is 1 which corresponds to the horizon (ie: filters out all
        zenith angles). Setting to zero turns off baseline dependent cut.
    extra_cut : float
        Increase the delay threshold beyond the baseline dependent term.
    telescope_orientation : one of ('NS', 'EW', 'none')
        Determines if the baseline-dependent delay cut is based on the north-south
        component, the east-west component or the full baseline length. For
        cylindrical telescopes oriented in the NS direction (like CHIME) use 'NS'.
        The default is 'NS'.
    """

    axis = config.enum(["freq"], default="freq")
    za_cut = config.Property(proptype=float, default=1.0)
    extra_cut = config.Property(proptype=float, default=0.0)

    def _get_baseline_cuts(self):
        """Get an array of delay cuts."""
        # Calculate delay cuts based on telescope orientation
        if self.telescope_orientation == "NS":
            blen = abs(self._baselines[:, 1])
        elif self.telescope_orientation == "EW":
            blen = abs(self._baselines[:, 0])
        else:
            blen = np.linalg.norm(self._baselines, axis=1)

        # Get the delay cut for each baseline. Round delay cuts
        # to three decimal places to reduce repeat calculations
        delay_cut = self.za_cut * blen / units.c * 1.0e6 + self.extra_cut
        delay_cut = np.maximum(delay_cut, self.halfwidths[0])

        return np.round(delay_cut, decimals=3)


class DPSSInpaintMMode(DPSSInpaintBaseline):
    """Inpaint with a baseline-dependent m cut.

    Attributes
    ----------
    axis : str
        Name of axis over which to inpaint. `freq` is the only
        accepted argument.
    """

    axis = config.enum(["ra"], default="ra")

    def _get_baseline_cuts(self):
        """Make the DPSS basis for each unique m cut.

        Returns a list of bases and a map.
        """
        # Calculate cuts based on telescope orientation.
        # Note that this is opposite from the baseline
        # component used for delay, since we care
        # about the direction of fringing here
        if self.telescope_orientation == "NS":
            blen = abs(self._baselines[:, 0])
        elif self.telescope_orientation == "EW":
            blen = abs(self._baselines[:, 1])
        else:
            blen = np.linalg.norm(self._baselines, axis=1)

        # Get highest frequency in MHz
        freq = self.telescope.freq_start
        dec = np.deg2rad(self.telescope.latitude)
        # Cut at the maximum `m` expected for each baseline.
        # Compensate for the fact the ra samples is in degrees
        mcut = (np.pi / 180) * freq * 1e6 * blen / (units.c * np.cos(dec))
        mcut = np.maximum(mcut, self.halfwidths[0])

        return np.round(mcut, decimals=2)


class StokesIMixin:
    """Change baseline selection assuming Stokes I only."""

    def _set_sel(self, data):
        """Set the local baselines."""
        # Baseline lengths extracted from the stack axis
        self._baselines = data.stack[data.vis[:].local_bounds]


class DPSSInpaintDelayStokesI(StokesIMixin, DPSSInpaintDelay):
    """Inpaint Stokes I with baseline-dependent delay cut."""


class DPSSInpaintMModeStokesI(StokesIMixin, DPSSInpaintMMode):
    """Inpaint Stokes I with baseline-dependent m-mode cut."""


def _flatten_axes(data, axes):
    """Move the specified axes to the front of a dataset.

    Not all the axes in `axes` need to be present, but at
    least one must exist
    """
    dax = list(data.attrs["axis"])

    axind = [dax.index(axis) for axis in axes if axis in dax]

    if not axind:
        raise ValueError(
            f"No matching axes. Dataset has axes {dax}, "
            f"but axes {axes} were requested."
        )

    ds = data[:].view(np.ndarray)

    return _move_front(ds, axind, ds.shape), axind


def _move_front(arr: np.ndarray, axis: int | list, shape: tuple) -> np.ndarray:
    """Move specified axes to the front and flatten remaining axes."""
    if np.isscalar(axis):
        axis = [axis]

    new_shape = [shape[i] for i in axis]
    # Move the N specified axes to the first N positions
    inds = list(range(len(axis)))
    # Move the specified axes to the front and flatten
    # the remaining axes
    arr = np.moveaxis(arr, axis, inds)

    return arr.reshape(*new_shape, -1)


def _inv_move_front(arr: np.ndarray, axis: int | list, shape: tuple) -> np.ndarray:
    """Move axes back to their original position and expand."""
    if np.isscalar(axis):
        axis = [axis]

    new_shape = [shape[i] for i in axis]
    new_shape += [sh for sh in shape if sh not in new_shape]
    inds = list(range(len(axis)))

    # Undo the flattening process
    arr = arr.reshape(new_shape)
    # Move axes back to their original positions
    arr = np.moveaxis(arr, inds, axis)

    return arr.reshape(shape)
