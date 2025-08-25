"""Helper functions for the delay PS estimation via ML/MAP optimisation."""

from typing import Protocol

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from ..util import kernels, tools


class OptFunc(Protocol):
    """A protocol for a function and its gradients to optimise.

    The function should be defined such that it can be minimised.
    """

    def value(self, x: np.ndarray) -> float:
        """Compute the value of the function.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        The value of the function being optimized.
        """
        raise NotImplementedError()

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the function with respect to the optimization parameters.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        The gradient of the function being optimized.
        """
        raise NotImplementedError()

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Compute the hessian (matrix of second derivatives) of the function with respect to the optimization parameters.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        The hessian of the function being optimized.
        """
        raise NotImplementedError()


class LogLikePS(OptFunc):
    """Compute the likelihood, gradient, and hessian for delay PS estimation.

    This class efficiently computes the *negative* likelihood, as well as its gradient
    and hessian. It will precompute and cache relevant quantities such that all of these
    can be calculated per iteration without recomputing. It is designed to be used with
    `scipy.optimize.minimize`.

    The parameters used for this are the log of the delay power spectrum samples.

    Parameters
    ----------
    X
        The covariance matrix of the data.
    MF
        The masked Fourier matrix which maps from delay space to frequency space and
        applies zero to any masked channels.
    N
        The noise covariance matrix.
    nsamp
        The number of samples used to calculate the covariance.
    fsel
        An optional array selection to apply to limit the frequencies used in the
        estimation. If not supplied, any frequencies entirely masked within `MF` are
        skipped.
    exact_hessian
        If set, use the exact Hessian for the calculation. Otherwise use the Fisher
        matrix in the same way as the original NRML methods.
    bounds
        Bounds on the minimisation parameters. Default is (1e-10, 1e10).
    """

    def __init__(
        self,
        X: np.ndarray,
        MF: np.ndarray,
        N: np.ndarray,
        nsamp: int,
        fsel: np.ndarray | slice | list | None = None,
        exact_hessian: bool = True,
        bounds: tuple = (1e-10, 1e10),
    ) -> None:
        if fsel is None:
            fsel = (MF != 0).any(axis=1)

        self.X = X[fsel][:, fsel]
        self.N = N[fsel]
        self.MF = MF[fsel]
        # Pre-compute the conjugate transpose since
        # it doesn't change during the minimization
        self.MFT = self.MF.T.conj()

        self.nsamp = nsamp
        self.exact_hessian = exact_hessian
        self._logbounds = tuple(sorted(np.log(x) for x in bounds))

    # Store the location we are calculating for.
    _s_a: np.ndarray | None = None

    def _precompute(self, x: np.ndarray) -> bool:
        """Pre-compute useful matrices for the given value.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        True if a pre-computation was done, otherwise False.
        """
        if np.array_equal(x, self._s_a):
            return False

        # Enforce bounds on the parameters. Do this
        # on the log values to avoid huge exponentials.
        self._s_a = np.clip(x, *self._logbounds)

        S = np.exp(self._s_a)
        dS = S

        # Compute the covariance and inverse covariance
        self._C = (self.MF * S[np.newaxis, :]) @ self.MFT
        np.einsum("ii->i", self._C)[:] += self.N
        # Get the Cholesky decomposition of the covariance
        # matrix. This is both faster for solving inverse
        # problems and provides a fast way to calculate the
        # log of the determinant
        self._Ch = la.cho_factor(self._C, check_finite=False)

        # Compute matrices used for the gradient and hessian
        self._XC = self.X - self._C

        self._U = dS[np.newaxis, :] ** 0.5 * self.MF
        self._Ut = la.cho_solve(self._Ch, self._U, check_finite=False)

        self._XC_Ut = self._XC @ self._Ut
        # These are used in the hessian, and are just
        # re-assigned for notation purposes
        self._W = self._U
        self._Wt = self._Ut
        self._XC_Wt = self._XC_Ut

        return True

    def value(self, x: np.ndarray) -> float:
        """Calculate the negative log-likelihood.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Value of negative log-likelihood for the given set of params.
        """
        self._precompute(x)

        # The log of the determinant of `C` is equal to
        # 2 times the sum of the log of the diagonal of the
        # Cholesky of C, which is much faster to compute
        lndet = 2 * np.log(np.einsum("ii->i", self._Ch[0])).real.sum()

        # Add the trace of XC^{-1}
        CiX = la.cho_solve(self._Ch, self.X, check_finite=False)
        lndet += np.einsum("ii->i", CiX).real.sum()

        return self.nsamp * lndet

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the negative log-likelihood.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Gradient of negative log-likelihood for the given set of params.
        """
        self._precompute(x)

        g = -(self._Ut.conj() * self._XC_Ut).real.sum(axis=0)

        return self.nsamp * g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the Hessian of the negative log-likelihood.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Hessian of negative log-likelihood for the given set of params.
        """
        self._precompute(x)

        Ua_Utb = self._U.T.conj() @ self._Ut
        Fab = Ua_Utb * Ua_Utb.T.conj()
        H = Fab.real

        if self.exact_hessian:
            Uta_dX_Utb = self._Ut.T.conj() @ self._XC_Ut
            H += (2 * Uta_dX_Utb * Ua_Utb.T).real
            # Add the diagonal element, which is just the gradient.
            # This is extremely fast to compute, so don't bother
            # trying to optimise with the call to `gradient`.
            t = -(self._Wt.conj() * self._XC_Wt).real.sum(axis=0)
            np.einsum("ii->i", H)[:] += t

        return self.nsamp * H


class GaussianProcessPrior(OptFunc):
    """A Gaussian process prior on the inputs.

    The kernel has a standard deviation of `width`, and the
    variance is `alpha**2`.

    Parameters
    ----------
    N : int
        The number of points in the function we are inferring. The
        sample locations are implicitly assumed to be at `np.arange(N)`.
    width : int
        The width of the kernel used in the covariance. Default is 5.
    alpha : float
        The scale of the distribution. Default is 1.0.
    kernel : str
        The name of the kernel to use. Supported kernels are
        'gaussian', 'rational', and 'matern'. Default is `gaussian`.
    reg : float
        Add a small diagonal entry of size `alpha**2 * reg`
        for numerical stability. Default is 1e-8.
    kernel_params : dict
        Additional parameters depending on the kernel being used.
        See `draco.util.kernels` for kernel details.
    """

    def __init__(
        self,
        N: int,
        *,
        width: int = 5,
        alpha: float = 1,
        kernel: str = "gaussian",
        reg: float = 1e-8,
        **kernel_params,
    ) -> None:
        # Get the covariance kernel. Alpha needs to be applied _after_
        # inverting with the regularisation, so just set it to 1.0 here.
        # Strictly do not support band-diagonal kernels for now.
        kernel_params.update({"N": int(N), "width": int(width), "alpha": 1.0})

        C = kernels.get_kernel(kernel, **kernel_params)

        if kernel == "moving_average":
            self.Ci = alpha * C
        else:
            self.Ci = la.inv(C + np.identity(N) * reg) / alpha**2

    # Store the location we are calculating for.
    _s_a: np.ndarray | None = None

    def _precompute(self, x: np.ndarray) -> bool:
        """Pre-compute matrices for the given value.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        True if a pre-computation was done, otherwise False.
        """
        if np.array_equal(x, self._s_a):
            return False

        self._s_a = x

        self._Cix = self.Ci @ x

        return True

    def value(self, x: np.ndarray) -> float:
        """Calculate the value of the gaussian process prior.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Value of the gaussian process prior for the given set of params.
        """
        self._precompute(x)

        return 0.5 * float(x @ self._Cix)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the gaussian process prior.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Gradient of the gaussian process prior for the given set of params.
        """
        self._precompute(x)

        return self._Cix

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the hessian of the gaussian process prior.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Hessian of the gaussian process prior for the given set of params.
        """
        return self.Ci


class AddFunctions(OptFunc):
    """Optimise the sum of several functions.

    The values, gradients and hessians of each function are added together.

    Parameters
    ----------
    functions
        A list of functions to optimise. The individual functions must all
        take the same set of parameters.
    """

    def __init__(self, functions: list[OptFunc]) -> None:
        if len(functions) <= 0:
            raise ValueError("At least one function must be supplied.")
        self.functions = functions

    def value(self, x: np.ndarray) -> float:
        """Calculate the sum of the individual function values.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Sum of individual function values for the given set of params.
        """
        return sum(f.value(x) for f in self.functions)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate the sum of the individual function gradients.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Sum of individual function gradients for the given set of params.
        """
        g = self.functions[0].gradient(x)
        for f in self.functions[1:]:
            g += f.gradient(x)
        return g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Calculate the sum of the individual function hessians.

        Parameters
        ----------
        x
            The array of parameters in the optimization.

        Returns
        -------
        Sum of individual function hessians for the given set of params.
        """
        h = self.functions[0].hessian(x)
        for f in self.functions[1:]:
            h += f.hessian(x)
        return h


def delay_power_spectrum_maxpost(
    data,
    N,
    Ni,
    initial_S: np.ndarray | None = None,
    window: str = "nuttall",
    fsel: np.ndarray | None = None,
    maxiter: int = 100,
    tol: float = 1e-3,
    bounds: tuple = (1e-15, 1e10),
):
    """Estimate the delay power spectrum with a maximum-likelihood estimator.

    This routine uses `scipy.optimize.minimize` to find the maximum likelihood power
    spectrum.

    64-bit precision is required for the data covariance and fourier matrices.

    Parameters
    ----------
    data : np.ndarray[:, freq]
        Data to estimate the delay spectrum of.
    N : int
        The length of the output delay spectrum. There are assumed to be `N/2 + 1`
        total frequency channels if assuming a real delay spectrum, or `N` channels
        for a complex delay spectrum.
    Ni : np.ndarray[freq]
        Inverse noise variance.
    initial_S : np.ndarray[delay]
        The initial delay power spectrum guess.
    window : one of {'nuttall', 'blackman_nuttall', 'blackman_harris', None}, optional
        Apply an apodisation function. Default: 'nuttall'.
    fsel : np.ndarray[freq], optional
        Indices of channels that we have data at. By default assume all channels.
    maxiter : int, optional
        Maximum number of iterations to run of the solver. Default is 100.
    tol : float, optional
        The convergence tolerance for the optimization that is passed to scipy.optimize.minimize.
        Default is 1e-3.
    bounds : tuple, optional
        Bounds on the minimisation paramaters. Default is (1e-15, 1e10).

    Returns
    -------
    samples : list
        List of spectrum samples.
    success : bool
        True if the solver successfully converged, False otherwise.
    """
    # This import can't be at the top level or else
    # we end up with a circular import
    from .delay import fourier_matrix

    nsamp, Nf = data.shape

    if fsel is None:
        fsel = np.arange(Nf)
    elif len(fsel) != Nf:
        raise ValueError(
            "Length of frequency selection must match frequencies passed. "
            f"{len(fsel)} != {data.shape[-1]}"
        )

    # Construct the Fourier matrix. 64-bit precision
    # is required for numerically stable results.
    F = fourier_matrix(N, fsel).astype(np.complex128, copy=False)
    data = data.astype(F.dtype, copy=True)

    # Compute the covariance matrix of the data
    # Window the frequency data if requested
    if window is not None:
        # Construct the window function
        w = tools.window_generalised(fsel / N, window=window)
        # Apply to data and projection matrix
        F *= w[:, np.newaxis]
        data *= w[np.newaxis, :]

    # Make the data covariance matrix
    X = (data.T @ data.conj()) / nsamp

    # Make the noise matrix from the diagonal of its inverse.
    # Assume that this is also diagonal
    Nm = tools.invert_no_zero(Ni)

    # Mask out any completely missing frequencies
    F[Ni == 0] = 0.0

    # Use the pseudo-inverse to give a starting point for the optimiser
    if initial_S is None:
        initial_S = (data @ la.pinv(F.T, rtol=1e-3)).var(axis=0)

    # Create a list to store intermediate samples
    # during the minimization routine. The first item
    # will always be the initial guess
    samples = [initial_S]

    # Construct the optimisation function as the sum of the
    # PS likelihood and a GP prior. The matern kernel is used
    # as a prior to reduce over-smoothing.
    optfunc = AddFunctions(
        [
            LogLikePS(X, F, Nm, nsamp, exact_hessian=True, bounds=bounds),
            GaussianProcessPrior(N, width=5, alpha=1.0, kernel="matern", nu=1.5),
        ]
    )

    try:
        # Minimize over the log of the delay PS. Each PS sample
        # (note the `exp`) is stored via the callback function.
        res = minimize(
            optfunc.value,
            x0=np.log(initial_S),
            jac=optfunc.gradient,
            hess=optfunc.hessian,
            method="Newton-CG",
            options={"maxiter": maxiter, "xtol": tol},
            callback=lambda xk: samples.append(np.exp(xk)),
        )
        success = res.success

    # LinAlgError gets thrown for certain baselines in _precompute during a Cholesky decomposition
    # of the covariance matrix (used in likelihood computation) when the covariance matrix isn't
    # positive definite. This appears to happen after one of the minimization parameters blows up,
    # likely from a numerical instability somewhere in the gradient/hessian computation. This has
    # only been observed to affect a small number of (almost entirely masked) baselines that have
    # very low weights.
    # ValueError also occasionally gets thrown from (almost certainly) the same numerical
    # instability: one of the minimization parameters blows up and causes an overflow when taking
    # np.exp() of the minimization parameter to get the delay spectrum. The ValueError gets thrown
    # again in the Cholesky decomposition when numpy runs a check_finite on the covariance (which
    # at that point contains infs/nans).
    # NOTE: I think that this is fixed now, but I'm leaving these checks and this note here for the
    # time being.
    except (la.LinAlgError, ValueError):
        success = False

    # NOTE: the final sample in samples is already the final result
    return samples, success
