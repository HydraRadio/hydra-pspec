import numpy as np
import scipy as sp
from scipy.stats import mode
from scipy.signal.windows import blackmanharris as BH
from scipy.stats import invgamma
from scipy.optimize import minimize, Bounds

from multiprocess import Pool
from . import utils
import os, time


def sample_S(s=None, sk=None, prior=None):
    """
    Draw samples of the bandpowers of S, p(S|s). This assumes that the conditional
    distributions for the bandpowers are uncorrelated with one another, i.e. the Fourier-
    space covariance S has no off-diagonals.

    Parameters:
        s (array_like):
            A set of real-space samples of the field, of shape
            `(Ntimes, Nfreq)`. This will be Fourier transformed.
            Alternatively, `sk` can be given.

        sk (array_like):
            A set of Fourier-space samples of the field, of shape
            `(Ntimes, Nfreq)`.

        prior (array_like):
            Array of delta function prior values, used to set certain modes to a
            fixed value.
    """
    if s is None and sk is None:
        raise ValueError("Must pass in s (real space) or sk (Fourier space) vector.")

    if sk is None:
        sk = np.fft.fft(s, axis=-1)
    Nobs, Nfreqs = sk.shape

    beta = np.sum(sk * sk.conj(), axis=0).real
    alpha = Nobs / 2.0 - 1.0

    x = np.zeros(Nfreqs)
    for i in range(Nfreqs):
        x[i] = invgamma.rvs(a=alpha) * beta[i]  # y = x / beta

    # Set prior
    if prior is not None:
        for i in range(Nfreqs):
            if prior[0, i] == 0:
                continue
            else:
                if x[i] > prior[0, i]:
                    x[i] = prior[0, i]
                if x[i] < prior[1, i]:
                    x[i] = prior[1, i]
    return x


def sprior(signals, bins, factor):

    # prior on cov samples

    # bins - number of bins past zero delay to take, either side. e.g. bins=2 takes delays [-2,-1,0,1,2] from centre
    # factor is maximum factor to multiply / divide the truth by
    Nobs, Nfreq = signals.shape

    sk_ = np.fft.fft(signals, axis=-1)
    ds = np.sum(sk_ * sk_.conj(), axis=0).real
    prior = np.zeros((2, Nfreq))

    prior[0] = ds * factor
    prior[1] = ds / factor

    prior[0, bins + 1 : -bins] = 0
    prior[1, bins + 1 : -bins] = 0

    return prior / (Nobs / 2 - 1)


def gcr_fgmodes_1d(vis, w, matrices, fgmodes, f0=None):
    """
    Perform the GCR step on a single time sample.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        w (array_like):
            Array of flags or weights (e.g. 1 for unflagged, 0 for flagged).
        matrices (array_like):
            Array containing precomputed matrices needed by the linear system.
        fgmodes (array_like):
            Foreground mode array, of shape (Nmodes, Nfreqs). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        nproc (int):

    """
    Nfreqs, Nmodes = fgmodes.shape
    d = vis.reshape((1, max(Nfreqs, len(vis.T))))

    # Extract precomputed matrices needed by the linear system
    Sh = matrices[0][0]
    Si = matrices[0][1]
    Ni = matrices[0][2]
    Sih = matrices[0][3]
    Nih = matrices[0][4]
    A = matrices[1][0]
    Ai = matrices[1][1]

    # Unit complex Gaussian random realisation
    omi, omj = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs, 1)
    omk, oml = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs, 1)
    oma, omb = (omi + 1.0j * omj) / 2**0.5, (omk + 1.0j * oml) / 2**0.5

    # Construct RHS vector
    b = np.zeros((Nfreqs + Nmodes, 1), dtype=complex)
    b[:Nfreqs] = Ni @ (w * d).T + Sih @ oma + Nih @ omb
    b[Nfreqs:] = fgmodes.T.conj() @ (Ni @ (w * d).T + Nih @ omb)

    # Run CG solver, preconditioned by M=Ai
    x0 = None
    if f0 is not None:
        x0 = np.concatenate((np.zeros(Nfreqs, dtype=complex), f0))
    xsoln, info = sp.sparse.linalg.cg(A, b, maxiter=1e5, x0=x0, M=Ai)

    # Return solution vector
    return xsoln


def gcr_fgmodes(vis, w, matrices, fgmodes, f0=None, nproc=1):
    """
    Perform the GCR step on all time samples, using parallelisation if
    possible.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        w (array_like):
            Array of flags or weights (e.g. 1 for unflagged, 0 for flagged).
        matrices (array_like):
            Array containing precomputed matrices needed by the linear system.
        fgmodes (array_like):
            Foreground mode array, of shape (Nmodes, Nfreqs). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        fourier_op (array_like):
            Pre-computed Fourier operator.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        nproc (int):
            Number of processes to use for parallelised functions.

    Returns:
        samples (array_like):
            Array of signal + foreground realisations for each time sample,
            of shape `(Ntimes, Nfreqs + Nmodes)`.
    """
    samples = np.zeros((vis.shape[0], vis.shape[1] + fgmodes.shape[1]), dtype=complex)
    idxs = np.arange(vis.shape[0])

    # Run GCR method on each time sample in parallel
    st = time.time()
    with Pool(nproc) as pool:
        samples = pool.map(
            lambda idx: gcr_fgmodes_1d(
                vis=vis[idx],
                w=w,
                matrices=matrices,
                fgmodes=fgmodes,
                f0=f0,
            ),
            idxs,
        )

    # Return sample
    print("%.1fs" % (time.time() - st), end=" ")
    return np.array(samples).reshape((vis.shape[0], -1))


def covariance_from_pspec(ps, fourier_op):
    """
    Transform the sampled power spectrum into a frequency-frequency covariance
    matrix that can be used for the next iteration.
    """
    Nfreqs = ps.size
    Csigfft = np.zeros((Nfreqs, Nfreqs), dtype=complex)
    Csigfft[np.diag_indices(Nfreqs)] = ps
    C = (fourier_op.T.conj() @ Csigfft @ fourier_op).real
    return C


def gibbs_step_fgmodes(
    vis, flags, signal_S, fgmodes, Ninv, ps_prior=None, f0=None, nproc=1
):
    """
    Perform a single Gibbs iteration for a Gibbs sampling scheme using a foreground model
    based on frequency templates for multiple foreground modes.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
        signal_S (array_like):
            Current value of the EoR signal frequency-frequency covariance.
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        ps_prior (array_like):
            EoR signal power spectrum prior.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        nproc (int):
            Number of processes to use for parallelised functions.

    Returns:
        signal_cr (array_like):
            Samples of the signal, shape `(Ntimes, Nfreqs)`.
        S_sample (array_like):
            Sample of the signal covariance, shape `(Nfreqs, Nfreqs)`. This is
            simply a transformation of the power spectrum.
        ps_sample (array_like):
            Sample of the signal power spectrum bandpowers, shape `(Nfreqs,)`.
        fg_amps (array_like):
            Sample of the foreground amplitudes, shape `(Nmodes,)`.
    """
    # Shape of data and operators
    Nvis, Nfreqs = vis.shape
    Nmodes = fgmodes.shape[1]
    Nparams = Nfreqs + Nmodes
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"

    # Precompute 2D Fourier operator matrix
    fourier_op = utils.fourier_operator(Nfreqs)

    # Construct matrix structure
    matrices = [0, 0]
    matrices[0] = np.zeros((5, Nfreqs, Nfreqs), dtype=complex)
    matrices[1] = np.zeros((2, Nparams, Nparams), dtype=complex)

    # Construct necessary operators for GCR
    matrices[0][0] = sp.linalg.sqrtm(signal_S)  # Sh
    matrices[0][1] = np.linalg.inv(signal_S)  # Si
    matrices[0][2] = flags.T * Ninv * flags  # Ni # FIXME
    matrices[0][3] = sp.linalg.sqrtm(matrices[0][1])  # Sih
    matrices[0][4] = sp.linalg.sqrtm(matrices[0][2])  # Nih

    # Construct operator matrix
    A = np.zeros((Nparams, Nparams), dtype=complex)
    A[:Nfreqs, :Nfreqs] = matrices[0][1] + matrices[0][2]  # Si + Ni
    A[:Nfreqs, Nfreqs:] = matrices[0][2] @ fgmodes
    A[Nfreqs:, :Nfreqs] = fgmodes.T.conj() @ matrices[0][2]
    A[Nfreqs:, Nfreqs:] = fgmodes.T.conj() @ matrices[0][2] @ fgmodes

    matrices[1][0] = A
    matrices[1][1] = np.linalg.pinv(A)  # pseudo-inverse, to be used as a preconditioner

    # (1) Solve GCR equation to get EoR signal and foreground amplitude realisations
    cr = gcr_fgmodes(
        vis=vis, w=flags, matrices=matrices, fgmodes=fgmodes, f0=f0, nproc=nproc
    )

    # Extract separate signal and FG parts from the solution
    signal_cr = cr[:, : -fgmodes.shape[1]]
    fg_amps = cr[:, -fgmodes.shape[1] :]

    # (2) Sample EoR signal power spectrum (and also convert to equivalent
    # covariance matrix sample)
    ps_sample = sample_S(s=signal_cr, prior=ps_prior)
    S_sample = covariance_from_pspec(ps_sample / (2 * Nfreqs**2), fourier_op)

    # Return samples
    return signal_cr, S_sample, ps_sample, fg_amps


def gibbs_sample_with_fg(
    vis,
    flags,
    S_initial,
    fgmodes,
    Ninv,
    ps_prior,
    Niter=100,
    seed=None,
    verbose=True,
    nproc=1,
):
    """
    Run a Gibbs chain on data for a single baseline, using a foreground model
    based on frequency templates for multiple foreground modes.

    This will return samples of EoR signal and foreground amplitude
    constrained realisations, and the signal frequency-frequency covariance
    and power spectrum.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
        S_initial (array_like):
            Initial guess for the EoR signal frequency-frequency covariance.
            A better guess should result in faster convergence.
        fgmodes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        ps_prior (array_like):
            EoR signal power spectrum prior.
        Niter (int):
            Number of iterations of the sampler to run.
        seed (int):
            Random seed to use for random parts of the sampler.
        verbose (bool):
            If True, output basic timing stats about each iteration.
        nproc (int):
            Number of processes to use for parallelised functions.

    Returns:
        signal_cr (array_like):
            Samples of the signal, shape `(Niter, Ntimes, Nfreqs)`.
        signal_S (array_like):
            Samples of the signal covariance, shape `(Niter, Nfreqs, Nfreqs)`.
            These are simply transformations of the power spectrum.
        signal_ps (array_like):
            Sample of the signal power spectrum bandpowers, shape
            `(Niter, Nfreqs)`.
        fg_amps (array_like):
            Samples of the foreground amplitudes, shape `(Niter, Nmodes)`.
    """
    # Set random seed
    np.random.seed(seed)

    # Get shape of data/foreground modes
    Ntimes, Nfreqs = vis.shape
    Nmodes = fgmodes.shape[1]
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"
    assert fgmodes.shape[0] == Nfreqs, "fgmodes must have shape (Nfreqs, Nmodes)"
    if len(Ninv.shape) == 3:
        assert (
            Ninv.shape[0] == Ntimes
        ), "Ninv shape must be (Ntimes, Nfreqs, Nfreqs) or (Nfreqs, Nfreqs)"

    # Set up arrays for sampling
    signal_cr = np.zeros((Niter, Ntimes, Nfreqs), dtype=complex)
    signal_S = np.zeros((Niter, Nfreqs, Nfreqs))
    signal_ps = np.zeros((Niter, Nfreqs))
    fg_amps = np.zeros((Niter, Ntimes, Nmodes), dtype=complex)

    # Set initial value for signal_S
    signal_S = S_initial.copy()

    # Loop over iterations
    for i in range(Niter):
        if verbose:
            print("IT#%04d" % (i + 1), end=", ")

        # Do Gibbs iteration
        signal_cr[i], signal_S, signal_ps[i], fg_amps[i] = gibbs_step_fgmodes(
            vis=vis * flags,
            flags=flags,
            signal_S=signal_S,
            fgmodes=fgmodes,
            Ninv=Ninv,
            ps_prior=ps_prior,
            f0=None,
            nproc=nproc,
        )

    return signal_cr, signal_S, signal_ps, fg_amps
