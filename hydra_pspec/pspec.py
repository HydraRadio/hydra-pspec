import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.stats import invgamma

from multiprocess import Pool, current_process
from . import utils
import time


def _check_prior_values(prior_min, prior_max):
    """
    Bounds check for prior values.
    """
    if np.any(prior_min <= 0):
        raise ValueError("prior_min must be greater than zero")
    if np.any(prior_max <= 0):
        raise ValueError("prior_max must be greater than zero")
    if not np.any(np.isfinite(prior_max)):
        raise ValueError("prior_max must be finite")
    if np.any(prior_max <= prior_min):
        raise ValueError("prior_max must be greater than prior_min")


def inversion_sample_invgamma(alpha, beta, prior_min, prior_max, ngrid=1000):
    """
    Draw a sample from an inverse gamma distribution between prior bounds using
    inversion sampling.
    
    This works by sampling the cdf of the inverse gamma distribution 
    on a (logarithmic) grid and then interpolating to convert a uniform 
    random draw into a random draw with the correct PDF.

    Parameters:
        alpha (float):
            Inverse gamma alpha parameter.
        beta (float):
            Inverse gamma beta (scale) parameter.
        prior_min (float):
            Minimum of the prior range.  The log10 of this value will be taken.
            As such, `prior_min` must be greater than zero.
        prior_max (float):
            Maximum of the prior range.  The log10 of this value will be taken.
            `prior_max` must be greater than zero, finite, and greater than
            `prior_min`.
        ngrid (int):
            Number of sample points to use for interpolator.  Defaults to 1000.

    Returns:
        sample (float):
            Sample drawn from the inverse gamma distribution between the 
            specified prior bounds.
    """
    # Check prior values
    _check_prior_values(prior_min, prior_max)
    
    # Sample cdf logarithmically between provided prior bounds
    x = np.logspace(np.log10(prior_min), np.log10(prior_max), ngrid)
    cdf = invgamma.cdf(x, a=alpha, loc=0, scale=beta)
    cdf -= cdf.min()  # shift minimum down to zero
    cdf /= cdf.max()  # rescale maximum to 1

    # Remove duplicate entries in cdf so interpolator can work properly; 
    # tends to result in sample points near the extrema of the prior bounds
    cdf_unique, idxs_unique = np.unique(cdf, return_index=True)
    u = np.random.uniform()
    # Draw sample using inversion sampling method
    # Warning: must use linear interpolation to avoid
    # very bad interpolation results
    sample = interp1d(cdf_unique, x[idxs_unique], kind='linear')(u)

    return sample


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
            `(Ntimes, Nfreq)`.  The monopole is expected to be at the center
            of the frequency axis, i.e. the frequency axis has been fftshifted.

        prior (array_like):
            Array of delta function prior values, used to set certain modes to a
            fixed value.
    """
    if s is None and sk is None:
        raise ValueError("Must pass in s (real space) or sk (Fourier space) vector.")

    if sk is None:
        axes = (1,)
        sk = np.fft.ifftshift(s, axes=axes)
        sk = np.fft.fftn(sk, axes=axes)
        sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape

    if prior is None:
        prior = np.zeros((2, Nfreqs), dtype=float)

    # The scale parameter for the inverse gamma distribution (beta) is
    # equivalent to (Ntimes - 1) times the variance over the time axis of the
    # delay spectrum of the Gaussian Constrained Realization of the EoR
    beta = np.sum(sk * sk.conj(), axis=0).real
    # The shape parameter (alpha) differs from that used in Eriksen et al. 2008
    # i.e. `alpha = Nobs/2 - 1` because our data vector is complex and has
    # twice as many numbers as a purely real data vector
    alpha = Nobs - 1.0

    # We obtain samples of the power spectrum (x) by instead sampling the random
    # variable y = x / beta and then obtain x via x = y * beta
    x = np.zeros(Nfreqs)
    for i in range(Nfreqs):
        if np.any(prior[:, i] > 0):
            # The pdf for a log-uniform prior is proportional to 1 / x.
            # Multiplying the inverse gamma likelihood by this prior results
            # in an additional factor of 1 / x which increases the effective
            # value of the shape parameter (alpha) by 1.  With a log-uniform
            # prior, we thus sample from an inverse gamma distribution with
            # shape parameter alpha + 1.
            x[i] = inversion_sample_invgamma(
                alpha+1, beta[i], prior[1, i], prior[0, i]
            )
        else:
            x[i] = invgamma.rvs(a=alpha) * beta[i]

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


def gcr_fg_modes_1d(
    idx, vis, matrices, fg_modes, f0=None, map_estimate=False, 
    preconditioner=None, verbose=False, multiprocess_seed=912983, 
    linear_solver=sp.sparse.linalg.cg, maxiter=100000, rtol=1e-10, atol=1e-8
):
    """
    Perform the GCR step on a single time sample.

    Parameters:
        idx (int):
            Time index. Used to generate a unique random seed for each process
            if using `multiprocess.Pool` and multiple processes.
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        matrices (array_like):
            Array containing precomputed matrices needed by the linear system.
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        preconditioner (int):
            If None, don't use a preconditioner. If -1, use a full 
            pseudo-inverse.
        verbose (bool):
            If True, output basic timing stats about each iteration.
        multiprocess_seed (int):
            Reference random seed used for all processes and time indices.
            Used to generate a unique random seed for each spawned process and
            each time index.
        linear_solver (func):
            Linear solver function, from `scipy.sparse.linalg`.
        maxiter (int):
            Maximum numbe rof iterations that the solver can use.
        rtol (float):
            Relative tolerance that the solver should try to achieve.
        atol (float):
            Absolute tolerance that the solver should try to achieve.

    """
    # If multiple process are spawned via `multiprocess.Pool`, each process
    # inherits the random seed of the parent process.  We need to set a unique
    # seed per process to avoid spurious correlations between GCRs at different
    # time indices.  We can do so using the process ID (PID, unique per
    # process) and time index (unique for each time).  For fewer than 1000
    # processes, we can guarantee a unique random seed by summing the
    # multiprocess_seed (a reference seed which is fixed for all processes and
    # times), the PID*1000, and the time index.
    # WARNING: if more than 1000 processes is every used this sum will not
    # guarantee a unique seed for each process!
    pid = current_process().pid
    seed = multiprocess_seed + idx
    np.random.seed(seed)
    
    # Get projection operator shape
    Nfreqs, Nmodes = fg_modes.shape
    
    # Get current data (for single time sample)
    d = vis[idx]

    # Extract precomputed matrices needed by the linear system
    S = matrices['S'] # (Nkbin, Nkbin) matrix; currently, NKbin = Nfreqs
    Sh = matrices['Sh'] # (Nkbin, Nkbin) matrix
    Ninv = matrices['Ninv'][idx]    # vector of shape (Nfreqs,)
    Nih = matrices['Nih'][idx]      # vector of shape (Nfreqs,)
    Ninv_mat = np.diag(Ninv)        # matrix with Ninv along the diagonal
    Nih_mat = np.diag(Nih)          # matrix with Nih along the diagonal

    if map_estimate:
        oma = np.zeros((Nfreqs), dtype=complex)
        omb = np.zeros((Nfreqs), dtype=complex)
    else:
        # Unit complex Gaussian random realisation
        omi, omj = np.random.randn(Nfreqs), np.random.randn(Nfreqs)
        omk, oml = np.random.randn(Nfreqs), np.random.randn(Nfreqs)
        oma, omb = (omi + 1.0j * omj) / 2**0.5, (omk + 1.0j * oml) / 2**0.5

    # Construct RHS vector
    b = np.zeros(Nfreqs + Nmodes, dtype=complex)
    b[:Nfreqs] = S @ (Ninv * d + Nih * omb) + Sh @ oma
    b[Nfreqs:] = fg_modes.T.conj() @ ( Ninv * d + Nih * omb)
    
    # Construct linear operator matrix
    Nparams = Nfreqs + Nmodes
    A = np.zeros((Nparams, Nparams), dtype=complex)
    A[:Nfreqs, :Nfreqs] = np.eye(Nfreqs) + S @ Ninv_mat  # 1 + S @ Ni
    A[:Nfreqs, Nfreqs:] = S @ Ninv_mat @ fg_modes
    A[Nfreqs:, :Nfreqs] = fg_modes.T.conj() @ Ninv_mat
    A[Nfreqs:, Nfreqs:] = fg_modes.T.conj() @ Ninv_mat @ fg_modes
    
    # Select preconditioner (may be slower than just solving!)
    if preconditioner is None:
        # Trivial identity preconditioner
        Ainv = np.eye(A.shape[0])
    else:
        Ainv = np.linalg.pinv(A, hermitian=False)
    
    # Run linear solver, preconditioned by M=Ainv
    x0 = None
    if f0 is not None:
        x0 = np.concatenate((np.zeros(Nfreqs, dtype=complex), f0))
    
    xsoln, info = linear_solver(A, 
                                b, 
                                maxiter=maxiter, 
                                rtol=rtol, 
                                atol=atol, 
                                x0=x0, 
                                M=Ainv)
    if verbose:
        residual = np.abs(A @ xsoln - b).mean()
    else:
        residual = None

    # Return solution vector
    return xsoln, residual, info


def gcr_fg_modes(
    vis, matrices, fg_modes, f0=None, nproc=1, map_estimate=False,
    preconditioner=None, linear_solver=sp.sparse.linalg.cg, verbose=False
):
    """
    Perform the GCR step on all time samples, using parallelisation if
    possible.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        matrices (dict):
            Dictionary containing precomputed matrices needed by the linear system.
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        fourier_op (array_like):
            Pre-computed Fourier operator.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        nproc (int):
            Number of processes to use for parallelised functions.
        map_estimate (bool):
            Provide the maximum a posteriori estimate rather than drawing a 
            sample.
        preconditioner (int):
            If None, don't use a preconditioner. If -1, use a full 
            pseudo-inverse.
        linear_solver (func):
            Linear solver function, from `scipy.sparse.linalg`.
        verbose (bool):
            If True, output basic timing stats about each iteration.

    Returns:
        samples (array_like):
            Array of signal + foreground realisations for each time sample,
            of shape `(Ntimes, Nfreqs + Nmodes)`.
    """
    # Set up results arrays
    samples = np.zeros((vis.shape[0], vis.shape[1] + fg_modes.shape[1]), 
                       dtype=complex)
    if verbose:
        residuals = np.zeros(vis.shape[0], dtype=float)
        info = np.zeros(vis.shape[0], dtype=float)
    else:
        residuals = None
        info = None
    idxs = np.arange(vis.shape[0])

    # Run GCR method on each time sample in parallel
    if verbose:
        st = time.time()
    
    """
    with Pool(nproc) as pool:
        yyy = zip(*pool.map(
            lambda idx: gcr_fg_modes_1d(
                idx=idx,
                vis=vis,
                matrices=matrices,
                fg_modes=fg_modes,
                f0=f0,
                map_estimate=map_estimate,
                verbose=verbose
            ),
            idxs,
        ))
        samples, residuals, info = yyy
    """
    samples, residuals, info = [], [], []
    tstart = time.time()
    for j, idx in enumerate(idxs):
        
        # Do 1D constrained realisation
        _samples, _residuals, _info = gcr_fg_modes_1d(
                                                idx=idx,
                                                vis=vis,
                                                matrices=matrices,
                                                fg_modes=fg_modes,
                                                f0=f0,
                                                map_estimate=map_estimate,
                                                preconditioner=preconditioner,
                                                linear_solver=linear_solver,
                                                verbose=verbose
                                            )
        samples.append(_samples)
        residuals.append(_residuals)
        info.append(_info)
        
    samples = np.array(samples).reshape((vis.shape[0], -1))
    residuals = np.array(residuals)
    info = np.array(info)

    # Return sample
    if verbose:
        print(f"{time.time() - st:<12.1f}", end="")
        print(f"{info.mean():<8.1f}", end="")
        print(f"{residuals.mean():<12.2e}", end="")
    return samples


def covariance_from_pspec(ps, fourier_op):
    """
    Transform the sampled power spectrum into a frequency-frequency covariance
    matrix that can be used for the next iteration.
    """
    Nfreqs = ps.size
    Csigfft = np.zeros((Nfreqs, Nfreqs), dtype=complex)
    
    # Assign power spectrum values to diagonal
    Csigfft[np.diag_indices(Nfreqs)] = ps
    
    # Apply Fourier operator to return to time/freq. space
    C = (fourier_op.T.conj() @ Csigfft @ fourier_op)
    return C


def gibbs_step_fg_modes(
    vis, flags, signal_S, fg_modes, inv_var, ps_prior=None, f0=None, nproc=1,
    map_estimate=False, preconditioner=None, linear_solver=sp.sparse.linalg.cg, 
    verbose=False
):
    """
    Perform a single Gibbs iteration for a Gibbs sampling scheme using a foreground model
    based on frequency templates for multiple foreground modes.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), of shape
            `(Ntimes, Nfreqs)`.
        signal_S (array_like):
            Current value of the EoR signal frequency-frequency covariance.
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        inv_var (array_like):
            Inverse noise variance, of shape `(Ntimes, Nfreqs)`.
        ps_prior (array_like):
            EoR signal power spectrum prior.
        f0 (array_like):
            Initial guess for the foreground amplitudes, with shape `(Nmodes,)`.
        nproc (int):
            Number of processes to use for parallelised functions.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        preconditioner (int):
            If None, don't use a preconditioner. If -1, use a full 
            pseudo-inverse.
        linear_solver (func):
            Linear solver function, from `scipy.sparse.linalg`.
        verbose (bool):
            If True, output basic timing stats about each iteration.

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
    Ntimes, Nfreqs = vis.shape
    Nmodes = fg_modes.shape[1]
    Nparams = Nfreqs + Nmodes
    assert flags.shape == vis.shape, "`flags` array must have the same shape as `vis`"

    # Precompute 2D Fourier operator matrix
    fourier_op = utils.fourier_operator(Nfreqs)
    
    # Bundle operator matrices together
    matrices = {
        'S':        signal_S,                  # S, signal covariance
        'Sh':       sp.linalg.sqrtm(signal_S), # Sh
        'Ninv':     flags * inv_var,           # inverse variance with flags
        'Nih':      flags * np.sqrt(inv_var)   # sqrt inverse variance
    }
        
    # (1) Solve GCR equation to get EoR signal and foreground amplitude realisations
    cr = gcr_fg_modes(
                     vis=vis, 
                     matrices=matrices, 
                     fg_modes=fg_modes, 
                     f0=f0, 
                     nproc=nproc,
                     map_estimate=map_estimate, 
                     preconditioner=preconditioner,
                     linear_solver=linear_solver,
                     verbose=verbose
    )

    # Extract separate signal and FG parts from the solution
    signal_cr = cr[:, : -fg_modes.shape[1]]
    fg_amps = cr[:, -fg_modes.shape[1] :]

    # Full model of data is sum of EoR (GCR) + FG model
    model = signal_cr + fg_amps @ fg_modes.T  # np.einsum('ijk,lk->ijl', fg_amps, fg_modes)

    # Chi-squared is computed as the sum of ( |data - model| / noise )^2,
    # i.e. as a sum of standard normal random variables
    chisq = np.abs(vis - model)**2 * inv_var
    if verbose:
        chisq_mean = chisq[np.bool(flags)].mean()
        if chisq_mean > 10:
            print(f"{chisq_mean:<9.1e}", end="")
        else:
            print(f"{chisq_mean:<9.3f}", end="")

    # (2) Sample EoR signal power spectrum (and also convert to equivalent
    # covariance matrix sample)
    ps_sample = sample_S(s=signal_cr, prior=ps_prior)
    
    # The factor of 1/Nfreqs**2 here is an FFT normalization
    S_sample = covariance_from_pspec(ps_sample / Nfreqs**2, fourier_op)

    # Log posterior
    # Each time is treated as an independent sample. So, the joint
    # log posterior for all times is the sum of the individual log
    # posteriors for each time.
    # WARNING: np.linalg.inv should be avoided for general, dense matrices.
    # S_sample should be diagonally dominant and thus this should be okay.
    # FIXME FIXME: Re-enable this
    Sinv = np.eye(S_sample.shape[0]) # np.linalg.pinv(S_sample)
    lnprob_like = (vis - model).conj() * inv_var * (vis - model)
    lnprob_prior = [signal_cr[i].conj() @ Sinv @ signal_cr[i] 
                            for i in range(Ntimes)] # ignore flags for prior
    ln_post = (np.sum(lnprob_like[np.bool(flags)]) + np.sum(lnprob_prior)).real
    if verbose:
        print(f"{ln_post:<12.1f}")

    # Return samples
    return signal_cr, S_sample, ps_sample, fg_amps, chisq, ln_post


def gibbs_sample_with_fg(
    vis,
    flags,
    S_initial,
    fg_modes,
    inv_var,
    ps_prior,
    Niter=100,
    seed=None,
    verbose=True,
    nproc=1,
    write_Niter=100,
    out_dir=None,
    map_estimate=False,
    preconditioner=None,
    linear_solver=sp.sparse.linalg.cg,
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
            Array of flags (1 for unflagged, 0 for flagged), of shape
            `(Ntimes, Nfreqs)`.
        S_initial (array_like):
            Initial guess for the EoR signal frequency-frequency covariance.
            A better guess should result in faster convergence.
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        inv_var (array_like):
            Inverse noise variance, of shape `(Ntimes, Nfreqs)`. Assumed to be 
            the same for the real and imaginary parts.
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
        write_Niter (int):
            Number of iterations between output file writing.
        out_dir (str or Path):
            Directory where samples will be saved to disk.  If None (default),
            samples are not written to disk.
        map_estimate (bool):
            Provide the maximum a posteriori sample only, i.e. sets
            `Niter = 1`.
        preconditioner (int):
            If None, don't use a preconditioner. If -1, use a full 
            pseudo-inverse.
        linear_solver (func):
            Linear solver function, from `scipy.sparse.linalg`.

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
        chisq (array_like):
            Chi-squared value per iteration, shape `(Niter, Ntimes, Nfreqs)`.
        ln_post (array_like):
            Natural log of the posterior probability per iteration, shape
            `(Niter,)`.
        write_time:
            Time spent writing data.
    """
    if map_estimate:
        Niter = 1
        write_Niter = 1
    else:
        # Set random seed
        np.random.seed(seed)

    # Get shape of data/foreground modes
    Ntimes, Nfreqs = vis.shape
    Nmodes = fg_modes.shape[1]
    assert flags.shape == vis.shape, "`flags` array must have the same shape as `vis`"
    assert fg_modes.shape[0] == Nfreqs, "fg_modes must have shape (Nfreqs, Nmodes)"
    assert inv_var.shape == vis.shape, "`inv_var` array must have the same shape as `vis`"
    
    # Check prior bounds
    _check_prior_values(ps_prior[1], ps_prior[0])
    
    # Set up arrays for sampling
    signal_cr = np.zeros((Niter, Ntimes, Nfreqs), dtype=complex)
    signal_S = np.zeros((Niter, Nfreqs, Nfreqs))
    signal_ps = np.zeros((Niter, Nfreqs))
    fg_amps = np.zeros((Niter, Ntimes, Nmodes), dtype=complex)
    
    # Useful debugging statistics
    chisq = np.zeros((Niter, Ntimes, Nfreqs))
    ln_post = np.zeros(Niter)

    # Set initial value for signal_S
    signal_S = S_initial.copy()

    # Loop over iterations
    if verbose:
        print("Iter     Time [s]    Info    |Ax - b|    Chisq    ln Post")
        print("-----    --------    ----    --------    -----    -------")
    write_time = 0
    for i in range(Niter):
        if verbose:
            print(f"{i+1:<9d}", end="")

        # Do Gibbs iteration
        signal_cr[i], signal_S, signal_ps[i], fg_amps[i], chisq[i], ln_post[i]\
            = gibbs_step_fg_modes(
                vis=vis * flags,
                flags=flags,
                signal_S=signal_S,
                fg_modes=fg_modes,
                inv_var=inv_var,
                ps_prior=ps_prior,
                f0=None,
                nproc=nproc,
                map_estimate=map_estimate,
                preconditioner=preconditioner,
                linear_solver=linear_solver,
                verbose=verbose
            )

        if out_dir is not None and (i+1) % write_Niter == 0:
            # Write current set of samples to disk
            write_time_start = time.perf_counter()
            utils.write_numpy_files(
                out_dir,
                signal_cr[:i+1],
                signal_S[:i+1],
                signal_ps[:i+1],
                fg_amps[:i+1],
                chisq[:i+1],
                ln_post[:i+1]
            )
            write_time_stop = time.perf_counter()
            write_time += write_time_stop - write_time_start
    
    if out_dir is not None and Niter % write_Niter > 0:
        # Write all samples to disk
        write_time_start = time.perf_counter()
        utils.write_numpy_files(
            out_dir,
            signal_cr,
            signal_S,
            signal_ps,
            fg_amps,
            chisq,
            ln_post
        )
        write_time_stop = time.perf_counter()
        write_time += write_time_stop - write_time_start

    if verbose:
        print()

    return signal_cr, signal_S, signal_ps, fg_amps, chisq, ln_post, write_time
