import numpy as np
import scipy as sp
from scipy.signal.windows import blackmanharris as BH
from scipy.stats import invgamma
from scipy.interpolate import interp1d
import scipy.linalg
from . import sys_solver as sys_sol
from multiprocess import Pool, current_process
from . import utils
import os, time
import cProfile
import pstats
import sys
import uvtools
from uvtools.dspec import gen_window
from uvtools.utils import FFT
from pyuvdata import UVData
from tqdm import tqdm  #For progress bars
from .plotting_functions import master_plotter #For plotting iterations
#uvd=UVData() #Loading uvh5 files
#pr=cProfile.Profile() #For profiling

def data_dly_fr(data, freqs, times, windows=None,
                    freq_window_kwargs=None, time_window_kwargs=None):
    """
    Transform data to delay fringe-rate space
    
    This function takes a 2D array of visibility data (in units of Jy), as well 
    as the corresponding frequency and time arrays (in units of Hz and JD, respectively), 
    and makes a 2x2 grid of plots where each plot shows each one of the possible choices 
    for Fourier transforming along an axis. The upper-left plot is in the frequency-time 
    domain; the upper-right plot is in the frequency-fringe-rate domain; the lower-left 
    plot is in the delay-time domain; and the lower-right plot is in the delay-fringe-rate 
    domain.
    
    Parameters
    ----------
    data : ndarray, shape=(NTIMES,NFREQS)
        Array containing the visibility to be plotted. Assumed to be in units of Jy. 
        
    freqs : ndarray, shape=(NFREQS,)
        Array containing the observed frequencies. Assumed to be in units of Hz.
        
    times : ndarray, shape=(NTIMES,)
        Array containing the observed times. Assumed to be in units of JD.
        
    windows : tuple of str or str, optional
        Choice of taper to use for the fringe-rate and delay transforms. Must be 
        either tuple, list, or string. If a tuple or list, then it must be either 
        length 1 or length 2; if it is length 2, then the zeroth entry is the taper 
        to be applied along the time axis for the fringe-rate transform, with the 
        other entry specifying the taper to be applied along the frequency axis 
        for the delay transform. Each entry is passed to uvtools.dspec.gen_window. 
        If ``windows`` is a length 1 tuple/list or a string, then it is assumed 
        that the same taper is to be used for both axes. Default is to use no 
        taper (or, equivalently, a boxcar).

    freq_window_kwargs : dict, optional
        Keyword arguments to pass to uvtools.dspec.gen_window for generating the 
        frequency taper. Default is to pass no keyword arguments.
        
    time_window_kwargs : dict, optional
        Keyword arguments to pass to uvtools.dspec.gen_window for generating the 
        time taper. Default is to pass no keyword arguments.
    
    Returns
    -------
    data_dl_fr :
        data in delay-fringe rate space
    """
    # do some data prep
    freq_window_kwargs = freq_window_kwargs or {}
    time_window_kwargs = time_window_kwargs or {}
    if windows is not None:
        time_window = gen_window(windows, times.size, **time_window_kwargs)
        freq_window = gen_window(windows, freqs.size, **freq_window_kwargs)
    else:
        time_window = gen_window(None, times.size, **time_window_kwargs)
        freq_window = gen_window(None, freqs.size, **freq_window_kwargs)
        
    time_window = time_window[:, None]
    freq_window = freq_window[None, :]
    data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)
    

    return data_fr_dly

'''ICDF sampler'''
def draw_icdf_samples(alpha, beta, x):
    
    """
    Draw a single sample from an inverse gamma distribution using inverse CDF sampling.

    This function performs inversion sampling from an inverse gamma distribution 
    defined by the shape parameter `alpha + 1` and scale parameter `beta`. The 
    sampling is constrained to the domain specified by `x`, and ensures that 
    the sample falls within this range by rescaling the CDF.

    Parameters
    ----------
    alpha : float
        Shape parameter (minus 1) of the inverse gamma distribution. 
        The full shape used is `alpha + 1` for consistency with certain prior formulations.
    
    beta : float
        Scale parameter of the inverse gamma distribution.

    x : ndarray
        A 1D array of values over which to compute the CDF and draw a sample.
        Should be sorted in increasing order and span the desired sampling range.

    Returns
    -------
    float
        A sample drawn from the inverse gamma distribution within the range defined by `x`.
    """
    
    cdf = invgamma.cdf(x, a=alpha+1, loc=0, scale=beta)
    cdf -= cdf.min() # shift minimum down to zero
    if cdf.max()==0.0:
        cdf /= (cdf.max()+1e-5) # rescale maximum to 1
    else:
        cdf /= cdf.max() # rescale maximum to 1

    # Remove duplicate entries in cdf so interpolator can work properly; 
    # tends to result in sample points near the extrema of the prior bounds anyway
    cdf_unique, idxs_unique = np.unique(cdf, return_index=True)
    u = np.random.uniform(high=cdf.max()) #High set to this value, otherwise the sample drawn is always out of the upper bound
    # Draw sample using inversion sampling method
    # Note: Must use linear interpolation to avoid very bad interpolation results
    return interp1d(cdf_unique, x[idxs_unique], kind='linear')(u)

def sample_pspec(s, prior, ngrid=120, sk=None,max_prior_iter=10000):
    """
    Draw a sample from an inverse gamma distribution using inversion 
    sampling between uniform prior bounds.
    
    This works by sampling the cdf of the inverse gamma distribution 
    on a (logarithmic) grid and then interpolating to convert a uniform 
    random draw into a random draw with the correct pdf.

    Parameters:
        alpha: (float)
            Inverse gamma alpha parameter.
        beta: (float)
            Inverse gamma beta (scale) parameter.
        ngrid: (int)
            Number of sample points to use for interpolator.

    Returns:
        sample: (float)
            Sample drawn from the inverse gamma distribution between the 
            specified prior bounds.
    """

    if sk is None:
        axes = (1,)
        sk = np.fft.fftshift(s, axes=axes)
        sk = np.fft.ifftn(sk, axes=axes) * np.sqrt(s.shape[1]) # note normalisation
        sk = np.fft.ifftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    
    alpha = Nobs-1
    beta = np.sum(sk * sk.conj(), axis=0).real # normalisation

    # Sample cdf logarithmically between provided prior bounds
    xgrid = np.logspace(np.log10(prior.min()), np.log10(prior.max()), ngrid) #FIXME: the prior min is 0, can't have that. 
    
    samples = np.zeros(Nfreqs)
    for i in range(Nfreqs):
        samples[i] = draw_icdf_samples(alpha, beta[i], xgrid)
    return samples


def sprior(signals, bins, factor):
    """
    Compute the prior on covariance samples based on the Fourier transform of the input signals.

    This function calculates a prior on the covariance of Fourier-transformed signals. The prior is defined
    by a range determined by a `factor` which scales the observed power spectrum, and only a specific number 
    of frequency bins around zero delay are retained, with others set to zero.

    Parameters
    ----------
    signals : numpy.ndarray
        A 2D array of shape (Nobs, Nfreq) where `Nobs` is the number of observations and `Nfreq` is the number
        of frequency channels. This array contains the observed signals to be transformed.

    bins : int
        The number of bins on either side of zero delay to retain in the prior. For example, `bins=2` will 
        retain the frequency bins corresponding to delays [-2, -1, 0, 1, 2].

    factor : float
        A scaling factor that defines the range of the prior. The upper bound of the prior is the observed 
        power spectrum multiplied by `factor`, and the lower bound is the observed power spectrum divided 
        by `factor`.

    Returns
    -------
    prior : numpy.ndarray
        A 2D array of shape (2, Nfreq) containing the prior bounds. The first row (`prior[0]`) contains 
        the upper bounds, and the second row (`prior[1]`) contains the lower bounds. Frequency bins outside 
        the specified range (determined by `bins`) are set to zero.
    """

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


def gcr_fg_and_signal_per_time(idx, 
                               vis, 
                               Einv, 
                               sqrtE, 
                               sqrtNinv, 
                               Nparams, 
                               sys_model, 
                               flags, 
                               Ninv, 
                               fg_modes, 
                               map_estimate=False, 
                               verbose=False,
                               multiprocess_seed=None,
                               solver='lgmres',
                               solver_tol=1e-12):
    """
    Solves the GCR equation for the joint foreground + signal model 
    for a single time
    
    Parameters:
        idx (int):
            Time index in the loop. Only used for setting the random seed 
            and debug output.
        vis (array_like):
            Visibility data being modelled (Ntimes, Nfreqs)
        Nparams (int):
            Number of model parameters.
        sys_model (array_like):
            Systematics gain model for this time index. Shape (Nfreqs,)
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
        Einv (array_like):
            Current value of the EoR signal frequency-frequency covariance inverse.
        sqrtE (array_like):
            Square-root of E matrix (Nfreqs, Nfreqs)
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        sqrtNinv (array_like):
            Square-root of Ninv, same shape as Ninv
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        solver_tol (float):
            Tolerance `tol` for scipy linear solvers.
        
    Returns:
        xsoln (array_like):
            Solution of the GCR for idx time index. First half is EoR solution, second half is foreground amplitudes. (2*Nfreqs, 1)

        residual (float):
            Residual |Axsoln-b|; indicates solution accuracy
        
        info (int):
            Info from the linear solver. Contains convergence information. 0 indicates success. 
    """
    # Set parallel-safe random seed
    pid = current_process().pid
    seed = None # FIXME: multiprocess_seed + pid*1000 + idx
    np.random.seed(seed)

    Nfreqs, Nmodes = fg_modes.shape
    d = vis.reshape((1, max(Nfreqs, len(vis.T))))  # Do NOT use order='F'

    # Construct necessary operators for GCR
    Ninv_sys = (sys_model.conj().T * Ninv.diagonal() *  sys_model)
    Ni_flagged = flags.T * Ninv_sys * flags  # Ninv with flags and systematics
    
    # Construct block operator matrix
    A = np.zeros((Nparams, Nparams), dtype=complex)
    
    # A_11: g^daggerdag E^-1 g + g^dagger * N^-1 * g
    A[:Nfreqs, :Nfreqs] = sys_model.conj()[:,np.newaxis] * Einv * sys_model[:,np.newaxis] \
                        + np.diag(Ni_flagged)
    
    # A_12: g^dagger * N^-1 * g * G
    A[:Nfreqs, Nfreqs:] = Ni_flagged[:,np.newaxis] * fg_modes
    
    # A_21: G^dagger * g^dagger * N^-1 * g
    A[Nfreqs:, :Nfreqs] = (fg_modes.conj() * Ni_flagged[:,np.newaxis]).T
    
    # A_22: G^dagger * g^dagger * N^-1 * g * G 
    A[Nfreqs:, Nfreqs:] = fg_modes.T.conj() @ (Ni_flagged[:,np.newaxis] * fg_modes)
    # Basic diagonal preconditioner
    Ainv_estimate = np.diag(1. / np.diag(A))
    #Ainv_estimate = np.linalg.pinv(A)

    # Construct fluctuation terms
    if map_estimate:
        oma = np.zeros((Nfreqs, 1), dtype=complex)
        omb = np.zeros((Nfreqs, 1), dtype=complex)
    else:
        # Unit complex Gaussian random realisation
        omi, omj = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs,1)
        omk, oml = np.random.randn(Nfreqs, 1), np.random.randn(Nfreqs,1)
        oma, omb = (omi + 1.0j * omj) / 2**0.5, (omk + 1.0j * oml) / 2**0.5  
    
    # Construct RHS vector
    b = np.zeros((Nfreqs + Nmodes, 1), dtype=complex)
    b[:Nfreqs] = (sys_model.conj() * Ninv.diagonal() * d).T \
               +  sys_model.conj()[:,np.newaxis] * (sqrtE @ oma + sqrtNinv[:,np.newaxis] * omb)
    b[Nfreqs:] = fg_modes.T.conj() @ (
                     (sys_model.conj() * Ninv.diagonal() * d).T \
                   + (sys_model.conj()[:,np.newaxis] * sqrtNinv[:,np.newaxis] * omb) )
    
    # Run CG solver, preconditioned by M ~ A^-1
    x0 = None
    # xsoln, info = sp.sparse.linalg.cgs(A, b, x0=x0, M=Ainv_estimate, tol=solver_tol, maxiter=8000)
    xsoln, info = sp.sparse.linalg.gmres(A, b, x0=x0, M=Ainv_estimate, tol=solver_tol, maxiter=8000)
    
    # Check solution
    if info > 0:
        # Try again with different solver
        xsoln, info2 = sp.sparse.linalg.bicgstab(A, 
                                                 b, 
                                                 x0=x0, 
                                                 M=Ainv_estimate, 
                                                 tol=solver_tol, 
                                                 maxiter=8000)
        if info2 != 0:
            raise ValueError("GCR solver failed after retry; pid %d, time idx %d, info %d, info2 %d" \
                             % (pid, idx, info, info2))
    if info < 0:
        raise ValueError("GCR solver failed; pid %d, time idx %d, info %d" \
                         % (pid, idx, info))

    # Print residual if verbose mode enabled
    if verbose:
        residual = np.sqrt( np.sum(np.abs(A @ xsoln - b[:, 0])**2.) ) # residual = |Ax - b|
    else:
        residual = None

    # Return solution vector
    return xsoln, residual, info


def gcr_fg_and_signal(
    vis, 
    flags, 
    fg_modes, 
    Nparams, 
    sys_model, 
    signal_ps, 
    Ninv, 
    fourier_op,
    nproc=1, 
    map_estimate=False,
    solver='lgmres',
    solver_tol=1e-12,
    verbose=False,
):
    """
    Perform the GCR step on all time samples, using parallelisation if
    possible.

    Parameters:
        vis (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        flags (array_like):
            Array of flags or weights (e.g. 1 for unflagged, 0 for flagged).
        signal_ps (array_like):
            Signal power spectrum.
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        sys_model (array_like):
            Current multiplicative systematics model, of shape `(Ntimes, Nfreqs)`.
        fourier_op (array_like):
            Pre-computed Fourier operator.
        nproc (int):
            Number of processes to use for parallelised functions.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        solver_tol (float):
            Tolerance `tol` for scipy linear solvers.
        verbose (bool):
            If True, output basic timing stats about each iteration.

    Returns:
        samples (array_like):
            Array of signal + foreground realisations for each time sample,
            of shape `(Ntimes, Nfreqs + Nmodes)`.
    """
    # Set up samples array
    samples = np.zeros((vis.shape[0], vis.shape[1] + fg_modes.shape[1]), dtype=complex)
    
    # Prepare residuals/info arrays for each time
    residuals, info = None, None
    if verbose:
        residuals = np.zeros(vis.shape[0], dtype=float)
        info = np.zeros(vis.shape[0], dtype=float)
    
    # Time indices
    time_idxs = np.arange(vis.shape[0])
    
    # Pre-compute quantities that are constant in time
    E = covariance_from_pspec(signal_ps, fourier_op)
    Einv = covariance_from_pspec(1./signal_ps, fourier_op)
    sqrtE = sp.linalg.sqrtm(E) 
    sqrtNinv = np.sqrt(np.diag(Ninv))
    
    # Run GCR solver on each time sample in parallel
    if verbose:
        t_start = time.time()
    
    # FIXME
    samples = []
    residuals = []
    info = []
    for idx in time_idxs:
        _s, _r, _i = gcr_fg_and_signal_per_time(
                idx=idx,
                vis=vis[idx],
                fg_modes=fg_modes,
                Nparams=Nparams,
                sys_model=sys_model[idx],
                flags=flags,
                Einv=Einv,
                sqrtE=sqrtE,
                Ninv=Ninv,
                sqrtNinv=sqrtNinv, 
                map_estimate=map_estimate,
                solver=solver,
                solver_tol=solver_tol,
                verbose=verbose,
                multiprocess_seed=100000
            )
        samples.append(_s)
        residuals.append(_r)
        info.append(_i)

    """
    with Pool(nproc) as pool:
        samples, residuals,  info = zip(*pool.map(
            lambda idx: gcr_fg_and_signal_per_time(
                idx=idx,
                vis=vis[idx],
                fg_modes=fg_modes,
                Nparams=Nparams,
                sys_model=sys_model[idx],
                flags=flags,
                Einv=Einv,
                sqrtE=sqrtE,
                Ninv=Ninv,
                sqrtNinv=sqrtNinv, 
                map_estimate=map_estimate,
                solver_tol=solver_tol,
                verbose=verbose,
                multiprocess_seed=100000
            ),
            time_idxs,
        )
        )
    """
    samples = np.array(samples).reshape((vis.shape[0], -1)) # Do NOT use order F
    residuals = np.array(residuals)
    info = np.array(info)

    # Return sample
    if verbose:
        print(f"{time.time() - t_start:<12.4f}", end="")
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
    Csigfft[np.diag_indices(Nfreqs)] = ps
    C = (fourier_op.T.conj() @ Csigfft @ fourier_op)
    return C


def goodness_of_fit_statistics(data, 
                               data_model, 
                               flags, 
                               Ninv, 
                               signal_amps, 
                               Sinv, 
                               include_prior=False,
                               verbose=False):
    """
    Calculate the chi^2 and log-posterior for a given model.

    Parameters:
        data (array_like):
            Array of complex visibilities for a single baseline, of shape
            `(Ntimes, Nfreqs)`.
        data_model (array_like):
            Data model to be compared with `data` (must have same shape).
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), with shape 
            `(Nfreqs,)`.
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        signal_amps (array_like):
            Signal amplitudes.
        Sinv (array_like):
            Signal covariance matrix (inverse).
        verbose (bool):
            Whether to output basic debug info.

    Returns:
        chisq (array_like):
            chi^2 value for each element on the data.

        ln_post (float):
            log posterior probability (unnormalised).
    """
    # Chi-squared is computed as the sum of ( |data - model - sys_model| / noise )^2,
    # i.e. as a sum of standard normal random variables.
    # FIXME: this will need to be changed to account for time-dependent
    # flags (i.e. when we have a different N per time).
    chisq = np.abs(data - data_model)**2 * Ninv.diagonal()[None, :]
    chisq_mean = chisq[:, flags].mean()
    chisq = chisq.real

    if verbose:
        chisq_mean = chisq[:, flags].mean()
        print(f"{chisq_mean:<9.3e}", end=" ")

    # Whether to include the prior term in ln_post
    use_prior = 0.
    if include_prior:
        use_prior = 1.

    # Log posterior; each time is treated as an independent sample, so the joint
    # ln_post for all times is the sum of the ones for each time.
    ln_post = np.sum(np.diagonal(
        -(
            (data - data_model)[:, flags].conj()
            @ Ninv[flags][:, flags]
            @ (data - data_model)[:, flags].T
        )
        #- use_prior*(
        #    signal_amps[:, flags].conj()
        #    @ Sinv[flags][:, flags]
        #    @ signal_amps[:, flags].T
        #)
    ))
    ln_post = np.real(ln_post)
    if verbose:
        print(f"{ln_post:<12.1f}")
    return chisq, ln_post


def gibbs_step(
    vis,
    flags,
    Ninv,
    signal_ps,
    signal_ps_prior,
    fg_modes,
    sys_modes,
    sys_amps,
    sys_prior,
    iter,
    sky_model=None,
    nproc=1,
    sample_systematics=True,
    sample_eor_fg=True,
    sample_signal_ps=True,
    map_estimate=False,
    solver='lgmres',
    solver_tol=1e-12,
    verbose=True
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
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        signal_ps (array_like):
            Current value of the EoR signal power spectrum.
        signal_ps_prior (array_like):
            EoR signal power spectrum prior.
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        sys_modes (array_like):
            Systematics mode matrix.
        sys_amps (array_like):
            Systematics coefficients from the previous iteration. Shape `(Nsys_modes,)`.
        sys_prior (array_like):
            Systematic coefficient prior covariance matrix, of shape 
            `(Nsys_modes, Nsys_modes)`.
        iter (int):
            Nth Gibbs sampler iteration (for plotting)
        sky_model (array_like):
            Sky model to use if the signal + FG GCR sampling step is switched off. 
            Otherwise, it will be overwritten in the first conditional sampling step.
        nproc (int):
            Number of processes to use for parallelised functions.
        sample_systematics (bool):
            Whether to sample systematics model parameters or keep them fixed.
        sample_signal_ps (bool):
            Whether to sample the signal power spectrum.
        map_estimate (bool):
            Provide the maximum a posteriori sample.
        solver_tol (float):
            Tolerance `tol` for scipy linear solvers.
        verbose (bool):
            If True, output basic timing stats about each iteration.

    Returns:
        signal_amps (array_like):
            Samples of the signal, shape `(Ntimes, Nfreqs)`.
        ps_sample (array_like):
            Sample of the signal power spectrum bandpowers, shape `(Nfreqs,)`.
        fg_amps (array_like):
            Sample of the foreground amplitudes, shape `(Nmodes,)`.
        sys_amps (array_like):
            Array of systematics amplitudes of shape (len(nm_list))
    """
    # Shape of data and operators
    Ntimes = vis.shape[0]
    Nfreqs = vis.shape[1] 
    Nfg_modes = fg_modes.shape[1]
    Nparams = Nfreqs + Nfg_modes
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"

    # Precompute 2D Fourier operator matrix
    fourier_op = utils.fourier_operator(Nfreqs)

    # Precompute current systematics model
    # Note: Be very careful which order this is reshaped!
    sys_model = (1. + (sys_modes @ sys_amps).reshape((Nfreqs, Ntimes)).T)  # Do NOT use order F
    if sample_eor_fg:
        # (1) Sample signal and foreground amplitudes using GCR
        cr = gcr_fg_and_signal(
                        vis=vis, 
                        fg_modes=fg_modes, 
                        Nparams=Nparams, 
                        sys_model=sys_model, 
                        flags=flags, 
                        signal_ps=signal_ps, 
                        Ninv=Ninv,
                        fourier_op=fourier_op, 
                        nproc=nproc, 
                        map_estimate=map_estimate,
                        solver=solver,
                        solver_tol=solver_tol,
                        verbose=verbose)   #Running test on the d=(1+delta g)s+n form of the equations 
        
        # Extract separate signal and FG parts from the solution
        signal_amps = cr[:, :-Nfg_modes]
        fg_amps = cr[:, -Nfg_modes:]
        
        # Update sky model (without multiplicative systematics); sum of EoR + FG model
        sky_model = (signal_amps + fg_amps @ fg_modes.T)
    else:
        sky_model = sky_model

    #import pylab as plt
    #plt.matshow(sky_model.real)
    #plt.colorbar()
    #plt.show()


    # (2) Sample multiplicative systematics parameters
    if sample_systematics:
        sys_amps = sys_sol.gcr_systematics(
                                    data=vis,
                                    Ninv=Ninv,
                                    sky_model=sky_model, 
                                    sys_modes=sys_modes,
                                    sys_prior=sys_prior, 
                                    verbose=verbose
                                    )
    
    # (3) Sample EoR signal power spectrum (and also convert to signal covariance matrix)
    if sample_signal_ps:
        signal_ps_sample = sample_pspec(s=signal_amps, prior=signal_ps_prior)

        # No need for factor of 1/Nfreqs**2 here as sample_pspec() changed to iFFT normalization
        Sinv_sample = covariance_from_pspec(1. / signal_ps_sample, fourier_op) #/ Nfreqs**2. # note FFT norm
    else:
        signal_ps_sample = signal_ps
        Sinv_sample = 0.
        

    # Calculate goodness of fit statistics
    chisq, ln_post = goodness_of_fit_statistics(
                                    data=vis, 
                                    data_model=sys_model * sky_model, 
                                    flags=flags, 
                                    Ninv=Ninv, 
                                    signal_amps=signal_amps, 
                                    Sinv=Sinv_sample, 
                                    verbose=verbose)
    
    # Return samples
    return signal_amps, signal_ps_sample, fg_amps, sys_amps, chisq, ln_post 


def gibbs_sample(
    vis,
    flags,
    Ninv,
    freqs,
    lsts,
    signal_ps_initial,
    signal_ps_prior,
    fg_modes,
    sys_modes,
    sys_prior,
    sys_initial,
    sky_model_initial=None,
    Niter=100,
    seed=None,
    sample_systematics=True,
    sample_eor_fg=True,
    sample_signal_ps=True,
    solver='lgmres',
    solver_tol=1e-12,
    verbose=True,
    nproc=1,
    write_Niter=100,
    out_dir=None,
    map_estimate=False
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
        Ninv (array_like):
            Inverse noise variance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        freqs:
            Frequency array (Nfreqs,)
        lsts:
            Time array in LSTS (Ntimes,)
        signal_ps_initial (array_like):
            Initial guess for the EoR signal power spectrum. A better guess 
            should result in faster convergence.
        signal_ps_prior (array_like):
            EoR signal power spectrum prior, or shape (2, Nfreqs). `ps_prior[0]` 
            contains the lower bound of the prior, `ps_prior[1]` the upper bound. 
        fg_modes (array_like):
            Foreground mode array, of shape `(Nfreqs, Nmodes)`. This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        sys_modes (array_like):
            Systematics mode array, of shape `(Nfreqs * Ntimes, Nsysmodes)`.
        sys_prior (array_like):
            Prior covariance for the systematic amplitudes, of shape 
            `(Nsysmodes, Nsysmodes)` .
        sys_initial (array_like):
            Initial guess of systematics parameters.
        Niter (int):
            Number of iterations of the sampler to run.
        seed (int):
            Random seed to use for random parts of the sampler.
        solver_tol (float):
            Tolerance `tol` for scipy linear solvers.
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
        

    Returns:
        signal_amps (array_like):
            Samples of the signal, shape `(Niter, Ntimes, Nfreqs)`.
        signal_ps (array_like):
            Sample of the signal power spectrum bandpowers, shape
            `(Niter, Nfreqs)`.
        fg_amps (array_like):
            Samples of the foreground amplitudes, shape `(Niter, Nmodes)`.
        sys_amps (array_like):
            Sample of systematics coefficient vectors (Niter, number of systematics modes)
        chisq (array_like):
            Chi-squared value per iteration, shape `(Niter, Ntimes, Nfreqs)`.
        ln_post (array_like):
            Natural log of the posterior probability per iteration, shape
            `(Niter,)`.
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
    Nsys_modes = sys_modes.shape[-1]
    assert sys_prior.shape[0] == sys_prior.shape[1] \
        == sys_initial.shape[0] == sys_modes.shape[-1], \
        "sys_modes, sys_prior, and sys_initial must have the same number of modes"
    assert sys_modes.shape[0] == Ntimes * Nfreqs, \
        "sys_modes must have shape (Ntimes * Nfreqs, Nsysmodes)"
    assert flags.shape == (Nfreqs,), "`flags` array must have shape (Nfreqs,)"
    assert fg_modes.shape[0] == Nfreqs, "fgmodes must have shape (Nfreqs, Nmodes)"
    assert signal_ps_prior.shape == (2, Nfreqs), "ps_prior must have shape (2, Nfreqs)"
    if len(Ninv.shape) == 3:
        assert (
            Ninv.shape[0] == Ntimes
        ), "Ninv shape must be (Ntimes, Nfreqs, Nfreqs) or (Nfreqs, Nfreqs)"
    
    # Check for sensible initial power spectrum
    assert np.all( np.logical_and(signal_ps_initial >= signal_ps_prior[0,:],
                                  signal_ps_initial <= signal_ps_prior[1,:]) ), \
           "Initial power spectrum ps_initial is not within ps_prior range."

    # Set up arrays for sampling
    signal_amps = np.zeros((Niter, Ntimes, Nfreqs), dtype=complex)
    signal_ps = np.zeros((Niter, Nfreqs))
    fg_amps = np.zeros((Niter, Ntimes, Nmodes), dtype=complex)
    sys_amps = np.zeros((Niter, Nsys_modes), dtype=complex)
    
    # Debugging statistics
    chisq = np.zeros((Niter, Ntimes, Nfreqs))
    ln_post = np.zeros(Niter)
    
    # Set initial values the signal power spectrum and systematics amplitudes
    signal_ps_current = signal_ps_initial
    sys_amps_current = sys_initial

    # Loop over iterations
    if verbose:
        print("Iter     Time [s]    Info    |Ax - b|    T_Sys(s)    Sys Info    Sys |Ax-b|    Chisq    ln Post")
        print("-----    --------    ----    --------    --------    --------    ----------    -----    -------")

    for i in range(Niter):
        if verbose:
            print(f"{i+1:<9d}", end="")

        # Do Gibbs iteration
        signal_amps[i], signal_ps[i], fg_amps[i], sys_amps[i], chisq[i], ln_post[i] \
            = gibbs_step(
                vis=vis * flags,
                flags=flags,
                Ninv=Ninv,
                signal_ps=signal_ps_current,
                signal_ps_prior=signal_ps_prior,
                fg_modes=fg_modes,
                sys_prior=sys_prior,
                sys_modes=sys_modes,
                sys_amps=sys_amps_current,
                sky_model=sky_model_initial,
                nproc=nproc,
                iter=i,
                map_estimate=map_estimate,
                solver=solver,
                solver_tol=solver_tol,
                sample_systematics=sample_systematics,
                sample_eor_fg=sample_eor_fg,
                sample_signal_ps=sample_signal_ps,
                verbose=verbose
            )

        # Update signal PS and systematics
        signal_ps_current = signal_ps[i]
        sys_amps_current = sys_amps[i]
        utils.append_gibbs_sample_h5(
            fp=out_dir,
            overwrite=(i == 0),          # truncate on the very first call
            signal_amps=signal_amps[i],
            signal_ps=signal_ps[i],
            fg_amps=fg_amps[i],
            sys_amps=sys_amps[i],
            chisq=chisq[i],
            ln_post=ln_post[i] # scalar is fine
        )
        
        if out_dir is not None and (i+1) % write_Niter == 0:
            # Write current set of samples to disk
            utils.write_numpy_files(
                out_dir,
                signal_amps[:i+1],
                signal_ps[:i+1],
                fg_amps[:i+1],
                sys_amps[:i+1],
                chisq[:i+1],
                ln_post[:i+1]
            )
    if out_dir is not None and Niter % write_Niter > 0:
        # Write all samples to disk
        utils.write_numpy_files(
            out_dir,
            signal_amps,
            signal_ps,
            fg_amps,
            sys_amps,
            chisq,
            ln_post
        )

    if verbose:
        print()

    return signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post
