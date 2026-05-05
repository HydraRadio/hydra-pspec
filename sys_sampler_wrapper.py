"""
sys_sampler_wrapper.py
----------------------
Wrapper script for running the Hydra-pspec Gibbs sampler with a multiplicative
systematics model. Loads or generates EoR and foreground visibilities, builds
a systematics gain model, defines priors, and runs the sampler.

Usage
-----
    python sys_sampler_wrapper.py

Output
------
    Sampler products written to `op_dir` (see Configuration section).
"""

import time
import sys

import numpy as np
import scipy.special
from pyuvdata import UVData
from astropy import units
import matplotlib.ticker as ticker
import cmcrameri.cm as cmc
import hydra_pspec as hp


# =============================================================================
# Timing
# =============================================================================
start_t = time.time()

with open('res/hydra_ascii.txt', 'r') as f:
    print(f.read())


# =============================================================================
# Configuration
# =============================================================================
Ntimes   = 80
Nfreqs   = 60
Nfgmodes = 10
Niter    = 5

# Set to True to draw EoR from a Gaussian random field;
# False to load the Burba et al. simulated EoR.
dummy_flag = False

np.random.seed(11)

# Output directory for sampler products
op_dir = './tests'

# Systematics mode pairs (delay index n, fringe-rate index m)
# Case I  : nm_list = [(3,0),  (4,0),  (5,0),  (6,0)]
# Case II : nm_list = [(10,0), (11,0), (12,0), (13,0)]
# Case III: nm_list = [(3,20), (4,20), (5,20), (6,20)]
nm_list = [(10, 0), (11, 0), (12, 0), (13, 0)]   # Case II

# True systematics amplitudes
sys_amps_true = np.array([1. + 4j, 2. + 3j, 3. + 2j, 4. + 1j])

# Noise power spectrum amplitude
noise_ps_val = 0.0004


# =============================================================================
# Helper functions
# =============================================================================
def calc_ps(s):
    """
    Compute the delay power spectrum of visibility data.

    Uses an inverse FFT normalisation to match the Hydra-pspec convention.

    Parameters
    ----------
    s : ndarray, shape (Ntimes, Nfreqs)
        Visibility data (real or complex).

    Returns
    -------
    ps : ndarray, shape (Nfreqs,)
        Time-averaged delay power spectrum.
    """
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs_ = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / Nfreqs_


# =============================================================================
# Frequency and LST grids
# =============================================================================
freqs = np.linspace(100., 120., 120)[:Nfreqs]   # MHz
lsts  = np.linspace(0., 1., Ntimes)

print(f"Ntimes={Ntimes}, Nfreqs={Nfreqs}, Nfgmodes={Nfgmodes}")
print(f"NM list: {nm_list}")


# =============================================================================
# Systematics model
# =============================================================================
sys_modes = hp.sys_solver.sys_modes(
    freqs_Hz   = freqs * 1e6,
    times_sec  = lsts * 24. / (2. * np.pi) * 3600.,
    modes      = nm_list,
)
sys_prior = 100.**2 * np.eye(sys_amps_true.size)

gain_true = (1. + (sys_modes @ sys_amps_true).reshape([Nfreqs, Ntimes]).T)
np.save(op_dir + '/gain_true.npy', gain_true)


# =============================================================================
# EoR field and power spectrum
# =============================================================================
fourier_op = hp.utils.fourier_operator(Nfreqs, unitary=True)

if dummy_flag:
    ps_true = 0.0012 * (1. + 0.3 * np.sin(3. * np.linspace(0., 1., Nfreqs)))
    S_true  = hp.pspec.covariance_from_pspec(ps_true, fourier_op)

    sqrt_S_true = np.linalg.cholesky(S_true)
    eor_true = (
        sqrt_S_true
        @ (np.random.randn(Nfreqs, Ntimes) + 1.j * np.random.randn(Nfreqs, Ntimes))
        / np.sqrt(2.)
    ).T
else:
    eor_true = np.load('res/npy_data/eor_true.npy')
    S_true   = np.load('res/test_data/eor-cov.npy')
    lsts     = np.load('res/npy_data/lsts_full.npy')[:Ntimes]
    freqs    = np.load('res/npy_data/freqs_full.npy')[:Nfreqs] * 10e-6
    eor_true = eor_true[:Ntimes, :Nfreqs]
    ps_true  = calc_ps(eor_true)

np.save(op_dir + '/eor_true.npy', eor_true)
print(f"EoR shape: {eor_true.shape}")


# =============================================================================
# Foregrounds
# =============================================================================
fg_true = np.load('res/npy_data/fg_true.npy')[:Ntimes, :Nfreqs]

fgmodes = np.array([
    scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
    for i in range(Nfgmodes)
]).T

print(f"FG modes shape: {fgmodes.shape}")

np.save(op_dir + '/fgmodes.npy', fgmodes)
np.save(op_dir + '/fg_true.npy', fg_true)


# =============================================================================
# Priors
# =============================================================================
ps_prior = np.column_stack((
    1e-7 * np.ones(Nfreqs),
    1e-1 * np.ones(Nfreqs),
))
ps_sample = hp.pspec.sample_pspec(s=eor_true, prior=ps_prior)
print(f"PS sample shape: {ps_sample.shape}")

S_sample    = hp.pspec.covariance_from_pspec(ps_sample, fourier_op)
Sinv_sample = hp.pspec.covariance_from_pspec(1. / ps_sample, fourier_op)


# =============================================================================
# Noise
# =============================================================================
noise_ps_true = noise_ps_val * np.ones(Nfreqs)
N_true = hp.pspec.covariance_from_pspec(noise_ps_true, fourier_op)
Ninv   = np.diag(1. / np.diag(N_true))
n      = (
    np.sqrt(N_true)
    @ (np.random.randn(freqs.size, Ntimes) + 1.j * np.random.randn(freqs.size, Ntimes))
    / np.sqrt(2.)
)


# =============================================================================
# Data
# =============================================================================
d = gain_true * (fg_true + eor_true) + n.T
np.save(op_dir + '/data_true.npy', d)


# =============================================================================
# Run the Gibbs sampler
# =============================================================================
signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post = hp.pspec.gibbs_sample(
    vis                = d,
    flags              = np.ones((len(freqs),), dtype=int),
    signal_ps_initial  = ps_true,
    fg_modes           = fgmodes,
    Ninv               = Ninv,
    signal_ps_prior    = ps_prior.T,
    Niter              = Niter,
    seed               = 10,
    freqs              = freqs,
    lsts               = np.linspace(0., 1., Ntimes),
    map_estimate       = False,
    verbose            = True,
    nproc              = 1,
    write_Niter        = Niter,
    out_dir            = op_dir,
    sys_modes          = sys_modes,
    sys_prior          = sys_prior,
    sys_initial        = sys_amps_true,
    solver_tol         = 1e-13,
    sample_systematics = True,
    sample_eor_fg      = True,
    sample_signal_ps   = True,
    sky_model_initial  = fg_true + eor_true,
)

print(f"Total time taken: {time.time() - start_t:.1f}s")