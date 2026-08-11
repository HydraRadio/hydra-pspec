#!/usr/bin/env python

from mpi4py import MPI

import numpy as np
import hydra_pspec as hp
import scipy as sp
import scipy.special
import time


def gibbs_sample_multiple_baselines(
    vis,
    flags,
    inv_var,
    fg_modes,
    mode_index_map,
    Niter=100,
    seed=None,
    verbose=True,
    map_estimate=False,
    preconditioner=None,
    linear_solver=sp.sparse.linalg.cg,
    comm=None
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
            `(Nbls, Ntimes, Nfreqs)`.
        flags (array_like):
            Array of flags (1 for unflagged, 0 for flagged), of shape
            `(Nbls, Ntimes, Nfreqs)`.
        inv_var (array_like):
            Inverse noise variance, of shape `(Nbls, Ntimes, Nfreqs)`. Assumed 
            to be the same for the real and imaginary parts.
        fg_modes (array_like):
            Foreground mode array, of shape (Nfreqs, Nmodes). This should be
            derived from a PCA decomposition of a model foreground covariance
            matrix or similar.
        mode_index_map (list):
            List of per-baseline index arrays mapping delay modes to 1D k 
            modes. See `utils.map_to_spherical_kbins()` for more information.
        ps_initial (array_like):
            Initial guess for the EoR signal power spectrum.
        ps_prior_bounds (array_like):
            EoR signal power spectrum prior bounds.
        ps_modes_per_bin (int):
            Number of delay modes in each delay power spectrum bin.
        Niter (int):
            Number of iterations of the sampler to run.
        seed (int):
            Random seed to use for random parts of the sampler.
        verbose (bool):
            If True, output basic timing stats about each iteration.
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
    """
    # Get MPI info
    num_workers = comm.Get_size()
    myid = comm.Get_rank()
    
    if map_estimate:
        Niter = 1
    else:
        # Set random seed (the constant is a prime number)
        if seed is None:
            seed = np.random.randint(0, 100000)
        np.random.seed(seed + 999331 * myid)
    
    # Check shape of data/foreground modes
    assert flags.shape == vis.shape, "`flags` array must have the same shape as `vis`"
    assert inv_var.shape == vis.shape, "`inv_var` array must have the same shape as `vis`"
    
    # Expand 2D arrays to 3D if they are only for a single baseline
    if len(vis.shape) == 2:
        vis = vis[np.newaxis,:,:]
        flags = flags[np.newaxis,:,:]
        inv_var = inv_var[np.newaxis,:,:]
    
    # Get array shapes
    Nbls, Ntimes, Nfreqs = vis.shape
    Nmodes = fg_modes.shape[1]
    Nparams = Nfreqs + Nmodes
    assert fg_modes.shape[0] == Nfreqs, "fg_modes must have shape (Nfreqs, Nmodes)"
    
    # Set up division of baselines between workers
    if num_workers > Nbls:
        raise ValueError("Must have no. baselines >= no. workers")
    baseline_split = np.array_split(np.arange(Nbls), num_workers)
    
    # Precompute Fourier operator
    fourier_op = hp.utils.fourier_operator(Nfreqs)    
        
    ##################
    # Check prior bounds
    Nkbins = mode_index_map[0].size #ps_initial.size
    ps_prior_min = 1e-2 * np.ones(Nkbins) # FIXME: Should be user-specified
    ps_prior_max = 1e2 * np.ones(Nkbins)
    ps_prior_bounds = (ps_prior_min, ps_prior_max)
    
    # Set up arrays for sampling
    my_Nbls = len(baseline_split[myid]) # no. baselines belonging to current worker
    signal_cr = np.zeros((Niter, my_Nbls, Ntimes, Nfreqs), dtype=complex)
    fg_amps = np.zeros((Niter, my_Nbls, Ntimes, Nmodes), dtype=complex)
    signal_ps = np.zeros((Niter, Nkbins))
    inv_gamma_Ndof = np.zeros(Nbls)
    inv_gamma_beta = np.zeros((Nbls, Nfreqs))
    
    # Useful debugging statistics
    #chisq = np.zeros((Niter, my_Nbls, Ntimes, Nfreqs))
    #ln_post = np.zeros(Niter)
    
    # Output mapping between baselines and workers
    if verbose:
        if myid == 0:
            print("\nWorker <-> baseline mapping\n---------------------------", 
                  flush=True)
        comm.barrier()
        time.sleep(0.01 * myid) # Wait for time defined by MPI rank
        print("Worker %5d: %s" % (myid, str(baseline_split[myid])), flush=True)
        time.sleep(0.1) # give buffers chance to flush
        comm.barrier()
    
    # Loop over iterations
    if verbose and myid == 0:
        print("\nWorker  Iter   Time [s]    Info    |Ax - b|    Chisq")
        print("------  ----   --------    ----    --------    -----", 
              flush=True)
    
    # Barrier to wait for header to be printed
    comm.barrier()
    
    # Set initial delay PS (FIXME)
    ps_delay = 10. * np.ones((Nbls, Nfreqs))
    
    for i in range(Niter):

        # Reset inverse gamma parameter values for this iteration
        # (This shouldn't really be needed as they should all be overwritten)
        inv_gamma_Ndof *= 0
        inv_gamma_beta *= 0.
        
        # Loop over baselines assigned to this worker and do GCR sample for each
        for j, bl in enumerate(baseline_split[myid]):
            
            if verbose:
                print(f"{myid:<7d} {i+1:<7d}", end="")
            
            # Precompute covariance matrix from current PS
            sig_S, sig_S_sqrt = hp.pspec.covariance_from_pspec(
                                                      ps=ps_delay[bl], 
                                                      fourier_op=fourier_op, 
                                                      modes_per_bin=1, 
                                                      mode='both')
            
            # Bundle operator matrices together
            matrices = {
                'S':        sig_S,                     # S, signal covariance
                'Sh':       sig_S_sqrt,                # Sh, sqrt of signal covariance
                'Ninv':     flags[bl] * inv_var[bl],     # inverse variance with flags
                'Nih':      flags[bl] * np.sqrt(inv_var[bl]) # sqrt inverse variance
            }
                
            # (1) Solve GCR equation to get EoR signal and FG amp. realisations
            cr = hp.pspec.gcr_fg_modes(
                                     vis=vis[bl], 
                                     matrices=matrices, 
                                     fg_modes=fg_modes, 
                                     map_estimate=map_estimate, 
                                     preconditioner=preconditioner,
                                     linear_solver=linear_solver,
                                     verbose=verbose
                    )

            # Extract separate signal and FG parts from the solution
            # cr has shape (Ntimes, Nfreqs + Nmodes)
            signal_cr[i,j] = cr[:, :-fg_modes.shape[1]]
            fg_amps[i,j] = cr[:, -fg_modes.shape[1]:]
            
            # Full model of data is sum of EoR (GCR) + FG model
            model = signal_cr[i,j] + fg_amps[i,j] @ fg_modes.T

            # Chi-squared is computed as the sum of ( |data - model| / noise )^2,
            # i.e. as a sum of standard normal random variables
            chisq = np.abs(vis[bl] - model)**2 * inv_var[bl]
            if verbose:
                chisq_mean = chisq[np.bool(flags[bl])].mean()
                if chisq_mean > 10:
                    print(f"{chisq_mean:<9.1e}", end="")
                else:
                    print(f"{chisq_mean:<9.3f}", end="")
            
            # Calculate inverse gamma distribution parameters for this baseline
            inv_gamma_Ndof[bl], inv_gamma_beta[bl] \
                = hp.pspec.calculate_inv_gamma_params(s=signal_cr[i,j])
            
            # Finish outputting diagnostic info; flush buffer in rank order
            if verbose:
                comm.barrier()
                time.sleep(0.01 * myid) # Wait for time defined by MPI rank
                print(flush=True)
            
        # (2) Draw power spectrum samples and return per-baseline delay spectrum
        ps_1d, ps_delay = hp.pspec.sample_pspec_multiple_baselines(
                                                mode_index_map=mode_index_map,
                                                Ndof=inv_gamma_Ndof,
                                                beta=inv_gamma_beta,
                                                prior_bounds=ps_prior_bounds,
                                                comm=comm
                                                )
        
        # FIXME: Should store results!
        
        # Wait for other workers to complete this iteration
        comm.barrier()

    return signal_cr, signal_ps, fg_amps, ps_1d
    
    

if __name__ == '__main__':
    
    # Set up MPI
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()
    myid = comm.Get_rank()
    
    # Define data shape and size
    np.random.seed(10)
    freqs_MHz = np.linspace(100., 110., 32)
    lsts = np.linspace(0., 1., 15)
    k_bin_edges = np.linspace(0., 10., 12)
    
    Nfreqs = freqs_MHz.size
    Ntimes = lsts.size
    Nbls = 6
    
    # Define FG modes
    Nmodes = 2
    fg_modes = np.array([scipy.special.legendre(i)(np.linspace(-1., 1., Nfreqs)) 
        				   for i in range(Nmodes)]).T
    
    # Random data
    vis = np.random.randn(Nbls, Ntimes, Nfreqs) \
        + 1.j * np.random.randn(Nbls, Ntimes, Nfreqs)
    vis += (10.*np.ones(Nmodes)) @ fg_modes.T
    flags = np.ones(vis.shape, dtype=int)
    inv_var = np.ones(vis.shape)
    
    # Define delay bins and baselines
    centre_freq_MHz = freqs_MHz[freqs_MHz.size // 2]
    tau_ns = hp.utils.tau_bins(freqs_MHz, modes_per_bin=1)
    bl_metres = np.array([14.2 + np.sqrt(14.2)*i for i in range(Nbls)])
    
    # Calculate mapping between baselines/delays and spherical k bins
    mode_index_map = hp.utils.map_to_spherical_kbins(
                                            k_bin_edges=k_bin_edges, 
                                            centre_freq_MHz=centre_freq_MHz, 
                                            bl_metres=bl_metres, 
                                            tau_ns=tau_ns, 
                                            h=0.67, 
                                            Omega_m=0.3)
    
    # Run Gibbs sampler
    signal_cr, signal_ps, fg_amps, ps_1d = gibbs_sample_multiple_baselines(
                                                    vis=vis,
                                                    flags=flags,
                                                    inv_var=inv_var,
                                                    fg_modes=fg_modes,
                                                    mode_index_map=mode_index_map,
                                                    Niter=40,
                                                    comm=comm
                                                    )
