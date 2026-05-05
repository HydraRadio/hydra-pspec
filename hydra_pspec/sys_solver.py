# Function definitions for GCR solver

import numpy as np
from math import pi as pi
import matplotlib.pylab as plt
from scipy.linalg import fractional_matrix_power as fmp
from sklearn.metrics import *
import scipy.linalg as sl
import scipy 
import time
from .plotting_functions import master_plotter


def fourier_mode_2d(freqs_Hz, times_sec, modes, box=None):
    """
    Construct a set of 2D Fourier modes from a list of wavenumber integers, 
    to form an incomplete set of 2D Fourier modes.

    Parameters
    ----------
    freqs_Hz (array_like):
        Frequency array, in Hz. Should be ordered.
        
    times_sec (array_like):
        Time array, in hours. Should be ordered.

    modes (list of tuple of int):
        List of mode integer pairs to include in operator.

    box (tuple of tuple):
        NOT IMPLEMENTED
        Keep all modes within a box, defined by the tuple:
        `((delay_min, delay_max), (frate_min, frate_max))`.
        The delays are in ns and the fringe rates in mHz.
    """
    Nfreqs, Ntimes = freqs_Hz.size, times_sec.size
    
    # Get grid spacing in expected units
    dfreq = (freqs_Hz[1] - freqs_Hz[0])
    dtime = (times_sec[1] - times_sec[0])

    # Get FFT wavenumbers
    kfreq = np.fft.fftfreq(Nfreqs, d=dfreq) # sec #* 1e9 # ns
    ktime = np.fft.fftfreq(Ntimes, d=dtime) # Hz * 1e3 # mHz

    # Get FFT mode integers
    nfreq = (np.fft.fftfreq(Nfreqs) * Nfreqs).astype(int)
    ntime = (np.fft.fftfreq(Ntimes) * Ntimes).astype(int)

    # Frequency/time grids with respect to origin
    f = freqs_Hz - freqs_Hz[0]
    t = times_sec - times_sec[0]

    # Get indices of modes we want to keep
    basis_fns = np.zeros((len(modes), Nfreqs, Ntimes), dtype=np.complex128)
    for i, mode in enumerate(modes):
        nf, nt = mode
        print(nf, nt)
        assert isinstance(nf, int), "modes must only contain pairs of integers"
        assert isinstance(nt, int), "modes must only contain pairs of integers"
        assert nf in nfreq, "Delay mode nf=%d not in available range (%d -- %d)." \
            % (nf, nfreq.min(), nfreq.max())
        assert nt in ntime, "Fringe rate mode nt=%d not in available range (%d -- %d)." \
            % (nt, ntime.min(), ntime.max())

        # Get mode indices
        idx_f = np.where(nfreq == nf)[0][0]
        idx_t = np.where(ntime == nt)[0][0]
        #mode_idxs.append( (idx_f, idx_t) )

        # print(kfreq[idx_f], ktime[idx_t])

        # Add basis function to operator
        basis_fns[i] = np.exp(2.*np.pi*1.j * (  kfreq[idx_f] * f[:,np.newaxis]
                                     + ktime[idx_t] * t[np.newaxis,:] ) ) \
                     / np.sqrt(Nfreqs * Ntimes)
        
    return basis_fns, kfreq * 1e9, ktime * 1e3


def sys_modes(freqs_Hz, times_sec, modes):
    """
    Construct systematic mode operator, which is a 2D Fourier basis.
    note: mode 0 is DL, mode 1 is FR
    """
    u, kfreq, ktime = fourier_mode_2d(freqs_Hz=freqs_Hz, 
                                      times_sec=times_sec, 
                                      modes=modes)
    return u.reshape((u.shape[0],-1)).T  #DO NOT CHANGE RESHAPING. No good, very bad things happen.   


def sq_mat_tr(A,flag='r'):
    '''
    Convert A_mat from the GCR equation into a square matrix for linear system solver.(Method 1)
    Original matrix A has shape of (2,2,n,n). This returns a matrix with shape 2*n,2*n

    Parameters:
        A: Matrix of shape (2,2,n,n)

    Returns:
        reshaped_A: Matrix A reshaped to (2*n,2*n)
    '''
    sh=np.shape(A)
    if flag=='c':
        reshaped_A = np.array(A).transpose(0, 3, 1, 2).reshape(sh[0] * sh[2], sh[1] * sh[3]) # Hardcoding to order F, DO NOT CHANGE
    elif flag=='r':
        reshaped_A = np.array(A).transpose(0, 2, 1, 3).reshape(sh[0] * sh[2], sh[1] * sh[3]) # Hardcoding to order F, DO NOT CHANGE
    return reshaped_A

def sq_mat_tr2(your_mat):
    '''
    Convert A_mat from the GCR equation into a square matrix for linear system solver.(Method 2)
    Original matrix your_mat has shape of (2,2,n,n). This returns a matrix with shape 2*n,2*n
    
    Parameters:
        A: Matrix of shape (2,2,n,n)
        
    Returns:
        reshaped_A: Matrix A reshaped to (2*n,2*n)
    '''
    sh=np.shape(your_mat)
    your_mat=np.array(your_mat)
    N=sh[2]
    square_mat = np.zeros((2*N, 2*N), dtype=your_mat.dtype) # empty matrix of the right type

    square_mat[:N,:N] = your_mat[0,0,:,:]
    square_mat[:N,N:] = your_mat[0,1,:,:]
    square_mat[N:,:N] = your_mat[1,0,:,:]
    square_mat[N:,N:] = your_mat[1,1,:,:]
    return square_mat

def inv_mat(mat):
    '''
    Inverses an invertible diagonal matrix mat without using np.linalg.inv()
    
    Paramters:
        mat: invertible matrix
    
    Returns:
        mat_inv: mat inverted
    '''
    mat=np.array(mat)
    diag_el=np.diag(mat)
    diag_inv=1/diag_el
    mat_inv=np.zeros(np.shape(mat))
    np.fill_diagonal(mat_inv,diag_inv)
    return mat_inv

def cholesky_inverse(A):
    """
    Inverts a positive-definite matrix A using Cholesky decomposition.
    
    Args:
    - A: A positive-definite matrix
    
    Returns:
    - A_inv: The inverse of matrix A
    """
    # Ensure A is a NumPy array
    A = np.array(A)
    
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(A)
    
    # Solve L * L.T = A for A^-1 using forward and backward substitutions
    
    # Step 1: Solve L * y = I for y using forward substitution
    n = A.shape[0]
    I = np.eye(n)
    y = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            temp_sum = sum(L[i, k] * y[k, j] for k in range(i))
            y[i, j] = (I[i, j] - temp_sum) / L[i, i]
    
    # Step 2: Solve L.T * x = y for x using backward substitution
    L_T = L.T
    A_inv = np.zeros_like(A)
    for i in range(n-1, -1, -1):
        for j in range(n):
            temp_sum = sum(L_T[i, k] * A_inv[k, j] for k in range(i+1, n))
            A_inv[i, j] = (y[i, j] - temp_sum) / L_T[i, i]
    
    return A_inv

def gcr_systematics(data,
                    Ninv,
                    sky_model, 
                    sys_modes,
                    sys_prior, 
                    solver_tol=1e-12,
                    verbose=False):
    """
    Gaussian constrained realisation sampler for a multiplicative 
    systematics model.

    Parameters:
        data (array_like):
            Data visibilities. This is the residual after subtracting off the 
        Ninv (array_like):
            Inverse of noise covariance matrix. This can either have shape
            `(Ntimes, Nfreqs, Nfreqs)`, one for each time, or can be a common
            one for all times with shape `(Nfreqs, Nfreqs)`.
        sky_model (array_like):
            Sky model (Nfreqs, Ntimes)
        sys_modes (array_like):
            Systematics basis functions with shape `(Nfreqs*Ntimes, Nsys_modes)`.
        sys_prior (array_like):
            Systematic coefficient prior covariance matrix, of shape 
            `(Nsys_modes, Nsys_modes)`.
        solver_tol (float):
            Tolerance for the scipy linear solvers.
        verbose: Bool
            Verbosity of printing results
        
    Returns:
        sys_amps: array_like
            Sampled vector of systematics coefficients. Shape `(Nsys_modes,)`.
    """
    if verbose:
        t_start = time.time()
    
    Ntimes, Nfreqs= data.shape
    Nsys_modes = sys_modes.shape[1]

    # Get the data residual r = d - sky_model = g sky_model, since our 
    # parametrisation is d = (1 + g) sky_model.
    resid = data - sky_model

    # Flatten the data for operation
    # FIXME: Do we need to specify F ordering?
    r = resid.flatten(order='F')
    s = sky_model.flatten(order='F')
    

    # Invert prior covariance. This will be used to implement a block diagonal 
    # inverse prior covariance, [[Binv, 0],[0, Binv]] (for real/imag split linear system)
    Binv = np.linalg.pinv(sys_prior) # pseudo-inverse
    
    diag_el = Ninv[0,0]
    Ninv = diag_el * np.ones(shape=Ntimes*Nfreqs, dtype=complex)
    sqrtNinv = np.sqrt(Ninv)

    # Complex Gaussian vectors with unit variance for fluctuations
    omega_d_re = np.random.normal(size=(Nfreqs*Ntimes)) / np.sqrt(2) # Real part
    omega_d_im = np.random.normal(size=(Nfreqs*Ntimes)) / np.sqrt(2) # Imaginary part
    
    # Construct the M_tilde sub-matrix
    sqrtNinv_s_re = sqrtNinv * s.real # N^-1/2 * s.real
    sqrtNinv_s_im = sqrtNinv * s.imag # N^-1/2 * s.imag
    m11 = sqrtNinv_s_re[:,np.newaxis] * sys_modes.real \
        - sqrtNinv_s_im[:,np.newaxis] * sys_modes.imag
    m12 = -1. * sqrtNinv_s_re[:,np.newaxis] * sys_modes.imag \
          -1. * sqrtNinv_s_im[:,np.newaxis] * sys_modes.real

    #M_tilde = np.zeros((2*))
    M_tilde = np.concatenate((np.concatenate((m11, m12), axis=1), 
                              np.concatenate((-1*m12, m11), axis=1)),
                             axis=0) 

    # Construct A matrix
    A = M_tilde.conj().T @ M_tilde
    A[:Nsys_modes, :Nsys_modes] += Binv
    A[Nsys_modes:, Nsys_modes:] += Binv
    
    nih_dre = sqrtNinv * r.real #N^-1/2 * resid.real
    nih_dim = sqrtNinv * r.imag #N^-1/2 * resid.imag

    # Multiplying gaussian fluctuations
    sqrtNinv_s_re = sqrtNinv_s_re * omega_d_re  
    sqrtNinv_s_im = sqrtNinv_s_im * omega_d_im

    # Construct b vector (blocks for the real and imaginary parts)
    b_re = m11.T @ nih_dre \
         - m12.T @ nih_dim \
         + (sys_modes.real.T @ sqrtNinv_s_re) \
         + (sys_modes.imag.T @ sqrtNinv_s_im)
    b_im = m12.T @ nih_dre \
         + m11.T @ nih_dim \
         - (sys_modes.imag.T @ sqrtNinv_s_re) \
         + (sys_modes.real.T @ sqrtNinv_s_im)
    b = np.concatenate((b_re, b_im), axis=0)
    
    # Try to construct preconditioner from pseudo-inverse
    Ainv_estimate = np.linalg.pinv(A)

    # Run linear solver
    sys_amps, info = scipy.sparse.linalg.cgs(A, b, M=Ainv_estimate, tol=solver_tol)

    # Check solution
    if info > 0:
        # Try again with different solver
        sys_amps, info = scipy.sparse.linalg.bicgstab(A, b, M=Ainv_estimate, tol=solver_tol)
        if info != 0:
            raise ValueError("GCR solver failed after retry; info %d" \
                             % info)
    if info < 0:
        raise ValueError("GCR solver failed; info %d" \
                         % info)
    
    # Calculate |Ax - b|
    residuals = np.sqrt(np.sum(np.abs(A @ sys_amps - b)**2.))
    # Re-pack separate real and imaginary parts into complex vector
    sys_amps = 1.0 * sys_amps[:Nsys_modes].real \
             + 1.j * sys_amps[Nsys_modes:].real
    if verbose:
        print(f"{time.time() - t_start:<12.1f}", end="\t")
        print(f"{info:<8.1f}", end=" ")
        print(f"{residuals:<12.2e}", end="")
    
    return sys_amps