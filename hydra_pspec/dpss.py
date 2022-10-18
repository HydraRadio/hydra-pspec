
import numpy as np
from scipy.optimize import minimize
from scipy.signal.windows import dpss, kaiser


def dpss_fit_modes(d, w, freqs, cov, nmodes=10, alpha=1.,
                   minimize_method='L-BFGS-B', taper=None):
    r"""
    Perform a weighted DPSS fit to masked complex 1D data.
    
    The log-likelihood for each DPSS mode takes the assumed form:
    
    $\log L_n = \tilde{x}^\dagger \tilde{C}^{-1} \tilde{x}$
    
    where $\tau_n = n / \Delta \nu$, $\Delta \nu$ is the bandwidth, and 
    
    $x = [d - A f_dpss(n, \nu))]$.

    The tilde denotes vectors/matrices from which the masked channels 
    (rows/columns) have been removed entirely.
    
    Parameters:
        d (array_like):
            Complex data array that has already had flagged channels removed.
        
        w (array_like):
            Flag array, where 
        
        freqs (array_like):
            Array of frequency values, in MHz.
        
        cov (array_like):
            Covariance matrix model.
        
        nmodes (int, optional):
            Number of DPSS modes to fit.
        
        alpha (float, optional):
            Bandwidth factor used in the DPSS functions. Higher values are more 
            concentrated towards the centre of the band.
        
        taper (array_like, optional):
            If specified, multiplies the data and sinusoid model by a taper 
            function to enforce periodicity. The taper should be evaluated 
            at the locations specified in `freqs`.
        
        minimize_method (str, optional):
            Which SciPy minimisation method to use. Default: `'L-BFGS-B'`.
        
    Returns:
        param1, param2 (array_like):
            If `fit_amp_phase` is True, these are the best-fit amplitude and 
            phase of the sinusoids. Otherwise, they are the real and imaginary 
            amplitudes of the sinusoids.
    """
    # Get shape of data etc.
    assert d.size == cov.shape[0] == cov.shape[1] == freqs.size == w.size, \
             "Data, flags, covariance, and freqs arrays must have same number of channels"
    
    # Taper
    if taper is None:
        taper = 1.
    else:
        assert taper.size == freqs.size, \
                     "'taper' must be evaluated at locations given in 'freqs'"
    
    # Precompute DPSS basis functions, shape: (nmodes, nfreqs)
    dpss_modes = dpss(freqs.size, 
                      NW=alpha, 
                      Kmax=nmodes, 
                      sym=False)
    
    # Invert covariance matrix
    invcov = np.linalg.inv(cov)
    
    # Log-likelihood (or log-posterior) function
    def loglike(p):
        # Real and imaginary coeffs are interleaved
        m = p[0::2,np.newaxis]*dpss_modes[:,:] + 1.j*p[1::2,np.newaxis]*dpss_modes[:,:]
        m = np.sum(m, axis=0)
        
        # Calculate residual and log-likelihood
        x = taper * w * (d - m)
        logl = 0.5 * np.dot(x.conj(), np.dot(invcov, x))
        return logl.real # Result should be real
            
    # Least-squares fit for all modes
    p0 = np.zeros(2*nmodes)
    result = minimize(loglike, p0, 
                      method=minimize_method, 
                      bounds=None)
    amps = result.x
    return dpss_modes, amps

