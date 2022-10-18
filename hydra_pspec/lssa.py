
import numpy as np
from scipy.optimize import minimize


def model_ap(amp, phase, tau, freqs):
    return amp * np.exp(-2.*np.pi*1.j*tau*freqs + 1.j*phase)


def model_aa(A_re, A_im, tau, freqs):
    return (A_re + 1.j*A_im) * np.exp(-2.*np.pi*1.j*tau*freqs)


def decorr_matrix(w, tau, freqs):
    """
    Calculate rotation matrix from Eq. 8 of Bryna's note, 
    needed to decorrelate the real and imaginary amplitudes of 
    the least squares-fitted cosine/sine modes.
    
    To use this matrix to decorrelate the amplitudes, do:
    `np.dot(rot, [A_real, A_imag])`
    
    Parameters
    ----------
    w : array_like
        Mask vector, 1 for unmasked, 0 for masked.
    
    tau : float
        Delay wavenumber.
    
    freqs : array_like
        Frequency array.
    
    Returns
    -------
    rot : array_like
        Rotation matrix to be applied to the amplitude vector.
        
    eigvals : array_like
        Eigenvalues of mode correlation matrix. Multiply the 
        variance of the mode, sigma^2, with these eigenvalues 
        to get the new variances (sigma1^2, sigma2^2); see 
        Eq. 9 of Bryna's note.
    """
    # Sine and cosine terms with mask
    cos = w*np.cos(2.*np.pi*tau*freqs)
    sin = w*np.sin(2.*np.pi*tau*freqs)
    
    # Covariance (overlap) matrix
    cov = np.zeros((2, 2))
    cov[0,0] = np.sum(cos*cos)
    cov[0,1] = cov[1,0] = np.sum(cos*sin)
    cov[1,1] = np.sum(sin*sin)
    
    # Calculate rotation angle directly
    theta = 0.5 * np.arctan2(2.*np.sum(cos*sin), 
                             np.sum(cos*cos) - np.sum(sin*sin))
    rot = np.array([[np.cos(theta), np.sin(theta)], 
                     [-np.sin(theta), np.cos(theta)]])
    rinv = np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])
    eigvals = np.diag(np.dot(rot, np.dot(cov, rinv)))
    
    # Eigendecomposition
    #eigvals, eigvec = np.linalg.eig(cov)
    
    # Rotation operator is inverse of the eigenvector matrix
    #rot = np.linalg.pinv(eigvec)
    return rot, eigvals



def decorr_pspec(A_re, A_im, w, tau, freqs):
    """
    Calculate the LSSA power spectrum, by using Bryna's decorrelation 
    scheme to re-weight the real and imaginary amplitudes.
    """
    ps = np.zeros(tau.size)
    
    # Loop over tau modes
    for i, t in enumerate(tau):
        # Get decorrelation matrix and eigenvalues
        rot, eigvals = decorr_matrix(w=w, tau=t, freqs=freqs)
        
        # Apply decorrelation rotation
        A1, A2 = np.matmul(rot, np.array([A_re[i], A_im[i]]))
        
        # Construct power spectrum (c.f. Eq. 12 of Bryna's note)
        # Multiplied num. and denom. by each eigval squared to avoid 1/0
        ps[i] = ((A1 * eigvals[1])**2. + (A2 * eigvals[0])**2.) \
                       / (eigvals[0]**2. + eigvals[1]**2.)
    return ps


def lssa_fit_modes(d, freqs, invcov=None, fit_amp_phase=True, tau=None, 
                   minimize_method='L-BFGS-B', taper=None):
    r"""
    Perform a weighted LSSA fit to masked complex 1D data.

    NOTE: The input data/covariance should have already had the flagged 
    channels removed. Use the `trim_flagged_channels()` function to do 
    this.
    
    The log-likelihood for each sinusoid takes the assumed form:
    
    $\log L_n = \tilde{x}^\dagger \tilde{C}^{-1} \tilde{x}$
    
    where $\tau_n = n / \Delta \nu$, $\Delta \nu$ is the bandwidth, and 
    
    $x = [d - A \exp(2 \pi i \nu \tau_n + i\phi)]$.

    The tilde denotes vectors/matrices from which the masked channels 
    (rows/columns) have been removed entirely.
    
    Parameters:
        d (array_like):
            Complex data array that has already had flagged channels removed.
        
        freqs (array_like):
            Array of frequency values, in MHz. Used to get tau values in 
            the right units only. Flagged channels must have already been 
            removed.
        
        invcov (array_like):
            Inverse of the covariance matrix (flagged channels must have been 
            removed before inverting).

        fit_amp_phase (bool, optional):
            If True, fits the (real) amplitude and (real) phase parameters 
            for each sinusoid. If False, fits the real and imaginary amplitudes.
        
        tau (array_like, optional):
            Array of tau modes to fit. If `None`, will use `fftfreq()` to 
            calculate the tau values. Units: nanosec.
        
        taper (array_like, optional):
            If specified, multiplies the data and sinusoid model by a taper 
            function to enforce periodicity. The taper should be evaluated 
            at the locations specified in `freqs`
        
        minimize_method (str, optional):
            Which SciPy minimisation method to use. Default: `'L-BFGS-B'`.
    
    Returns:
        tau (array_like):
            Wavenumbers, calculated as tau_n = n / L, in nanoseconds.
            
        param1, param2 (array_like):
            If `fit_amp_phase` is True, these are the best-fit amplitude and 
            phase of the sinusoids. Otherwise, they are the real and imaginary 
            amplitudes of the sinusoids.
    """
    # Get shape of data etc.
    bandwidth = (freqs[-1] - freqs[0]) / 1e3 # assumed MHz, convert to GHz
    assert d.size == invcov.shape[0] == invcov.shape[1] == freqs.size, \
               "Data, inv. covariance, and freqs array must have same number of channels"
    
    # Calculate tau values
    if tau is None:
        tau = np.fft.fftfreq(n=freqs.size, d=freqs[1]-freqs[0]) * 1e3 # nanosec
    
    # Taper
    if taper is None:
        taper = 1.
    else:
        assert taper.size == freqs.size, \
               "'taper' must be evaluated at locations given in 'freqs'"
    
    # Log-likelihood (or log-posterior) function
    def loglike(p, n):
        if fit_amp_phase:
            m = model_ap(amp=p[0], phase=p[1], tau=tau[n], freqs=freqs)
        else:
            m = model_aa(A_re=p[0], A_im=p[1], tau=tau[n], freqs=freqs)
        
        # Calculate residual and log-likelihood
        x = taper * (d - m)
        logl = 0.5 * np.dot(x.conj(), np.dot(invcov, x))
        return logl.real # Result should be real
    
    # Set appropriate bounds for fits
    max_abs = np.max(np.abs(d))
    if fit_amp_phase:
        bounds = [(-100.*max_abs, 100.*max_abs), (0., 2.*np.pi)]
    else:
        bounds = [(-1000.*max_abs, 1000.*max_abs), (-1000.*max_abs, 1000.*max_abs)]
    
    # Do least-squares fit for each tau
    param1 = np.zeros(tau.size)
    param2 = np.zeros(tau.size)
    
    for n in range(tau.size):
        p0 = np.zeros(2)

        # Rough initial guess
        if fit_amp_phase:
            p0[0] = 0.2 * np.max(np.abs(d))
            p0[1] = 0.5 * np.pi
        else:
            p0[0] = 0.2 * np.max(d.real) # rough guess at amplitude
            p0[1] = 0.2 * np.max(d.imag)
        
        # Least-squares fit for mode n
        result = minimize(loglike, p0, args=(n,), 
                          method=minimize_method, 
                          bounds=bounds)
        param1[n], param2[n] = result.x
    
    return tau, param1, param2