
import numpy as np


def fourier_operator(n):
    """
    Fourier operator for matrix side length n.

    Parameters:
    	n (int):
    		Length of the data that the operator will be applied to.

    Returns:
    	F (array_like):
    		Complex Fourier operator matrix of shape `(n, n)`.
    """
    F = np.zeros((n,n), dtype=complex)
    for i in range(n):
        y = np.zeros(n)
        y[i] = 1
        F[i] = np.fft.fft(y)
    return F


def naive_pspec(data, subtract_mean=True, taper=True):
    """
	Compute the naive power spectrum of some data, by calculating the 
	product of the FFT'd data with its complex conjugate.

	Parameters:
		data (aray_like):
			Array of complex data to compute the power spectrum of.
		subtract_mean (bool):
			If True, subtract the mean of the data before calculating the 
			power spectrum.
		taper (bool):
			If True, apply a Blackman-Harris taper to the data before 
			computing the power spectrum.

	Returns:
		ps (array_like):
			Complex-valued power spectrum, with fftshift applied.
    """
    if meansub:
        d = data - np.mean(data, axis=1)[:,np.newaxis]
    
    if taper:
        d *= BH(s)
        
    return np.fft.fftshift(abs(np.fft.fft(d))**2)


def trim_flagged_channels(w, x):
    """
    Remove flagged channels from a 1D or 2D (square) array. This is 
    a necessary pre-processing step for LSSA.

    Parameters:
        w (array_like):
            1D array of mask values, where 1 means unmasked and 0 means 
            masked.
        
        x (array_like):
            1D or square 2D array to remove the masked channels from.

    Returns:
        xtilde (array_like):
            Input array with the flagged channels removed.
    """
    # Check inputs
    assert np.shape(x) == (w.size,) or np.shape(x) == (w.size, w.size), \
             "Input array must have shape (w.size) or (w.size, w.size)"

    # 1D case
    if len(x.shape) == 1:
        return x[w == 1.]
    else:
        return x[:,w == 1.][w == 1.,:]
