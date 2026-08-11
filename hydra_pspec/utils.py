
import numpy as np
from pyuvdata import UVData
import pyuvdata.utils as uvutils
from scipy.signal.windows import blackmanharris
from scipy.integrate import quad
from astropy import units
from astropy.units import Quantity
import ast
import subprocess
import os
import shutil
from pathlib import Path
from datetime import datetime


def _ps_mode_map(Nfreqs, modes_per_bin):
    """
    Mapping between Fourier (delay) modes and power spectrum bins.
    
    Parameters:
        Nfreqs (int):
            Number of frequency channels.
        modes_per_bin (int):
            Number of delay modes in each delay power spectrum bin.
    
    Returns:
        mapping (list of array_like):
            List of arrays, one array for each delay bin. Each array contains 
            the Fourier mode indices for that bin.
    """
    return np.array_split(np.arange(Nfreqs), Nfreqs // modes_per_bin)


def fourier_operator(n):
    """
    Fourier operator for matrix side length n.

    Multiplying a data vector by this matrix operator is equivalent to running
    the following code:
    ```
    data = ...
    # ifftshift and fftshift are interchangeable
    data_fft = numpy.fft.ifftshift(data)
    data_fft = numpy.fft.fft(data_fft)
    data_fft = numpy.fft.fftshift(data_fft)
    ```

    Parameters:
    	n (int):
    		Length of the data that the operator will be applied to. This 
    		should be a multiple of 2 (but doesn't have to be a power of two). 

    Returns:
    	fourier_op (array_like):
    		Complex Fourier operator matrix of shape `(n, n)`.
    """
    if n % 2 == 1:
        raise ValueError("Length must even (divisible by 2)")
    i_x = (np.arange(n) - n//2).reshape(1, -1)
    i_k = (np.arange(n) - n//2).reshape(-1, 1)
    fourier_op = np.exp(-2*np.pi*1j * (i_k * i_x / n))

    return fourier_op


def tau_bins(freqs_MHz, modes_per_bin=1):
    """
    Calculate delay bin centre values for a given frequency band and binning. 
    Includes the correct ordering of the delay bins (matched to the ordering of 
    the `fourier_operator()` function).
    
    Parameters:
        freqs_MHz (array_like):
            Frequency array.
        modes_per_bin (int):
            Number of delay modes in each delay power spectrum bin.
            
    Returns:
        tau (array_like):
            Array of delay values at bin centre, in ns.
    """
    # Get mode map
    mode_idxs = _ps_mode_map(Nfreqs=freqs_MHz.size, modes_per_bin=modes_per_bin)
    
    # Calculate delay value for each DFT mode
    dfreq = (freqs_MHz[1] - freqs_MHz[0]) * 1e6 # MHz -> Hz
    tau = np.fft.fftshift(np.fft.fftfreq(freqs_MHz.size, d=dfreq)) * 1e9 # ns
    
    # Average mode values for the chosen binning
    tau_c = np.zeros(len(mode_idxs))
    for i in range(len(mode_idxs)):
        tau_c[i] = np.mean(tau[mode_idxs[i]])
    return tau_c


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
    if len(data.shape) == 1:
        Nfreqs = data.size
    elif len(data.shape) == 2:
        Nfreqs = data.shape[1]

    if subtract_mean:
        d = data - np.mean(data, axis=1)[:,np.newaxis]
    
    if taper:
        d *= blackmanharris(Nfreqs)
        
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


def pad_input_data(data, flags, inv_var, freqs, Npad=10):
    """
    Introduce a flagged padded area in the frequency dimension to avoid 
    periodicity issues etc.

    Parameters:
        data (array_like):
            Data array, of shape `(Ntimes, Nfreqs)`.
        flags (array_like):
            Flag array, of shape `(Ntimes, Nfreqs)`.
        inv_var (array_like):
            Inverse variance array, of shape `(Ntimes, Nfreqs)`.
        freqs (array_like):
            Frequency array, in MHz, of shape `(Nfreqs,)`.
        Npad (int):
            Number of padding channels to add on either side of 
            the frequency range.

    Returns:
        padded_data (array_like):
            Padded data array, of shape `(Ntimes, Nfreqs + 2*Npad)`.
        padded_flags (array_like):
            Padded flag array, of shape `(Ntimes, Nfreqs + 2*Npad)`.
        padded_inv_var (array_like):
            Padded inverse variance array, of shape `(Ntimes, Nfreqs + 2*Npad)`.
        padded_freqs (array_like):
            Padded frequency array, or shape `(Nfreqs + 2*Npad)`.
    """
    Ntimes, Nfreqs = data.shape
    assert data.shape == flags.shape == inv_var.shape, \
        "data, flags, and inv_var array shapes do not match"
    assert freqs.size == Nfreqs, "freqs array is the wrong size"

    # Create new arrays with padded frequency dimension
    padded_data = np.zeros((Ntimes, Nfreqs+2*Npad), dtype=data.dtype)
    padded_flags = np.zeros((Ntimes, Nfreqs+2*Npad), dtype=flags.dtype)
    padded_inv_var = np.zeros((Ntimes, Nfreqs+2*Npad), dtype=inv_var.dtype)
    padded_freqs = np.zeros((Nfreqs+2*Npad,), dtype=freqs.dtype)
    
    # Insert original data into padded array
    padded_data[:,Npad:-Npad] = data
    padded_flags[:,Npad:-Npad] = flags
    padded_inv_var[:,Npad:-Npad] = inv_var
    
    # Fill-in padded freqs array
    df = freqs[1] - freqs[0]
    padded_freqs[Npad:-Npad] = freqs
    for i in range(Npad + 1):
        padded_freqs[Npad - i] = padded_freqs[Npad] - i*df
        padded_freqs[-Npad + i - 1] \
            = padded_freqs[padded_freqs.size - Npad - 1] + i*df
    
    return padded_data, padded_flags, padded_inv_var, padded_freqs


def form_pseudo_stokes_vis(uvd, convention=1.0):
    """
    Form pseudo-Stokes I visibilities from xx and yy.

    Parameters:
        uvd (pyuvdata.UVData):
            UVData object containing XX and YY polarization visibilities.
        convention (float):
            Factor for getting pI from XX + YY, i.e.
            pI = convention * (XX + YY).  Defaults to 1.0.

    Returns:
        uvd (pyuvdata.UVData):
            UVData object containing pI visibilities.

    """
    assert isinstance(uvd, UVData), "uvd must be a pyuvdata.UVData object."

    if uvutils.polstr2num("pI") not in uvd.polarization_array:
        xx_pol_num = uvutils.polstr2num("xx")
        yy_pol_num = uvutils.polstr2num("yy")
        xpol_ind = np.where(uvd.polarization_array == xx_pol_num)[0]
        ypol_ind = np.where(uvd.polarization_array == yy_pol_num)[0]
        uvd.data_array[..., xpol_ind] += uvd.data_array[..., ypol_ind]
        uvd.data_array *= convention
        uvd.select(polarizations=["xx"])

    return uvd


def filter_freqs(freq_str, freqs_in):
    """
    Returns a subset of `freqs_in` based on the frequency info in `freq_str`.

    Parameters
    ----------
    freq_str : str
        Can be either a single frequency, a comma delimited list of frequencies
        (e.g. '100,110.4,150'), or a minimum and maximum frequency joined by
        '-' (e.g. '100-200.3').  Cannot contain spaces.  Frequencies are
        assumed to be in MHz.  If specifying individual frequencies and the
        specified frequency is not explicitly present in `freqs_in`, the
        closest frequency in `freqs_in` will be kept.
    freqs_in : array-like
        Frequencies in MHz in data to be filtered.

    Returns
    -------
    freqs_out : astropy.units.Quantity
        Masked frequency array containing only the frequencies from `freqs_in`
        that match `freq_str`.

    """
    if not isinstance(freqs_in, Quantity):
        freqs_in = Quantity(freqs_in, unit="MHz")
    else:
        freqs_in = freqs_in.to("MHz")
    freqs_in_range_str = (
        f"{freqs_in.min().value:.2f} - {freqs_in.max().value:.2f} MHz"
    )

    if '-' in freq_str:
        min_freq, max_freq = freq_str.split('-')
        min_freq = ast.literal_eval(min_freq) * units.MHz
        max_freq = ast.literal_eval(max_freq) * units.MHz
        freq_mask = np.logical_and(
            freqs_in >= min_freq, freqs_in <= max_freq
        )
        if np.sum(freq_mask) == 0:
            print(
                f"Frequency range {freq_str} MHz outside of the frequencies in"
                f" `freqs_in`, {freqs_in_range_str}."
            )
    else:
        if ',' in freq_str:
            freqs = [ast.literal_eval(freq) for freq in freq_str.split(',')]
        else:
            freqs = [ast.literal_eval(freq_str)]
        freqs = Quantity(freqs, unit='MHz')
        freqs_in_range = np.array(
            [freqs_in.min() <= freq <= freqs_in.max() for freq in freqs],
            dtype=bool
        )
        if not np.all(freqs_in_range):
            print(
                f"Frequency(ies) {freqs[~freqs_in_range]} are not within the "
                f"range of frequencies in `freqs_in`, {freqs_in_range_str}."
            )
        freqs_inds = [np.argmin(np.abs(freqs_in - freq)) for freq in freqs]
        freq_mask = np.zeros(freqs_in.size, dtype=bool)
        freq_mask[freqs_inds] = True

    freqs_out = freqs_in[freq_mask]

    return freqs_out


def comoving_kperp_kpara(centre_freq_MHz, bl_metres, tau_ns, h=0.67, Omega_m=0.3):
    """
    Calculate the comoving k_perp and k_para values for a given baseline and 
    delay binning scheme at a given central frequency. 
    
    This calculation assumes a flat cosmology with only matter and Lambda, and 
    assumes the delay approximation (i.e. narrow total bandwidths so that 
    cosmological evolution can be neglected).
    
    Parameters:
        centre_freq_MHz (float):
            Centre frequency of the band, in MHz.
        bl_metres (float):
            Baseline length, in metres.
        tau_ns (array_like):
            Delay bin centres, in ns.
        h (float, optional):
            Dimensionless Hubble parameter.
        Omega_m (float, optional):
            Fractional total matter density today.
    
    Returns:
        k_perp (float):
            Perpendicular comoving Fourier mode in h/Mpc.
        k_par (array_like):
            Radial comoving Fourier modes in h/Mpc.
    """
    # Define constants
    C = 299792.458 # km/s
    freq_21_hz = 1.420405751e9 # Hz
    
    # Calculate central redshift
    z_c = freq_21_hz / (centre_freq_MHz * 1e6) - 1.
     
    # Dimensionless expansion rate
    E = lambda z: np.sqrt(Omega_m * (1 + z)**3. + (1. - Omega_m))
    
    # Comoving distance [h^-1 Mpc] and derivatives
    d_comov = quad(lambda z: 1. / E(z), 0., z_c)[0] * (C / 100.)
    dRperp_dtheta = d_comov
    dRpara_df = (1 + z_c)**2. / E(z_c) * (C / 100.) / freq_21_hz
    
    # Conversion factor from delay [ns] to k_parallel mode [h Mpc-1]
    # See hera_pspec.conversions (refs: Parsons 2012, Pober 2014, Kohn 2018)
    tau_to_kpara = 2. * np.pi / dRpara_df / 1e9
    
    # Conversion factor from baseline length [metres] to k_perp mode [h Mpc-1]
    bl_to_kperp = 2. * np.pi * centre_freq_MHz * 1e6 / (C * 1e3 * dRperp_dtheta)
    
    return bl_metres * bl_to_kperp, tau_ns * tau_to_kpara


def map_to_spherical_kbins(k_bin_edges, centre_freq_MHz, bl_metres, tau_ns, 
                           h=0.67, Omega_m=0.3):
    """
    Map baselines and delay modes to 1D (spherical) comoving Fourier bins.
    
    Note that modes that fall outside the binned range will automatically be 
    mapped to the first or last bin, as all modes should be mapped.
    
    Parameters:
        k_bin_edges (array_like):
            Edges of the spherical k bins, in h/Mpc units. The bin edges should 
            be strictly positive and in ascending order.
        centre_freq_MHz (float):
            Centre frequency of the band, in MHz.
        bl_metres (array_like):
            Array of baseline lengths, in metres.
        tau_ns (array_like):
            Delay bin centres, in ns.
        h (float, optional):
            Dimensionless Hubble parameter.
        Omega_m (float, optional):
            Fractional total matter density today.
    
    Returns:
        index_map (list):
            List of per-baseline index arrays mapping delay modes to 1D k 
            modes. For each baseline, there is an array of size `Ndelays`. Each 
            array entry designates the 1D k-bin that that delay maps to.
    """
    bl_metres = np.atleast_1d(bl_metres)
    
    # Check that binning is well-specified
    assert np.all(k_bin_edges >= 0.), \
        "k_bin_edges must have only positive values"
    assert np.all(np.diff(k_bin_edges) > 0.), \
        "k_bin_edges must be in ascending order"
    
    # Loop over baselines
    index_map = []
    for i, bl in enumerate(bl_metres):
        
        # Calculate kperp and kpara for each baseline
        kperp, kpara = comoving_kperp_kpara(
                                        centre_freq_MHz=centre_freq_MHz, 
                                        bl_metres=bl, 
                                        tau_ns=tau_ns, 
                                        h=h, 
                                        Omega_m=Omega_m
                                        )
        k = np.sqrt(kperp**2. + kpara**2.)
        
        # Assign delay modes to bins
        idxs = np.digitize(k, bins=k_bin_edges, right=False)
        index_map.append(idxs)
    
    return index_map
    

def get_git_version_info(directory=None):
    """
    Get git version info from repository in `directory`.

    Parameters
    ----------
    directory : str
        Path to GitHub repository.  If None, uses one directory up from
        __file__.

    Returns
    -------
    version_info : dict
        Dictionary containing git hash information.

    """
    cwd = os.getcwd()
    if directory is None:
        directory = Path(__file__).parent
    os.chdir(directory)

    version_info = {}
    version_info['git_origin'] = subprocess.check_output(
        ['git', 'config', '--get', 'remote.origin.url'],
        stderr=subprocess.STDOUT)
    version_info['git_hash'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        stderr=subprocess.STDOUT)
    version_info['git_description'] = subprocess.check_output(
        ['git', 'describe', '--dirty', '--tag', '--always'])
    version_info['git_branch'] = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        stderr=subprocess.STDOUT)
    for key in version_info.keys():
        version_info[key] = version_info[key].decode('utf8').strip('\n')
    
    os.chdir(cwd)

    return version_info


def add_mtime_to_filepath(fp, join_char="-"):
    """
    Appends the mtime to a filename or directory before the file suffix.

    Modifies the existing file on disk.

    Parameters
    ----------
    fp : str or Path
        Path to file or directory.
    join_char : str
        Character used to append mtime to filename or directory.  
        Defaults to '-'.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    mtime = datetime.fromtimestamp(os.path.getmtime(fp))
    mtime = mtime.isoformat()
    if fp.is_file():
        fp.rename(fp.with_stem(f"{fp.stem}{join_char}{mtime}"))
    elif fp.is_dir():
        shutil.move(
            fp,
            fp.with_name(f"{fp.name}{join_char}{mtime}")
        )


def write_numpy_files(
    fp,
    signal_cr,
    signal_S,
    signal_ps,
    fg_amps,
    chisq,
    ln_post
):
    """
    Write sampling arrays to disk as numpy files.

    Parameters
    ----------
    fp : str or Path
        Output directory for files.
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

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    np.save(fp / f"gcr-eor.npy", signal_cr)
    np.save(fp / f"cov-eor.npy", signal_S)
    np.save(fp / f"dps-eor.npy", signal_ps)
    np.save(fp / f"fg-amps.npy", fg_amps)
    np.save(fp / f"chisq.npy", chisq)
    np.save(fp / f"ln-post.npy", ln_post)
