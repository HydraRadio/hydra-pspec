
import numpy as np
from pyuvdata import UVData
import pyuvdata.utils as uvutils
from scipy.signal.windows import blackmanharris
from astropy import units
from astropy.units import Quantity
import ast
import subprocess
import os
import shutil
from pathlib import Path
import h5py
from datetime import datetime

def fourier_operator(n, unitary=True):
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
    		Length of the data that the operator will be applied to.
        unitary (bool):
            Whether the matrix should be unitary, i.e. F^dagger F = I.

    Returns:
    	fourier_op (array_like):
    		Complex Fourier operator matrix of shape `(n, n)`.
    """
    norm = 1.
    if unitary:
        norm = np.sqrt(n)

    i_x = (np.arange(n) - n//2).reshape(1, -1)
    i_k = (np.arange(n) - n//2).reshape(-1, 1)

    fourier_op = np.exp(-2*np.pi*1j * (i_k * i_x / n)) / norm
    return fourier_op


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

# Make a  power spectrum
def calc_ps(s):
    # NOTE: This uses inverse FFT instead of FFT to get the right normalisation
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    Nobs, Nfreqs = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / Nfreqs # CHECK: This takes an average

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
    signal_amps,
    signal_ps,
    fg_amps,
    sys_amps,
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
    b_sys (array_like):
        Samples of the systematics amplitudes, shape `(Niter,len(nm_list))
    chisq (array_like):
        Chi-squared value per iteration, shape `(Niter, Ntimes, Nfreqs)`.
    ln_post (array_like):
        Natural log of the posterior probability per iteration, shape
        `(Niter,)`.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    np.save(fp / f"gcr-eor.npy", signal_amps)
    np.save(fp / f"dps-eor.npy", signal_ps)
    np.save(fp / f"fg-amps.npy", fg_amps)
    np.save(fp / f"b-sys.npy", sys_amps)    
    np.save(fp / f"chisq.npy", chisq)
    np.save(fp / f"ln-post.npy", ln_post)


# def write_numpy_files_dynamic(
#     fp,
#     signal_amps,
#     signal_ps,
#     fg_amps,
#     sys_amps,
#     chisq,
#     ln_post
# ):
#     """
#     Write sampling arrays to disk as numpy files for every iteration.

#     Parameters
#     ----------
#     fp : str or Path
#         Output directory for files.
#     signal_cr (array_like):
#         Samples of the signal, shape `(Niter, Ntimes, Nfreqs)`.
#     signal_S (array_like):
#         Samples of the signal covariance, shape `(Niter, Nfreqs, Nfreqs)`.
#         These are simply transformations of the power spectrum.
#     signal_ps (array_like):
#         Sample of the signal power spectrum bandpowers, shape
#         `(Niter, Nfreqs)`.
#     fg_amps (array_like):
#         Samples of the foreground amplitudes, shape `(Niter, Nmodes)`.
#     b_sys (array_like):
#         Samples of the systematics amplitudes, shape `(Niter,len(nm_list))
#     chisq (array_like):
#         Chi-squared value per iteration, shape `(Niter, Ntimes, Nfreqs)`.
#     ln_post (array_like):
#         Natural log of the posterior probability per iteration, shape
#         `(Niter,)`.

#     """
#     if not isinstance(fp, Path):
#         fp = Path(fp)
#     with open(fp / f"gcr-eor_dynamic.npy",'a') as file:
#         np.savetxt(file,signal_amps)
#     with open(fp / f"dps-eor_dynamic.npy",'a') as file:
#         np.savetxt(file,signal_ps)
#     with open(fp / f"fg-amps_dynamic.npy",'a') as file:
#         np.savetxt(file,fg_amps)  
#     with open(fp / f"b-sys_dynamic.npy",'a') as file:
#         np.savetxt(file,sys_amps)  
#     with open(fp / f"chisq_dynamic.npy",'a') as file:
#         np.savetxt(file,chisq) 
#     with open(fp / f"ln-post_dynamic.npy",'a') as file:
#         print(np.array([ln_post]))
#         np.savetxt(file,np.array([ln_post])) 

def append_gibbs_sample_h5(fp, overwrite=False, flush=True,batch_axis=None, **arrays):
    """
    Append Gibbs samples to an HDF5 file, creating datasets on first use.

    - If batch_axis is None (default): each **array** must be ONE sample
      (e.g., (n,m)), and this appends 1 row.
    - If batch_axis is an int (e.g., 0): treats that axis as a batch of samples
      (e.g., (B,n,m)) and appends B rows at once.

    Complex dtypes are preserved.

    Parameters
    ----------
    fp : str or Path
        Output directory (creates fp/'gibbs_samples.h5').
    overwrite : bool
        If True and file exists, delete it (use at the start of a run).
    flush : bool
        Flush file to disk after writing.
    batch_axis : None or int
        If int, the axis in each array that indexes multiple samples to append.
    **arrays : name=array_like
        Per-quantity sample(s). Shapes must be consistent across calls.
    """
    fp = Path(fp)
    fp.mkdir(parents=True, exist_ok=True)
    h5_path = fp / "gibbs_samples.h5"

    if overwrite and h5_path.exists():
        os.remove(h5_path)

    with h5py.File(h5_path, "a") as f:
        for name, arr in arrays.items():
            arr = np.asarray(arr)

            # Arrange as (B, ...) where B=number of samples to append this call
            if batch_axis is None:
                batch = arr[np.newaxis, ...]   # (1, ...)
                per_sample_shape = arr.shape
                dtype = arr.dtype
            else:
                batch = np.moveaxis(arr, batch_axis, 0)  # (B, ...)
                per_sample_shape = batch.shape[1:]
                dtype = batch.dtype

            # Create dataset on first sight
            if name not in f:
                f.create_dataset(
                    name,
                    shape=(0,) + per_sample_shape,
                    maxshape=(None,) + per_sample_shape,
                    dtype=dtype,
                    chunks=(max(1, min(32, batch.shape[0])),) + per_sample_shape,
                    compression="gzip"
                )

            dset = f[name]

            # Validate shape consistency
            if dset.shape[1:] != per_sample_shape:
                raise ValueError(
                    f"{name}: incoming per-sample shape {per_sample_shape} "
                    f"does not match existing {dset.shape[1:]}."
                )

            # Append rows
            i0 = dset.shape[0]
            i1 = i0 + batch.shape[0]
            dset.resize(i1, axis=0)
            dset[i0:i1, ...] = batch

        if flush:
            f.flush()