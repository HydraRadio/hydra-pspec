
import numpy as np
import hydra_pspec as hp
import pylab as plt
import scipy.special
import argparse
from pathlib import Path
import ast
import os
import subprocess
from datetime import datetime
import time
import sys

from pyuvdata import UVData
from hera_pspec.data import DATA_PATH
from astropy import units
from astropy.units import Quantity

from hydra_pspec.utils import form_pseudo_stokes_vis

try:
    from mpi4py import MPI

    HAVE_MPI = True

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except ImportError:
    HAVE_MPI = False
    # This can be made more robust
    import sys
    print("This script requires MPI.  Exiting.")
    sys.exit()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ant_str",
    type=str,
    help="Comma delimited list of antenna pairs joined by underscores, e.g. "
         "'1_11,12_14'.  Used via the `ant_str` kwarg in "
         "`pyuvdata.UVData.select`."
)
parser.add_argument(
    "--Nfgmodes",
    type=int,
    default=8,
    help="Number of FG eigenmodes for FG model."
)
parser.add_argument(
    "--fg_eig_dir",
    type=str,
    help="Path to directory containing per-baseline FG eigenvector arrays."
)
parser.add_argument(
    "--freq_range",
    type=str,
    help="Frequencies to use in the data.  Can be either a single frequency, "
         "a comma delimited list of frequencies ('100,110,150'), or a minimum"
         " and maximum frequency joined by '-' ('100-200').  Frequency string"
         " cannot contain spaces."
)
parser.add_argument(
    "--Niter",
    type=int,
    default=100,
    help="Number of iterations."
)
parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    default=False,
    help="Display debug/timing statements."
)
parser.add_argument(
    "--Nproc",
    type=int,
    default=1,
    help="Number of threads per MPI process."
)
parser.add_argument(
    "--out_dir",
    type=str,
    help="Path to directory for writing output(s)."
)
parser.add_argument(
    "--clobber",
    action="store_true",
    default=False,
    help="Clobber existing files."
)
parser.add_argument(
    "file_paths",
    type=str,
    nargs="+",
    help="Path(s) to UVData compatible visibility file."
)
args = parser.parse_args()


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

def add_mtime_to_filename(fp, join_char='-'):
    """
    Appends the mtime to a filename before the file suffix.

    Modifies the existing file on disk.

    Parameters
    ----------
    fp : str or Path
        Path to file.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    mtime = datetime.fromtimestamp(os.path.getmtime(fp))
    mtime = mtime.isoformat()
    fp.rename(fp.with_stem(f"{fp.stem}{join_char}{mtime}"))

def write_numpy_file(fp, arr, clobber=False):
    """
    Write a numpy file to disk with checks for existing files.

    Parameters
    ----------
    fp : str or Path
        Path to file.
    clobber : bool
        If True, overwrite file if it exists.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if fp.exists() and not clobber:
        add_mtime_to_filename(fp)
    np.save(fp, arr)


if rank == 0:
    # Load example data from HERA PSpec
    if not args.file_paths:
        # dfiles = [
        #     "zen.2458042.12552.xx.HH.uvXAA",
        #     "zen.2458042.12552.xx.HH.uvXAA"
        # ]
        dfiles = ["zen.2458042.12552.xx.HH.uvXAA"]
        file_paths = [os.path.join(DATA_PATH, df) for df in dfiles[:1]]
    else:
        file_paths = sorted([Path(fp) for fp in args.file_paths])
    nfiles = len(file_paths)
    print(f"\nReading {nfiles} file(s)", end="\n\n")

    if args.ant_str:
        ant_str = args.ant_str
    else:
        ant_str = ""
    uvd = UVData()
    if args.freq_range:
        uvd.read(file_paths[0], read_data=False)
        # uvd.freq_array[0] might not work with future versions of pyuvdata
        freqs_in = Quantity(uvd.freq_array[0], unit="Hz")
        freqs_to_keep = filter_freqs(args.freq_range, freqs_in)
    else:
        freqs_to_keep = None
    uvd.read(file_paths, ant_str=ant_str, frequencies=freqs_to_keep)
    uvd.conjugate_bls()
    if args.file_paths:
        uvd = form_pseudo_stokes_vis(uvd)
    # Assuming for now that size == Nbls
    antpairs = uvd.get_antpairs()[:size]
    Nbls = len(antpairs)

    freqs = Quantity(uvd.freq_array[0], unit="Hz")
    freq_str = (
        f"{freqs.min().to('MHz').value:.3f}-"
        + f"{freqs.max().to('MHz').value:.3f}MHz"
    )

    if args.fg_eig_dir:
        fg_eig_dir = Path(args.fg_eig_dir)
    all_data_weights = []
    for i_bl, antpair in enumerate(antpairs):
        if args.fg_eig_dir:
            # fgmodes has shape (Nfreqs, Nfgmodes)
            fgmodes = np.load(
                fg_eig_dir
                / f"{antpair[0]}-{antpair[1]}"
                / f"evecs-{freq_str}.npy"
            )
            fgmodes = fgmodes[:, :args.Nfgmodes]
        else:
            # Generate approximate set of FG modes from Legendre polynomials
            fgmodes = np.array([
                scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
                for i in range(args.Nfgmodes)
            ]).T

        bl_data_weights = {
            "bl": antpair,
            "d": uvd.get_data(antpair + ("xx",)),
            "w": uvd.get_flags(antpair + ("xx",)),
            "fgmodes": fgmodes,
            "freq_str": freq_str
        }
        all_data_weights.append(bl_data_weights)
else:
    all_data_weights = None

# Send per-baseline visibilities to each process
data = comm.scatter(all_data_weights)
bl = data["bl"]
d = data["d"]
w = ~data["w"]
fgmodes = data["fgmodes"]
freq_str = data["freq_str"]

Ntimes, Nfreqs = d.shape

# Initial guess at EoR power spectrum
# S_initial = np.eye(Nfreqs)
S_initial = np.dot(d.conj().T, d) / Ntimes

# Simple guess for noise variance
Ninv = np.eye(Nfreqs)

# Power spectrum prior
# This has shape (2, Ndelays). The first dimension is for the upper and
# lower prior bounds respectively. If the prior for a given delay is
# set to zero, no prior is applied. Otherwise, the solution is restricted
# to be within the range ps_prior[1] < soln < ps_prior[0].
ps_prior = np.zeros((2, Nfreqs))
ps_prior[0, Nfreqs//2-3:Nfreqs//2+3] = 0.1
ps_prior[1, Nfreqs//2-3:Nfreqs//2+3] = -0.1

# Run Gibbs sampler
# signal_cr = (Niter, Ntimes, Nfreqs)
# signal_S = (Nfreqs, Nfreqs)
# signal_ps = (Niter, Nfreqs)
# fg_amps = (Niter, Ntimes, Nfgmodes)
start = time.time()
signal_cr, signal_S, signal_ps, fg_amps = hp.pspec.gibbs_sample_with_fg(
    d,
    w[0],  # FIXME
    S_initial,
    fgmodes,  # FIXME
    Ninv,
    ps_prior,
    Niter=args.Niter,
    seed=None,
    verbose=args.verbose,
    nproc=args.Nproc,
)
elapsed = time.time() - start

results_dict = {
    "signal_cr": signal_cr,
    "signal_S": signal_S,
    "signal_ps": signal_ps,
    "fg_amps": fg_amps,
    "elapsed": elapsed
}
data = (bl, results_dict)

if args.out_dir:
    out_dir = args.out_dir
else:
    out_dir = "./"
out_dir = Path(out_dir)
if rank == 0:
    print(f"\nWriting output(s) to {out_dir}", end="\n\n")
out_path = (
    out_dir
    / f"{bl[0]}-{bl[1]}"
    / f"results-{freq_str}.npy"
)
out_path.parent.mkdir(exist_ok=True)
try:
    git_info = get_git_version_info()
except:
    git_info = None
out_dict = {
    "res": results_dict,
    "git": git_info,
    "args": args
}
write_numpy_file(out_path, out_dict, clobber=args.clobber)

# Gather results from all baselines
data = comm.gather(data, root=0)
if rank == 0:
    data = dict(data)
    times = [data[bl_key]["elapsed"] for bl_key in data]
    times_avg = np.mean(times) * units.s
    if times_avg.value > 3600:
        times_avg = times_avg.to("h")
    elif times_avg.value > 60:
        times_avg = times_avg.to("min")
    print(f"Average evaluation time for {args.Niter} iterations: {times_avg}")

# if rank == 0:
#     data = dict(data)

    # if Nbls > 1:
    #     # TODO: How do we combine the PS once we gather them?  Simple average?
    #     bl_avg_ps = np.zeros((Niter, Nfreqs))
    #     for i_bl, antpair in enumerate(antpairs):
    #         bl_avg_ps += data[antpair]["res"]["signal_ps"]
    #     bl_avg_ps /= Nbls

    # df = np.diff(freqs).mean()
    # delays = Quantity(
    #     np.fft.fftshift(np.fft.fftfreq(Nfreqs, d=df.to("Hz").value)),
    #     unit="s"
    # )

    # nrows = Nbls + (Nbls > 1)
    # plot_width = 5.0
    # plot_height = 4.0
    # figsize = (plot_width, plot_height * nrows)
    # fig, axs = plt.subplots(nrows, 1, figsize=figsize)
    # if nrows == 1:
    #     axs = [axs]

    # if Nbls > 1:
    #     ax = axs[0]
    #     ax.plot(
    #         delays.to("ns").value,
    #         np.fft.fftshift(bl_avg_ps, axes=(1,)).T,
    #         color="k",
    #         alpha=0.3
    #     )
    #     ax.set_title("Baseline Average")

    # for i_bl, bl in enumerate(antpairs):
    #     if Nbls > 1:
    #         i_ax = i_bl + 1
    #     else:
    #         i_ax = i_bl
    #     ax = axs[i_ax]
    #     ax.plot(
    #         delays.to("ns").value,
    #         np.fft.fftshift(data[bl]["res"]["signal_ps"], axes=(1,)).T,
    #         color="k",
    #         alpha=0.3
    #     )
    #     ax.set_title(f"({bl[0]}, {bl[1]})")
    
    # for i_ax, ax in enumerate(axs):
    #     ax.set_ylabel(r"Delay Power Spectrum")
    # axs[-1].set_xlabel(r"Delay [ns]")

    # fig.tight_layout()
