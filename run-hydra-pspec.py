
import numpy as np
import hydra_pspec as hp
import pylab as plt
import scipy.special
import argparse
from pathlib import Path
import os
from datetime import datetime
import time
import sys

from pyuvdata import UVData
from astropy import units
from astropy.units import Quantity

from hydra_pspec.utils import (
    form_pseudo_stokes_vis, filter_freqs, get_git_version_info
)

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
    "--flags",
    type=str,
    help="Path to a single file or a directory containing per-baseline flags. "
         "Files must be readable by `np.load` and must have the same shape as "
         "the visibilities for a single baseline being analyzed."
)
parser.add_argument(
    "--flags_file",
    type=str,
    help="If passing a directory containing per-baseline flags to --flags, "
         "--flags_file specifies the name of the file to load in each "
         "baseline's subdirectory."
)
parser.add_argument(
    "--noise",
    type=str,
    help="Path to a single file or a directory containing per-baseline noise. "
         "Files must be readable by `np.load` and must have the same shape as "
         "the visibilities for a single baseline being analyzed."
)
parser.add_argument(
    "--noise_file",
    type=str,
    help="If passing a directory containing per-baseline flags to --noise, "
         "--noise_file specifies the name of the file to load in each "
         "baseline's subdirectory."
)
parser.add_argument(
    "--nsamples",
    type=str,
    help="Path to a single file or a directory containing per-baseline "
         "Nsamples arrays. Files must be readable by `np.load` and must have "
         "the same shape as the visibilities for a single baseline being "
         "analyzed."
)
parser.add_argument(
    "--nsamples_file",
    type=str,
    help="If passing a directory containing per-baseline flags to --nsamples, "
         "--nsamples_file specifies the name of the file to load in each "
         "baseline's subdirectory."
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


def check_shape(shape, d_shape, desc=""):
        assert shape == d_shape, (
            f"The {desc} array has shape {shape} which does not match the "
            f"shape of the per-baseline data, {d_shape}."
        )

def add_mtime_to_filename(fp, join_char="-"):
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
        from hera_pspec.data import DATA_PATH
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

    bl_data_shape = (uvd.Ntimes, uvd.Nfreqs)
    if args.flags:
        flags_path = Path(args.flags)
        flags_path_is_dir = flags_path.is_dir()
        if not flags_path_is_dir:
            flags = np.load(flags_path)
            check_shape(flags.shape, bl_data_shape, desc="flags")
    if args.noise:
        noise_path = Path(args.noise)
        noise_path_is_dir = noise_path.is_dir()
        if not noise_path_is_dir:
            noise = np.load(noise_path)
            check_shape(noise.shape, bl_data_shape, desc="noise")
    if args.nsamples:
        nsamples_path = Path(args.nsamples)
        nsamples_path_is_dir = nsamples_path.is_dir()
        if not nsamples_path_is_dir:
            nsamples = np.load(nsamples_path)
            check_shape(nsamples.shape, bl_data_shape, desc="nsamples")

    if args.fg_eig_dir:
        fg_eig_dir = Path(args.fg_eig_dir)
    all_data_weights = []
    for i_bl, antpair in enumerate(antpairs):
        bl_str = f"{antpair[0]}-{antpair[1]}"

        if args.fg_eig_dir:
            # fgmodes has shape (Nfreqs, Nfgmodes)
            fgmodes = np.load(fg_eig_dir / bl_str / f"evecs-{freq_str}.npy")
            fgmodes = fgmodes[:, :args.Nfgmodes]
        else:
            # Generate approximate set of FG modes from Legendre polynomials
            fgmodes = np.array([
                scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
                for i in range(args.Nfgmodes)
            ]).T
        
        d = uvd.get_data(antpair + ("xx",))
        if args.flags and flags_path_is_dir:
            bl_flags_path = flags_path / bl_str / args.flags_file
            flags = np.load(bl_flags_path)
            check_shape(flags.shape, d.shape, desc=f"flags ({bl_str})")

        if args.nsamples and nsamples_path_is_dir:
            bl_nsamples_path = nsamples_path / bl_str / args.nsamples_file
            nsamples = np.load(bl_nsamples_path)
            check_shape(nsamples.shape, d.shape, desc=f"nsamples ({bl_str})")
        else:
            nsamples = None

        if args.noise and noise_path_is_dir:
            bl_noise_path = noise_path / bl_str / args.noise_file
            noise = np.load(bl_noise_path)
            check_shape(noise.shape, d.shape, desc=f"noise ({bl_str})")
            if nsamples is not None:
                noise /= np.sqrt(nsamples)
            d += noise

        bl_data_weights = {
            "bl": antpair,
            "d": d,
            "w": flags,
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
    fgmodes,
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
