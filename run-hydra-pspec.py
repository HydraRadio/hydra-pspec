
import numpy as np
import scipy
import scipy.special
from pathlib import Path
from pprint import pprint
import os
import time
import sys
import json

from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.typing import Path_fr, Path_dw
from pyuvdata import UVData
from astropy import units
from astropy.units import Quantity

import hydra_pspec as hp
from hydra_pspec.utils import (
    form_pseudo_stokes_vis,
    filter_freqs,
    get_git_version_info,
    add_mtime_to_filepath
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


parser = ArgumentParser()
parser.add_argument(
    "--ant_str",
    type=str,
    default="cross",
    help="Comma delimited list of antenna pairs joined by underscores, e.g. "
         "'1_11,12_14'.  Used via the `ant_str` kwarg in "
         "`pyuvdata.UVData.select`."
)
parser.add_argument(
    "--sigcov0",
    type=str,
    help="Path to a single file or a directory containing a per-baseline "
         "initial guess for the EoR signal covariance.  Files must be readable "
         "by `np.load`."
)
parser.add_argument(
    "--sigcov0_file",
    type=str,
    help="If passing a directory containing per-baseline initial guesses for "
         "the EoR signal covariance to --sigcov0, --sigcov0_file specifies the"
         " name of the file to load in each baseline's subdirectory."
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
    "--fg_eig_file",
    type=str,
    help="If passing a directory containing per-baseline FG eigenvectors to "
         "--fg_eig_dir, --fg_eig_file specifies the name of the file to load "
         " in each baseline's subdirectory."
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
    help="If passing a directory containing per-baseline noise to --noise, "
         "--noise_file specifies the name of the file to load in each "
         "baseline's subdirectory."
)
parser.add_argument(
    "--noise_cov",
    type=str,
    help="Path to a single file or a directory containing per-baseline noise "
         "covariance matrices.  Files must be readable by `np.load`."
)
parser.add_argument(
    "--noise_cov_file",
    type=str,
    help="If passing a directory containint per-baseline noise covariance "
         "matrices to --noise_cov, --noise_cov_file specifies the name of the "
         "file to load in each baseline's subdirectory."
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
    help="If passing a directory containing per-baseline nsamples to "
         "--nsamples, --nsamples_file specifies the name of the file to load "
         "in each baseline's subdirectory."
)
parser.add_argument(
    "--n_ps_prior_bins",
    type=int,
    default=3,
    help="Sets the number of bins to the left and right of the delay=0 mode "
         "bound by the prior (set via --ps_prior_lo and --ps_prior_hi).  If "
         "n_ps_prior_bins is set to 3 (default), then 3 bins to the left and"
         " 3 bins to the right of the delay=0 bin will be affected by the "
         "prior."
)
parser.add_argument(
    "--ps_prior_lo",
    type=float,
    default=0.0,
    help="Sets the lower bound of the prior on the delay power spectrum. "
         "Defaults to 0 which corresponds to no lower bound."
)
parser.add_argument(
    "--ps_prior_hi",
    type=float,
    default=0.0,
    help="Sets the upper bound of the prior on the delay power spectrum. "
         "Defaults to 0 which corresponds to no upper bound."
)
parser.add_argument(
    "--map_estimate",
    action="store_true",
    help="Calculate the maximum a posteriori estimate only (1 iteration)."
)
parser.add_argument(
    "--Niter",
    type=int,
    default=100,
    help="Number of iterations."
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for `numpy.random.seed`."
)
parser.add_argument(
    "-v", "--verbose",
    dest="verbose",
    action="store_true",
    default=False,
    help="Display debug/timing statements."
)
parser.add_argument(
    "--Nproc",
    type=int,
    default=0,
    help="Number of threads per MPI process.  Defaults to the maximum number "
         "of cores available divided by the number of baselines."
)
parser.add_argument(
    "--out_dir",
    type=Path_dw,
    default="./",
    help="Path to directory for writing output(s).  Defaults to './'."
)
parser.add_argument(
    "--dirname",
    type=str,
    help="Name of subdirectory for output(s).  Files will be written to "
         "`Path(args.out_dir) / args.dirname`.  Defaults to "
         "f'results-{freqs.min()}-{freqs.max()}MHz-Niter-{Niter}'."
)
parser.add_argument(
    "--clobber",
    action="store_true",
    default=False,
    help="Clobber existing files."
)
parser.add_argument(
    "--write_Niter",
    type=int,
    default=100,
    help="Number of iterations between output file writing.  Smaller numbers "
         "yield more overhead for I/O.  Larger numbers risk losing more "
         "samples if a job fails or times out."
)
parser.add_argument(
    "file_paths",
    type=Path_fr,
    nargs="+",
    help="Path(s) to UVData compatible visibility file."
)
parser.add_argument(
    '--config',
    action=ActionConfigFile
)
args = parser.parse_args()


def check_shape(shape, d_shape, desc=""):
        assert shape == d_shape, (
            f"The {desc} array has shape {shape} which does not match the "
            f"shape of the per-baseline data, {d_shape}."
        )

def check_load_path(fp):
        """
        Check if file path `fp` points to a file (load) or directory (pass).

        Parameters
        ----------
        fp : str or Path
            Path to file or directory.

        """
        if not isinstance(fp, Path):
            fp = Path(fp)
        
        fp_is_dir = fp.is_dir()
        if fp_is_dir:
            return fp_is_dir, None
        else:
            data = np.load(fp)
            return fp_is_dir, data


if rank == 0:
    if "config" in args.__dict__:
        print(f"Loading config file {str(args.config[0])}", end="\n\n")
    pprint(args.__dict__)

    if not args.file_paths:
        # Load example data from HERA PSpec
        from hera_pspec.data import DATA_PATH
        dfiles = ["zen.2458042.12552.xx.HH.uvXAA"]
        file_paths = [os.path.join(DATA_PATH, df) for df in dfiles[:1]]
    else:
        file_paths = sorted([Path(fp) for fp in args.file_paths])
    nfiles = len(file_paths)
    print(f"\nReading {nfiles} file(s)")

    uvd = UVData()
    if args.freq_range:
        uvd.read(file_paths[0], read_data=False)
        # uvd.freq_array[0] might not work with future versions of pyuvdata
        freqs_in = Quantity(uvd.freq_array[0], unit="Hz")
        freqs_to_keep = filter_freqs(args.freq_range, freqs_in)
        freqs_to_keep = freqs_to_keep.to("Hz").value
    else:
        freqs_to_keep = None
    uvd.read(file_paths, ant_str=args.ant_str, frequencies=freqs_to_keep)
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

    if args.Nproc == 0:
        ncores = len(os.sched_getaffinity(0))  # number of cores available
        nproc = ncores // Nbls  # number of cores per MPI task
        if nproc == 0:
            nproc = 1
    else:
        nproc = args.Nproc
    print(f"\nnproc = {nproc}")

    bl_data_shape = (uvd.Ntimes, uvd.Nfreqs)
    cov_ff_shape = (uvd.Nfreqs, uvd.Nfreqs)
    if args.flags:
        flags_path = Path(args.flags)
        flags_path_is_dir, flags = check_load_path(flags_path)
        if not flags_path_is_dir:
            check_shape(flags.shape, bl_data_shape, desc="flags")
    if args.noise:
        noise_path = Path(args.noise)
        noise_path_is_dir, noise = check_load_path(noise_path)
        if not noise_path_is_dir:
            check_shape(noise.shape, bl_data_shape, desc="noise")
    if args.nsamples:
        nsamples_path = Path(args.nsamples)
        nsamples_path_is_dir, nsamples = check_load_path(nsamples_path)
        if not nsamples_path_is_dir:
            check_shape(nsamples.shape, bl_data_shape, desc="nsamples")
    if args.sigcov0:
        sigcov0_path = Path(args.sigcov0)
        sigcov0_path_is_dir, sigcov0 = check_load_path(sigcov0_path)
        if not sigcov0_path_is_dir:
            check_shape(sigcov0.shape, cov_ff_shape, desc="signal covariance")
    if args.noise_cov:
        noise_cov_path = Path(args.noise_cov)
        noise_cov_path_is_dir, noise_cov = check_load_path(noise_cov_path)
        if not noise_cov_path_is_dir:
            check_shape(noise_cov.shape, cov_ff_shape, desc="noise covariance")

    if args.fg_eig_dir:
        fg_eig_dir = Path(args.fg_eig_dir)
    all_data_weights = []
    for i_bl, antpair in enumerate(antpairs):
        bl_str = f"{antpair[0]}-{antpair[1]}"

        if args.fg_eig_dir:
            # fgmodes has shape (Nfreqs, Nfgmodes)
            if not args.fg_eig_file:
                fgmodes_path = fg_eig_dir / bl_str / f"evecs-{freq_str}.npy"
            else:
                fgmodes_path = fg_eig_dir / bl_str / args.fg_eig_file
            fgmodes = np.load(fgmodes_path)
            fgmodes = fgmodes[:, :args.Nfgmodes]
        else:
            # Generate approximate set of FG modes from Legendre polynomials
            fgmodes = np.array([
                scipy.special.legendre(i)(np.linspace(-1., 1., freqs.size))
                for i in range(args.Nfgmodes)
            ]).T
        
        if args.sigcov0 and sigcov0_path_is_dir:
            bl_sigcov0_path = sigcov0_path / bl_str / args.sigcov0_file
            sigcov0 = np.load(bl_sigcov0_path)
            check_shape(
                sigcov0.shape, cov_ff_shape,
                desc=f"signal covariance ({bl_str})"
            )
        if args.noise_cov and noise_cov_path_is_dir:
            bl_noise_cov_path = noise_cov_path / bl_str / args.noise_cov_file
            noise_cov = np.load(bl_noise_cov_path)
            check_shape(
                noise_cov.shape, cov_ff_shape,
                desc=f"noise covariance ({bl_str})"
            )
        
        d = uvd.get_data(antpair + ("xx",), force_copy=True)
        if args.flags and flags_path_is_dir:
            bl_flags_path = flags_path / bl_str / args.flags_file
            flags = np.load(bl_flags_path)
            check_shape(flags.shape, d.shape, desc=f"flags ({bl_str})")
        elif not args.flags:
            flags = uvd.get_flags(antpair + ("xx",))

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
            "freq_str": freq_str,
            "nproc": nproc
        }
        if args.sigcov0:
            bl_data_weights["S_initial"] = sigcov0
        if args.noise_cov:
            bl_data_weights["N"] = noise_cov
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
nproc = data["nproc"]

Ntimes, Nfreqs = d.shape

# Initial guess at EoR power spectrum
if "S_initial" in data:
    S_initial = data["S_initial"]
else:
    S_initial = np.eye(Nfreqs)

if "N" in data:
    Ninv = np.linalg.inv(data["N"])
else:
    # Simple guess for noise variance
    Ninv = np.eye(Nfreqs)

# Power spectrum prior
# This has shape (2, Ndelays). The first dimension is for the upper and
# lower prior bounds respectively. If the prior for a given delay is
# set to zero, no prior is applied. Otherwise, the solution is restricted
# to be within the range ps_prior[1] < soln < ps_prior[0].
ps_prior = np.zeros((2, Nfreqs))
if args.ps_prior_lo != 0 or args.ps_prior_hi != 0:
    ps_prior_inds = slice(
        Nfreqs//2 - args.n_ps_prior_bins,
        Nfreqs//2 + args.n_ps_prior_bins + 1
    )
    ps_prior[0, ps_prior_inds] = args.ps_prior_hi
    ps_prior[1, ps_prior_inds] = args.ps_prior_lo

# Output file set up
out_dir = Path(args.out_dir)
if not args.dirname:
    out_dir /= f"results-{freq_str}-Niter-{args.Niter}"
else:
    if args.map_estimate:
        out_dir /= args.dirname + "-map-estimate"
    else:
        out_dir /= args.dirname
if out_dir.exists() and not args.clobber:
    # Check for existing output files to avoid overwriting if clobber=False
    add_mtime_to_filepath(out_dir)
out_dir /= f"{bl[0]}-{bl[1]}"
out_dir.mkdir(exist_ok=True, parents=True)
# Catalog git version
try:
    git_info = get_git_version_info()
except:
    git_info = None
with open(out_dir / "git.json", "w") as f:
    # Catalog git version info
    json.dump(git_info, f)
# Catalog command line arguments
parser.save(args, out_dir / "args.json", format="json", skip_none=False)
if rank == 0:
    print(f"\nWriting output(s) to {out_dir.absolute()}", end="\n\n")

# Run Gibbs sampler
# signal_cr = (Niter, Ntimes, Nfreqs) [complex]
# signal_S = (Nfreqs, Nfreqs) [complex]
# signal_ps = (Niter, Nfreqs) [float]
# fg_amps = (Niter, Ntimes, Nfgmodes) [complex]
start = time.time()
signal_cr, signal_S, signal_ps, fg_amps, chisq, ln_post = \
    hp.pspec.gibbs_sample_with_fg(
        d,
        w[0],  # FIXME
        S_initial,
        fgmodes,
        Ninv,
        ps_prior,
        Niter=args.Niter,
        seed=args.seed,
        map_estimate=args.map_estimate,
        verbose=args.verbose,
        nproc=nproc,
        write_Niter=args.write_Niter,
        out_dir=out_dir
    )
elapsed = time.time() - start

# samples = {
#     "signal_cr": signal_cr,
#     "signal_S": signal_S,
#     "signal_ps": signal_ps,
#     "fg_amps": fg_amps,
#     "chisq": chisq,
#     "ln_post": ln_post,
#     "elapsed": elapsed
# }
# data = (bl, samples)

# # Gather results from all baselines
# data = comm.gather(data, root=0)
# if rank == 0:
#     data = dict(data)
#     times = [data[bl_key]["elapsed"] for bl_key in data]
#     times_avg = np.mean(times) * units.s
#     if times_avg.value > 3600:
#         times_avg = times_avg.to("h")
#     elif times_avg.value > 60:
#         times_avg = times_avg.to("min")
#     print(f"Average evaluation time for {args.Niter} iterations: {times_avg}")

