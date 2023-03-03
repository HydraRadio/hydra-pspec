"""
This script calculates time-averaged, frequency-frequency covariance matrices
per baseline from pyuvdata compatible visibility files.  The eigenvectors of
each covariance matrix can also be computed via the `--eig` command line
argument.

The `--out-dir` command line argument specifies where the output data will be
saved to disk.  A dictionary object is saved in this output directory
containing useful metadata for posterity.  A subdirectory is created for each
baseline analyzed in which is saved the corresponding covariance matrix and
eigendecomposition as binary numpy files.
"""

import os
import ast
from datetime import datetime
import numpy as np
from pathlib import Path
from pyuvdata import UVData
import pyuvdata.utils as uvutils
from argparse import ArgumentParser
from tqdm import tqdm
from astropy import units
from astropy.units import Quantity

from hydra_pspec.utils import get_git_version_info


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


parser = ArgumentParser(
    description=(
        "Calculate time-averaged covariance matrices from visibilities."
    )
)
parser.add_argument(
    "file_paths",
    nargs="+",
    type=str,
    help="Path to pyuvdata.UVData compatible file."
)
parser.add_argument(
    "--out-dir",
    dest="out_dir",
    type=str,
    default="",
    help="Path to directory for output data.  Defaults to './'"
)
parser.add_argument(
    "--ant-str",
    dest="ant_str",
    type=str,
    help="String for pyuvdata.UVData.select's ant_str argument."
)
parser.add_argument(
    "--freq-range",
    dest="freq_range",
    type=str,
    help="Frequencies to use in the data.  Can be either a single frequency, "
         "a comma delimited list of frequencies, '100.1,110,150', or a "
         "minimum and maximum frequency joined by '-', '100-200'.  Cannot "
         "contain spaces."
)
parser.add_argument(
    "--eig",
    action="store_true",
    help="Perform an eigendecomposition of the covariance matrix of each "
         "baseline and store the eigenvectors and values in the output data "
         "dictionary."
)
parser.add_argument(
    "--clobber",
    action="store_true",
    default=False,
    help="Overwrite files if they exist."
)
args = parser.parse_args()

if args.ant_str:
    ant_str = args.ant_str
else:
    ant_str = "cross"

file_paths = sorted([Path(fp) for fp in args.file_paths])
nfiles = len(file_paths)
print(f"\nReading {nfiles} file(s)", end="\n\n")
uvd = UVData()
if args.freq_range:
    # Is it a bad idea to only read the metadata from file_paths[0]?
    uvd.read(file_paths[0], read_data=False)
    # uvd.freq_array[0] might not work with future versions of pyuvdata
    freqs_in = Quantity(uvd.freq_array[0], unit="Hz")
    freqs_to_keep = filter_freqs(args.freq_range, freqs_in)
else:
    freqs_to_keep = None
uvd.read(file_paths, ant_str=ant_str, frequencies=freqs_to_keep)
uvd.conjugate_bls("ant1<ant2")

# Useful metadata
freqs = Quantity(uvd.freq_array[0], unit="Hz")
freq_str = (
    f"{freqs.min().to('MHz').value:.3f}-{freqs.max().to('MHz').value:.3f}MHz"
)
lsts = Quantity(np.unique(uvd.lst_array) * 12.0 / np.pi, unit="h")
lst_str = f"lst-{lsts[0].to('h').value:.2f}-{lsts[-1].to('h').value:.2f}"
bls = uvd.get_antpairs()
uvws = np.zeros((len(bls), 3))
for i_bl, bl in enumerate(bls):
    uvws[i_bl] = uvd.uvw_array[uvd.antpair2ind(bl)[0]]

print(f"\nForming pI visibilities", end="\n\n")
if uvutils.polstr2num("pI") not in uvd.polarization_array:
    # Make pI visibilities from 0.5 * (XX + YY)
    xpol_ind = np.where(uvd.polarization_array == uvutils.polstr2num("xx"))[0]
    ypol_ind = np.where(uvd.polarization_array == uvutils.polstr2num("yy"))[0]
    uvd.data_array[..., xpol_ind] += uvd.data_array[..., ypol_ind]
    uvd.data_array *= 0.5
    uvd.select(polarizations=["xx"])

# Set up directory structure
if args.out_dir == "":
    base_dir = Path("./")
else:
    base_dir = Path(args.out_dir)
base_dir /= f"{lst_str}"
base_dir.mkdir(exist_ok=True)
git_info = get_git_version_info()
metadata_dict = {
    "git": git_info, "args": args, "freqs": freqs.to("Hz").value,
    "lsts": lsts.to("h").value, "uvws": uvws, "bls": bls
}
write_numpy_file(
    base_dir / "metadata-dict.npy", metadata_dict, clobber=args.clobber
)

# Get per-baseline time-averaged freq-freq covariance matrices
print("Calculating covariance matrices" + args.eig*" and Eigen vecs/vals")
for i_bl, bl in enumerate(tqdm(bls, desc="Baselines")):
    bl_dir = base_dir / f"{bl[0]}-{bl[1]}"
    bl_dir.mkdir(exist_ok=True)
    bl_data = uvd.get_data(bl)  # shape (Ntimes, Nfreqs)
    cov_mat = np.cov(bl_data.T)
    cov_file = f"cov-{freq_str}.npy"
    write_numpy_file(bl_dir / cov_file, cov_mat, clobber=args.clobber)
    if args.eig:
        evals, evecs = np.linalg.eig(cov_mat)
        evec_file = f"evecs-{freq_str}.npy"
        write_numpy_file(bl_dir / evec_file, evecs, clobber=args.clobber)
        eval_file = f"evals-{freq_str}.npy"
        write_numpy_file(bl_dir / eval_file, evals, clobber=args.clobber)
        del evals, evecs
    del cov_mat
