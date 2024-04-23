import numpy as np
import matplotlib.pyplot as plt
from pyuvdata import UVData
from astropy import units
from jsonargparse import ArgumentParser
from pathlib import Path

from hydra_pspec.utils import form_pseudo_stokes_vis


parser = ArgumentParser()
parser.add_argument(
    "--vis-eor",
    type=str,
    default="./vis-eor.uvh5",
    help="Path to UVData compatible visibility file containing EoR "
         "visibilities.  Defaults to './vis-eor.uvh5'."
)
parser.add_argument(
    "--res-dir",
    type=str,
    default="./results-seed-7123689-Niter-1000/0-1/",
    help="Path to a directory containing outputs from hydra-pspec.  Defaults "
         "to './results-seed-7123689-Niter-1000/0-1/'."
)
parser.add_argument(
    "--conf-interval",
    type=int,
    default=95,
    help="Confidence interval for delay power spectrum posteriors."
)
parser.add_argument(
    "--Nburn",
    type=int,
    default=0,
    help="Number of samples to skip due to burn in.  Defaults to 0."
)
args = parser.parse_args()


# Load in EoR visibilities
uvd = UVData()
uvd.read(args.vis_eor)
uvd.conjugate_bls()
uvd = form_pseudo_stokes_vis(uvd)
# The test data only contains a single baseline (0, 1) and the pseudo-Stokes I
# visibilities after `form_pseudo_stokes_vis` are stored in the XX polarization
vis_eor = uvd.get_data((0, 1, "xx"))  # shape (Ntimes, Nfreqs)

# Get freuqency metadata
freqs = uvd.freq_array * units.Hz
if uvd.use_future_array_shapes:
    freqs = freqs[0]
df = freqs[1] - freqs[0]
Nfreqs = freqs.size

# Compute the delay power spectrum of the input EoR signal
axes = (1,)
ds_eor_true = np.fft.ifftshift(vis_eor, axes=axes)
ds_eor_true = np.fft.fftn(ds_eor_true, axes=axes)
ds_eor_true = np.fft.fftshift(ds_eor_true, axes=axes)
dps_eor_true = (np.abs(ds_eor_true)**2).mean(axis=0)
delays = np.fft.fftshift(np.fft.fftfreq(Nfreqs, d=df.to("1/ns")))

# Load in results from hydra_pspec
dps_eor_hp = np.load(Path(args.res_dir) / "dps-eor.npy")
ln_post = np.load(Path(args.res_dir) / "ln-post.npy")
if args.Nburn > 0:
    dps_eor_hp = dps_eor_hp[args.Nburn:]
    ln_post = ln_post[args.Nburn:]
# Posterior-weighted mean delay power spectrum
dps_eor_hp_pwm = np.average(dps_eor_hp, weights=ln_post, axis=0)
# Confidence interval of delay power spectrum posteriors
percentile = args.conf_interval/2 + 50
dps_eor_hp_ubound = np.percentile(dps_eor_hp, percentile, axis=0)
dps_eor_hp_lbound = np.percentile(dps_eor_hp, 100-percentile, axis=0)
dps_eor_hp_err = np.vstack((
    dps_eor_hp_pwm - dps_eor_hp_lbound,
    dps_eor_hp_ubound - dps_eor_hp_pwm
))

# Plot the true and recovered delay power spectra
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(delays, dps_eor_true, "k:", label="True")
ax.errorbar(
    delays,
    dps_eor_hp_pwm,
    yerr=dps_eor_hp_err,
    color="k",
    ls="",
    marker="o",
    capsize=3,
    label=f"Recovered ({args.conf_interval}% Confidence)"
)
ax.legend(loc="upper right")
ax.set_xlabel(r"$\tau$ [ns]")
ax.set_ylabel(r"$P(\tau)$ [arb. units]")
ax.set_title("EoR Delay Power Spectrum Comparison")
ax.set_yscale("log")
ax.set_ylim(ymin=0.1, ymax=3)
ax.grid()
fig.tight_layout()
plt.show()
