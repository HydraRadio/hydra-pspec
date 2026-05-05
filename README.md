# hydra-pspec

Gibbs-sampling power spectrum estimator for the EoR (Epoch of Reionization)
signal in radio interferometric visibilities. Jointly samples the EoR signal,
foreground amplitudes, multiplicative antenna gain systematics, and the EoR
delay power spectrum — all within a single self-consistent Bayesian framework.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Environment setup](#environment-setup)
3. [Quick start — synthetic simulation](#quick-start--synthetic-simulation)
4. [MPI example — three baselines](#mpi-example--three-baselines)
5. [Covariance matrix calculation](#covariance-matrix-calculation)
6. [Core package API](#core-package-api)
   - [pspec.py — Gibbs sampler](#pspecpy--gibbs-sampler)
   - [sys_solver.py — systematics solver](#sys_solverpy--systematics-solver)
   - [utils.py — utilities](#utilspy--utilities)
   - [oqe.py — optimal quadratic estimator](#oqepy--optimal-quadratic-estimator)
   - [dpss.py and lssa.py — foreground fitting](#dpsspy-and-lssapy--foreground-fitting)
7. [Outputs](#outputs)
8. [Known issues](#known-issues)

---

## Repository layout

```
hydra-pspec/
├── hydra_pspec/                     # Core Python package
│   ├── __init__.py
│   ├── pspec.py                     # Gibbs sampler, GCR solver, power spectrum sampling
│   ├── sys_solver.py                # Multiplicative systematics GCR solver
│   ├── utils.py                     # Fourier operators, I/O helpers, pyuvdata utilities
│   ├── oqe.py                       # Optimal quadratic estimator (reference implementation)
│   ├── dpss.py                      # DPSS-based foreground mode fitting
│   ├── lssa.py                      # LSSA sinusoid fitting
│   └── plotting_functions.py        # Waterfall plot helpers
│
├── scripts/
│   ├── simple_example.py            # Minimal usage example (requires hera_pspec)
│   ├── calc-vis-cov-matrices.py     # Compute per-baseline visibility covariance matrices
│   ├── 3bl-example/
│   │   ├── 3bl-example.md           # MPI example documentation
│   │   └── config-3bl-mpi.yaml      # Config for 3-baseline MPI run
│   └── scaling_tests/               # Strong-scaling benchmarks and SLURM helpers
│
├── res/
│   ├── npy_data/                    # Pre-computed synthetic visibilities
│   │   ├── eor_true.npy             # EoR visibilities
│   │   ├── fg_true.npy              # Foreground visibilities
│   │   ├── freqs_full.npy           # Frequency array (Hz)
│   │   ├── lsts_full.npy            # LST array
│   │   └── sky_true.npy             # Combined sky model (EoR + FG)
│   └── test_data/                   # Airy beam test dataset
│       ├── README.md                # Test data documentation
│       ├── config.yaml              # Paths and dataset parameters
│       ├── vis-eor-fgs.uvh5         # EoR + FG visibilities  (Ntimes=203, Nfreqs=120)
│       ├── vis-eor.uvh5             # EoR-only visibilities
│       ├── vis-ptsrc-gsm.uvh5       # Point source + GSM model
│       ├── vis-eor-ptsrc-gsm.uvh5   # EoR + point source + GSM
│       ├── eor-cov.npy              # EoR frequency–frequency covariance  (120, 120)
│       ├── fgmodes.npy              # Foreground basis matrix              (120, 120)
│       ├── noise.npy                # Noise visibilities                   (203, 120)
│       └── noise-cov.npy            # Noise covariance (diagonal)          (120, 120)
│
├── config/
│   ├── py10.yml                     # Full conda environment specification
│   └── py10_clean.yml               # Minimal conda environment specification
│
├── sys_sampler_wrapper.py           # Synthetic data simulation and sampler wrapper
└── README.md
```

---

## Environment setup

Create and activate the conda environment from the provided spec:

```bash
conda env create -f config/py10_clean.yml
conda activate py10
```

Install the package in development mode from the repository root:

```bash
pip install -e .
```

**Dependencies**

| Package | Purpose |
|---|---|
| Python ≥ 3.8 | |
| numpy | Array operations |
| scipy | Linear solvers, optimisation |
| h5py | HDF5 streaming output |
| multiprocess | Parallelised per-time GCR solves |
| pyuvdata | Reading/writing `.uvh5` files |
| uvtools | Window functions, FFT utilities |
| astropy | Units and constants |
| scikit-learn | PCA / SVD utilities |
| matplotlib | Plotting |
| tqdm | Progress bars |
| mpi4py | MPI parallelism *(optional)* |

---

## Quick start — synthetic simulation

`sys_sampler_wrapper.py` generates a synthetic dataset (EoR + foregrounds +
multiplicative gain systematics) and runs the full Gibbs sampler.

```bash
python sys_sampler_wrapper.py
```

### Simulation parameters

| Parameter | Default | Description |
|---|---|---|
| `Ntimes` | 80 | Number of time integrations |
| `Nfreqs` | 60 | Number of frequency channels |
| `Nfgmodes` | 10 | Number of Legendre foreground modes |
| `Niter` | 5 | Number of Gibbs iterations |
| `noise_ps_val` | 0.0004 | Noise power spectrum (arb. units) |
| `nm_list` | `[(10,0),(11,0),(12,0),(13,0)]` | (delay-mode, fringe-rate-mode) pairs for systematics |
| `sys_amps_true` | `[1+4j, 2+3j, 3+2j, 4+1j]` | True systematic amplitudes |
| `dummy_flag` | `False` | If `True`, use random data instead of `res/npy_data/` |

### Data model

The synthetic visibility is constructed as:

```
d = (1 + g) * (fg + eor) + noise
```

where `g = sys_modes @ sys_amps` is the multiplicative gain perturbation
expressed in the 2D Fourier (delay–fringe-rate) basis.

### Outputs

Results are saved to `./tests/` by default. See the [Outputs](#outputs) section
for full details.

---

## MPI example — three baselines

The `scripts/3bl-example/` directory contains a ready-to-run MPI example using
the test data in `res/test_data/`.

```bash
mpirun -n 3 python run-hydra-pspec.py --config scripts/3bl-example/config-3bl-mpi.yaml
```

Each MPI rank processes one baseline. Per-baseline results are written to
subdirectories of the configured output directory.

See `scripts/3bl-example/3bl-example.md` for a full walkthrough.

### Scaling tests

`scripts/scaling_tests/` provides tools for strong-scaling benchmarks:

- `set_up_scaling_data.py` — generate synthetic data for scaling runs
- `plot_speed_up.py` — plot speed-up curves from timing logs
- `config-scaling-test.yaml` — reference configuration
- `scaling_tests_README.md` — detailed instructions

---

## Covariance matrix calculation

`scripts/calc-vis-cov-matrices.py` computes time-averaged frequency–frequency
covariance matrices from a visibility dataset. These covariances can be used to
initialise or constrain the foreground model.

```bash
python scripts/calc-vis-cov-matrices.py data/*.uvh5 \
    --out-dir cov_out/ \
    --freq-range 100-200 \
    --eig
```

Outputs are written to per-baseline subdirectories of `--out-dir`:

| File | Description |
|---|---|
| `cov.npy` | Frequency–frequency covariance matrix |
| `evecs.npy` | Eigenvectors *(with `--eig`)* |
| `evals.npy` | Eigenvalues *(with `--eig`)* |

---

## Core package API

### `pspec.py` — Gibbs sampler

The main entry point is `gibbs_sample()`, which runs a full Gibbs chain for a
single baseline.

#### `gibbs_sample`

```python
from hydra_pspec.pspec import gibbs_sample

signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post = gibbs_sample(
    vis,                  # (Ntimes, Nfreqs) complex — visibility data
    flags,                # (Nfreqs,)        bool/int — 1=unflagged, 0=flagged
    Ninv,                 # (Nfreqs, Nfreqs) or (Ntimes, Nfreqs, Nfreqs) — inverse noise cov
    freqs,                # (Nfreqs,)        float — frequency array in Hz
    lsts,                 # (Ntimes,)        float — LST array
    signal_ps_initial,    # (Nfreqs,)        float — initial EoR delay power spectrum
    signal_ps_prior,      # (2, Nfreqs)      float — [lower, upper] prior bounds
    fg_modes,             # (Nfreqs, Nmodes) complex — foreground basis (e.g. Legendre)
    sys_modes,            # (Ntimes*Nfreqs, Nsys_modes) complex — systematics basis
    sys_prior,            # (Nsys_modes, Nsys_modes) float — systematics prior covariance
    sys_initial,          # (Nsys_modes,)    complex — initial systematics amplitudes
    sky_model_initial=None,
    Niter=100,
    seed=None,
    sample_systematics=True,
    sample_eor_fg=True,
    sample_signal_ps=True,
    solver='lgmres',
    solver_tol=1e-12,
    verbose=True,
    nproc=1,
    write_Niter=100,      # write numpy files every this many iterations
    out_dir=None,         # output directory (Path or str); None = do not save
    map_estimate=False,   # return MAP estimate only (no posterior sampling)
)
```

Returns a 6-tuple — see [Outputs](#outputs) for shapes and descriptions.

#### Other functions in `pspec.py`

| Function | Description |
|---|---|
| `gibbs_step(vis, flags, Ninv, signal_ps, signal_ps_prior, fg_modes, sys_modes, sys_amps, sys_prior, iter, ...)` | Single Gibbs iteration; returns the same 6-tuple as `gibbs_sample` but for one step |
| `gcr_fg_and_signal(vis, flags, fg_modes, Nparams, sys_model, signal_ps, Ninv, fourier_op, nproc=1, ...)` | Parallelised GCR solver for the joint EoR + FG draw across all times |
| `gcr_fg_and_signal_per_time(idx, vis, Einv, sqrtE, sqrtNinv, Nparams, sys_model, flags, Ninv, fg_modes, ...)` | GCR solve for a single time sample; called internally by `gcr_fg_and_signal` |
| `sample_pspec(s, prior, ngrid=120, sk=None, max_prior_iter=10000)` | Draw EoR delay power spectrum sample from inverse-gamma posterior |
| `sprior(signals, bins, factor)` | Compute uniform prior bounds on the delay power spectrum from data |
| `covariance_from_pspec(ps, fourier_op)` | Convert delay power spectrum → frequency–frequency covariance matrix |
| `goodness_of_fit_statistics(data, data_model, flags, Ninv, signal_amps, Sinv, ...)` | Compute χ² and log-posterior |
| `draw_icdf_samples(alpha, beta, x)` | Draw one sample from an inverse-gamma distribution via inverse CDF |
| `data_dly_fr(data, freqs, times, windows=None, ...)` | 2D FFT: freq–time → delay–fringe-rate |

---

### `sys_solver.py` — systematics solver

Constructs the 2D Fourier basis for multiplicative gain systematics and provides
the GCR sampler that draws systematics amplitudes conditioned on the sky model.

#### `sys_modes`

```python
from hydra_pspec.sys_solver import sys_modes

S = sys_modes(
    freqs_Hz,   # (Nfreqs,) float — frequency array in Hz
    times_sec,  # (Ntimes,) float — time array in seconds
    modes,      # list of (delay_mode, fr_mode) integer pairs
)
# Returns: S with shape (Nfreqs*Ntimes, Nmodes), complex
```

#### `fourier_mode_2d`

```python
from hydra_pspec.sys_solver import fourier_mode_2d

basis_fns, kfreq, ktime = fourier_mode_2d(
    freqs_Hz,   # (Nfreqs,) float
    times_sec,  # (Ntimes,) float
    modes,      # list of (nfreq, ntime) integer pairs
)
# basis_fns: (Nmodes, Nfreqs, Ntimes) complex
# kfreq: delay values in ns
# ktime: fringe-rate values in mHz
```

#### `gcr_systematics`

```python
from hydra_pspec.sys_solver import gcr_systematics

sys_amps = gcr_systematics(
    data,        # (Ntimes, Nfreqs)             complex — visibility data
    Ninv,        # (Nfreqs, Nfreqs) or (Ntimes, Nfreqs, Nfreqs) — inverse noise cov
    sky_model,   # (Ntimes, Nfreqs)             complex — EoR + FG model
    sys_modes,   # (Ntimes*Nfreqs, Nsys_modes)  complex — systematics basis
    sys_prior,   # (Nsys_modes, Nsys_modes)     float   — prior covariance
    solver_tol=1e-12,
    verbose=False,
)
# Returns: sys_amps (Nsys_modes,) complex
```

#### Other functions in `sys_solver.py`

| Function | Description |
|---|---|
| `cholesky_inverse(A)` | Invert a positive-definite matrix via Cholesky decomposition |
| `sq_mat_tr(A, flag='r')` | Reshape (2,2,n,n) block matrix to (2n, 2n) square matrix |
| `inv_mat(mat)` | Invert a diagonal matrix without `np.linalg.inv` |

---

### `utils.py` — utilities

#### I/O helpers

| Function | Description |
|---|---|
| `write_numpy_files(fp, signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post)` | Write all sampler output arrays to `fp/` as `.npy` files |
| `append_gibbs_sample_h5(fp, overwrite=False, flush=True, batch_axis=None, **arrays)` | Append one Gibbs sample to `fp/gibbs_samples.h5` (resizable HDF5 datasets) |
| `add_mtime_to_filepath(fp, join_char="-")` | Rename a file/directory by appending its modification time |

#### Signal processing

| Function | Description |
|---|---|
| `fourier_operator(n, unitary=True)` | Build an n×n DFT matrix using the fftshift convention |
| `naive_pspec(data, subtract_mean=True, taper=True)` | Quick delay power spectrum via FFT with optional Blackman-Harris taper |
| `calc_ps(s)` | Delay power spectrum from signals array (Nobs, Nfreqs); uses inverse-FFT normalisation |
| `trim_flagged_channels(w, x)` | Remove flagged channels from 1D or 2D arrays given a boolean mask |

#### pyuvdata helpers

| Function | Description |
|---|---|
| `form_pseudo_stokes_vis(uvd, convention=1.0)` | Form pseudo-Stokes I from XX and YY polarisations in a UVData object |
| `filter_freqs(freq_str, freqs_in)` | Parse a frequency selection string (`'100-200'`, `'100,120,150'`) and return a filtered Quantity array |

#### Miscellaneous

| Function | Description |
|---|---|
| `get_git_version_info(directory=None)` | Return dict with current git hash, branch, and remote URL |

---

### `oqe.py` — optimal quadratic estimator

Reference implementation of the Optimal Quadratic Estimator (OQE) for the EoR
delay power spectrum. Intended for comparison with the Gibbs sampler output.

Key functions:

| Function | Signature | Description |
|---|---|---|
| `q_h(V, s, R, taper=None)` | `(V, s, R)` → `qhat` | HERA-style cross-correlation quadratic estimate |
| `q_hp(V, s, R, ncpu)` | `(V, s, R, ncpu)` → `qhat` | Parallelised version of `q_h` |
| `F(s, R)` | `(s, R)` → `F` | Fisher information matrix |
| `Ft(s, R)` | `(s, R)` → `F` | Optimised (trace-normalised) Fisher matrix |
| `p(q, M)` | `(q, M)` → `p` | Apply a weighting/normalisation matrix M to raw estimates q |
| `getqs(Vis, R)` | `(Vis, R)` → `(q, F)` | Full OQE pipeline: compute estimates and Fisher matrix |
| `bias(tau, s, R, C_noise_total)` | — | Noise bias term for the quadratic estimate |
| `Sig_QEN(R, C_noise, norm)` | — | Noise variance of the quadratic estimate |
| `Sig_QESN(R, C_noise, C_signal, norm)` | — | Signal + noise variance |

---

### `dpss.py` and `lssa.py` — foreground fitting

These modules provide alternative foreground fitting methods that can be used to
construct the foreground mode matrix passed to `gibbs_sample`.

#### `dpss.py`

```python
from hydra_pspec.dpss import dpss_fit_modes

dpss_modes, amps = dpss_fit_modes(
    d,                       # complex data (flagged channels removed)
    w,                       # flag array (Nfreqs,)
    freqs,                   # frequency array in MHz
    cov,                     # covariance matrix model
    nmodes=10,               # number of DPSS modes
    alpha=1.,                # bandwidth factor
    taper=None,              # optional taper function
    minimize_method='L-BFGS-B',
)
# dpss_modes: (nmodes, nfreqs)
# amps: (2*nmodes,) — real and imaginary amplitude pairs
```

#### `lssa.py`

```python
from hydra_pspec.lssa import lssa_fit_modes

tau, param1, param2 = lssa_fit_modes(
    d,                       # complex data (flagged channels removed)
    freqs,                   # frequency array in MHz
    invcov=None,             # inverse covariance matrix (optional)
    fit_amp_phase=True,      # True → fit (amplitude, phase); False → fit (Re, Im)
    tau=None,                # delay modes (default: fftfreq of freqs)
    taper=None,              # optional taper
    minimize_method='L-BFGS-B',
)
```

Additional helper in `lssa.py`:

| Function | Description |
|---|---|
| `decorr_matrix(w, tau, freqs)` | Rotation matrix to decorrelate real/imaginary amplitude errors |
| `decorr_pspec(A_re, A_im, w, tau, freqs)` | LSSA delay power spectrum with decorrelation weighting |

---

## Outputs

### Return values of `gibbs_sample`

| Array | Shape | Description |
|---|---|---|
| `signal_amps` | (Niter, Ntimes, Nfreqs) | Sampled EoR signal visibilities |
| `signal_ps` | (Niter, Nfreqs) | Sampled EoR delay power spectrum |
| `fg_amps` | (Niter, Ntimes, Nmodes) | Sampled foreground amplitudes |
| `sys_amps` | (Niter, Nsys_modes) | Sampled systematics coefficients |
| `chisq` | (Niter, Ntimes, Nfreqs) | χ² per iteration, time, and frequency |
| `ln_post` | (Niter,) | Log-posterior per iteration |

### Files written to `out_dir`

When `out_dir` is provided, numpy files are written every `write_Niter`
iterations and an HDF5 file is appended at every iteration.

| File | Shape | Description |
|---|---|---|
| `gcr-eor.npy` | (Niter, Ntimes, Nfreqs) | Sampled EoR visibilities |
| `dps-eor.npy` | (Niter, Nfreqs) | Sampled EoR delay power spectrum |
| `fg-amps.npy` | (Niter, Ntimes, Nmodes) | Sampled foreground amplitudes |
| `b-sys.npy` | (Niter, Nsys_modes) | Sampled systematics coefficients |
| `chisq.npy` | (Niter, Ntimes, Nfreqs) | χ² values |
| `ln-post.npy` | (Niter,) | Log-posterior values |
| `gibbs_samples.h5` | — | HDF5 file with all of the above, appended per iteration |

The HDF5 file avoids accumulating large arrays in memory and is safe to read
while the sampler is still running (datasets are flushed after each append).

---

## Known issues

- **`scripts/simple_example.py`**: Calls the deprecated `gibbs_sample_with_fg`
  API. This script is not functional on the current branch.
- **Test scripts with hardcoded paths**: Some older test files use absolute
  paths or paths that assume a non-standard working directory. Check the path
  at the top of each script before running.
- **No `run-hydra-pspec.py` on this branch**: The MPI runner referenced in the
  `scripts/3bl-example/` config is not committed here; the 3-baseline example
  serves as documentation for the expected calling convention.
