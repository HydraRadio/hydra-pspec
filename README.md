# hydra-pspec-systematic

Gibbs-sampling power spectrum estimator for the EoR (Epoch of Reionization)
signal in radio interferometric visibilities. Jointly samples the EoR signal,
foreground amplitudes, multiplicative antenna gain systematics, and the EoR
delay power spectrum, all within a single self-consistent Bayesian framework.

---

## Repository layout

```
hydra-pspec/
├── hydra_pspec/
│   ├── __init__.py
│   ├── pspec.py              # Gibbs sampler (main module)
│   ├── sys_solver.py         # Multiplicative systematics GCR solver
│   ├── utils.py              # I/O, Fourier operators, and utility functions
│   ├── dpss.py               # DPSS foreground fitting
│   ├── lssa.py               # LSSA foreground fitting
│   ├── oqe.py                # Optimal Quadratic Estimator (reference)
│   ├── plotting_functions.py # Diagnostic plotting
│   └── config_plots.py       # Plot configuration
├── scripts/
│   ├── calc-vis-cov-matrices.py  # Compute visibility covariance matrices
│   ├── simple_example.py         # Minimal usage example (legacy API)
│   ├── 3bl-example/              # 3-baseline MPI example
│   │   ├── config-3bl-mpi.yaml
│   │   └── 3bl-example.md
│   └── scaling_tests/            # MPI strong-scaling benchmarks
└── README.md
```

---

## Environment

```bash
conda activate hydra   # Python 3.10.14
```

Install the package in development mode:

```bash
pip install -e .
```

**Dependencies:**

| Package | Notes |
|---|---|
| Python ≥ 3.8 | |
| numpy | |
| scipy | |
| h5py | Streaming HDF5 output |
| multiprocess | Parallelised GCR solver |
| pyuvdata | Reading/writing .uvh5 files |
| uvtools | Fourier transforms and tapering |
| astropy | Units, constants |
| scikit-learn | Matrix utilities in `sys_solver` |
| matplotlib | |
| tqdm | Progress bars |
| mpi4py | MPI parallelism (optional, for multi-baseline runs) |

---

## Quick start — 3-baseline MPI example

The `scripts/3bl-example/` directory provides a minimal working example of
running the Gibbs sampler over 3 baselines in parallel with MPI.

### Setup

Place visibility data and auxiliary files under `scripts/3bl-example/test_data/`:

- `vis-eor-fgs.uvh5` — input visibilities (`.uvh5` format)
- `fgmodes.npy` — foreground mode matrix, shape `(Nfreqs, Nfgmodes)`
- `eor-cov.npy` — initial EoR signal covariance
- `noise.npy`, `noise-cov.npy` — noise model

### Run

```bash
cd scripts/3bl-example
mpirun -n 3 python run-hydra-pspec.py --config config-3bl-mpi.yaml
```

Key configuration parameters (from `config-3bl-mpi.yaml`):

| Parameter | Value | Description |
|---|---|---|
| `ant_str` | `"0_1,0_3,0_5"` | Comma-separated antenna pairs to process |
| `seed` | 7123689 | Random seed |
| `Niter` | 1000 | Gibbs iterations |
| `Nfgmodes` | 12 | Number of foreground modes |
| `ps_prior_lo` / `ps_prior_hi` | 0.1 / 2 | Power spectrum prior bounds |
| `dirname` | `results-seed-7123689-Niter-1000/` | Output directory |

### Outputs

Written to `dirname/` per baseline:

| File | Shape | Description |
|---|---|---|
| `gcr-eor.npy` | (Niter, Ntimes, Nfreqs) | Sampled EoR visibilities |
| `dps-eor.npy` | (Niter, Nfreqs) | Sampled EoR delay power spectrum |
| `fg-amps.npy` | (Niter, Ntimes, Nfgmodes) | Sampled foreground amplitudes |
| `b-sys.npy` | (Niter, Nsys_modes) | Sampled systematic amplitudes |
| `chisq.npy` | (Niter, Ntimes, Nfreqs) | χ² per iteration |
| `ln-post.npy` | (Niter,) | Log-posterior per iteration |
| `gibbs_samples.h5` | — | Streaming HDF5 copy of all samples |

`gibbs_samples.h5` is written incrementally one row per iteration (no large
in-memory accumulation) and contains the same quantities as the `.npy` files
under dataset keys `signal_amps`, `signal_ps`, `fg_amps`, `sys_amps`,
`chisq`, `ln_post`.

---

## Covariance matrix calculation — `scripts/calc-vis-cov-matrices.py`

Computes time-averaged frequency–frequency covariance matrices from a
visibility dataset, used to initialise the foreground model in `run-hydra-pspec.py`.

```bash
conda run -n py10 python scripts/calc-vis-cov-matrices.py \
    data/*.uvh5 \
    --out-dir cov_out/ \
    --freq-range 100-200 \
    --eig
```

Outputs per-baseline subdirectory: `cov.npy`, `evecs.npy`, `evals.npy`.

---

## Core package API — `hydra_pspec/`

### `pspec.py` — Gibbs sampler

The main entry point is `gibbs_sample()`. All sampling toggles
(`sample_eor_fg`, `sample_systematics`, `sample_signal_ps`) are exposed as
arguments to `gibbs_step()`.

```python
from hydra_pspec.pspec import gibbs_sample

signal_amps, signal_ps, fg_amps, sys_amps, chisq, ln_post = gibbs_sample(
    vis, flags, Ninv, freqs, lsts,
    signal_ps_initial, signal_ps_prior,
    fg_modes, sys_modes, sys_prior, sys_initial,
    Niter=1000,
    out_dir='results/',
    seed=42,
)
```

Key functions:

| Function | Description |
|---|---|
| `gibbs_sample(...)` | Full Gibbs chain — main entry point |
| `gibbs_step(...)` | Single Gibbs iteration |
| `gcr_fg_and_signal(...)` | GCR solver for joint EoR + FG draw (all times) |
| `gcr_fg_and_signal_per_time(...)` | GCR solver for a single time sample |
| `sample_pspec(...)` | Draw from inverse-gamma power spectrum posterior |
| `draw_icdf_samples(alpha, beta, x)` | Inverse-CDF sampler for inverse-gamma distribution |
| `sprior(signals, bins, factor)` | Compute data-driven prior bounds on the power spectrum |
| `covariance_from_pspec(...)` | Convert delay PS to freq–freq covariance |
| `goodness_of_fit_statistics(...)` | χ² and log-posterior |
| `data_dly_fr(...)` | 2D FFT: freq–time → delay–fringe-rate |

### `sys_solver.py` — systematics

| Function | Description |
|---|---|
| `fourier_mode_2d(freqs_Hz, times_sec, modes)` | Build 2D Fourier basis from (delay, fr) mode pairs |
| `sys_modes(freqs_Hz, times_sec, modes)` | Systematic mode operator `(Nfreqs×Ntimes, Nmodes)` |
| `gcr_systematics(...)` | GCR draw for multiplicative systematic amplitudes |
| `cholesky_inverse(A)` | Invert positive-definite matrix via Cholesky decomposition |

### `utils.py` — utilities

| Function | Description |
|---|---|
| `fourier_operator(n)` | Dense n×n DFT matrix (fftshift convention) |
| `naive_pspec(data, ...)` | Quick delay power spectrum via FFT |
| `write_numpy_files(fp, ...)` | Write all sampler output arrays to `fp/` as `.npy` |
| `append_gibbs_sample_h5(fp, ...)` | Stream one Gibbs sample at a time to `fp/gibbs_samples.h5` |
| `get_git_version_info()` | Retrieve current git hash and branch |
| `form_pseudo_stokes_vis(uvd)` | UVData XX+YY → pseudo-Stokes I |
| `filter_freqs(freq_str, freqs)` | Parse frequency selection string to Quantity array |
| `add_mtime_to_filepath(fp)` | Append modification timestamp to a filename or directory |

### `dpss.py` and `lssa.py` — foreground fitting

| Function | Description |
|---|---|
| `dpss_fit_modes(d, w, freqs, cov, ...)` | Weighted DPSS fit to masked complex 1D data |
| `lssa_fit_modes(d, freqs, invcov, ...)` | Weighted LSSA sinusoid fit to masked data |

### `oqe.py` — Optimal Quadratic Estimator

Reference implementation of the OQE power spectrum estimator for comparison
with the Gibbs sampler output.

| Function | Description |
|---|---|
| `Q(tau, s)` | Band-power matrix for delay mode `tau` |
| `qhat(x, tau, s, R, bias)` | Unnormalised bandpower estimate (auto-correlation) |
| `qhat_h(x1, x2, tau, s, R)` | Cross-power bandpower estimate (HERA-style) |
| `F(s, R)` | Fisher information matrix |
| `q_h(V, s, R, ...)` | Bandpower estimates over all τ for a set of visibility pairs |
| `p(q, M)` | Normalised power spectrum estimate |
| `getqs(Vis, R)` | Full OQE pipeline — builds F, returns `qs`, `MB`, `MA` |

---
