# hydra-pspec-systematic

Gibbs-sampling power spectrum estimator for the EoR (Epoch of Reionization)
signal in radio interferometric visibilities. Jointly samples the EoR signal,
foreground amplitudes, multiplicative antenna gain systematics, and the EoR
delay power spectrum, all within a single self-consistent Bayesian framework.

---

## Repository layout

```
hydra_copy/
├── sohini_test.py                   # Main simulation entry point
├── run-hydra-pspec.py               # MPI runner for real HERA visibility data
│
├── hydra_pspec/                     # Core Python package
│   ├── __init__.py
│   ├── pspec.py                     # Gibbs sampler, GCR solver, power spectrum sampling
│   ├── sys_solver.py                # Multiplicative systematics GCR solver
│   ├── utils.py                     # Fourier operators, I/O helpers, git utilities
│   ├── dpss.py                      # DPSS-based foreground mode fitting
│   ├── lssa.py                      # LSSA sinusoid fitting
│   ├── oqe.py                       # Optimal quadratic estimator
│   ├── post_processing_funcs.py     # Results plotting and analysis (~3400 lines)
│   └── plotting_functions.py        # Waterfall plot helpers
│
├── scripts/
│   ├── calc-vis-cov-matrices.py     # Compute per-baseline visibility covariance matrices
│   ├── extract_hera_val_data.py     # Extract .npy arrays from HERA validation uvh5 files
│   └── hera_val_gibbs_wrapper.py    # Gibbs sampler wrapper for HERA validation data
│
├── tests/
│   ├── example.py                   # Basic usage example (uses older API — see Known Issues)
│   ├── phil_test.py                 # Synthetic EoR/FG comparison test
│   ├── solver_comps.py              # Solver comparison (low_dl_fr_0 config)
│   └── filtered_and_masked_run.py   # Masked/filtered data test (high_dl_fr_0 config)
│
├── tools/
│   ├── hera_val/
│   │   ├── sim_prep.py              # hera_sim integration — add crosstalk/reflections/noise
│   │   ├── sim_prep_old.py              # older version of sim_prep.py
│   │   ├── sim_config.yaml          # Systematics injection parameters
│   │   ├── test-3.2.0.ipynb         # HERA validation notebook (loads from res/hera_val_npy/)
│   │   ├── test_2.ipynb         # Takes npy visibilities and injects uncertainties into it. 
│   │   ├── Sys_Gen.ipynb            # Systematics generation notebook
│   │   ├── RIMEz_beam_poly.npy      # RIMEz beam polynomial for noise modelling
│   │   └── nm_list.npz              # Pre-computed systematic mode indices
│   ├── test_run_sohini.sh           # Sequential runner for all three simulation cases
│   └── create_jobscript.sh          # SLURM jobscript generator (strong-scaling tests)
│
├── res/
│   ├── hydra_ascii.txt              # ASCII art banner
│   ├── npy_data/
│   │   ├── fg_true.npy              # Pre-computed foreground visibilities
│   │   └── eor_true.npy             # Pre-computed EoR visibilities
│   ├── hera_val_npy/                # Pre-extracted HERA validation arrays
│   │   ├── vis_fg_eor.npy           # FG + EoR visibility       (203, 120) complex
│   │   ├── vis_fg.npy               # FG-only visibility        (203, 120) complex
│   │   ├── vis_eor.npy              # EoR-only visibility       (203, 120) complex
│   │   ├── vis_corrupted.npy        # FG + EoR + systematics    (203, 120) complex
│   │   ├── freqs_Hz.npy             # Frequency array           (120,)
│   │   ├── lsts_rad.npy             # LST array                 (203,)
│   │   ├── nm_list.npy              # Systematic mode indices   (61, 2)
│   │   └── mask_indices.npy         # Delay–fringe-rate mask    (2, 61)
│   └── test_data/                   # Airy beam test dataset (uvh5 + covariances)
│       ├── vis-eor.uvh5
│       ├── vis-eor-fgs.uvh5
│       ├── vis-eor-ptsrc-gsm.uvh5
│       ├── vis-ptsrc-gsm.uvh5
│       ├── eor-cov.npy
│       ├── fgmodes.npy
│       ├── noise.npy
│       ├── noise-cov.npy
│       ├── config.yaml
│       └── README.md                # Test data documentation
│
├── paper_plots/
│   └── 100k_runs/
│       ├── low_dl_fr_0/             # Case I results  (nm = 3–6,   fr = 0)
│       ├── high_dl_fr_0/            # Case II results (nm = 10–13, fr = 0) — default
│       └── low_dl_fr_20/            # Case III results (nm = 3–6,  fr = 20)
│
├── outputs/                         # Run logs and test outputs
├── config/
│   └── pyproject.toml               # Build metadata and dependencies
└── README.md
```

---

## Environment

```bash
conda activate py10   # Python 3.10.14
```

Install the package in development mode:

```bash
pip install -e .
```

**Dependencies** (from `config/pyproject.toml`):

| Package | Notes |
|---|---|
| Python ≥ 3.8 | |
| numpy | |
| scipy | |
| multiprocess | Parallelised GCR solver |
| pyuvdata | Reading/writing .uvh5 files |
| astropy | Units, constants |
| jsonargparse | CLI argument parsing in `run-hydra-pspec.py` |
| matplotlib | |
| mpi4py | MPI parallelism in `run-hydra-pspec.py` |

---

## Quick start — synthetic simulations

`sohini_test.py` generates synthetic visibility data (EoR + foregrounds +
multiplicative gain systematics) and runs the full Gibbs sampler.

### Simulation cases

Three configurations are defined; uncomment the desired block near the top of
`sohini_test.py`:

| Case | `nm_list` (delay-mode, fr-mode) pairs | `op_dir` |
|---|---|---|
| I | (3,0),(4,0),(5,0),(6,0) | `paper_plots/100k_runs/low_dl_fr_0/` |
| II *(default)* | (10,0),(11,0),(12,0),(13,0) | `paper_plots/100k_runs/high_dl_fr_0/` |
| III | (3,20),(4,20),(5,20),(6,20) | `paper_plots/100k_runs/low_dl_fr_20/` |

Cases differ in which delay–fringe-rate Fourier modes are used to model the
multiplicative gain systematics.

### Run

```bash
conda run -n py10 python sohini_test.py
```

The output directory (`op_dir`) is created automatically.

### Key simulation parameters

| Parameter | Value | Description |
|---|---|---|
| `Ntimes` | 80 | Time integrations |
| `Nfreqs` | 60 | Frequency channels |
| `Nfgmodes` | 10 | Foreground Legendre modes |
| `Niter` | 100 000 | Gibbs iterations |
| `noise_ps` | 0.0004 | Noise power spectrum (arb. units) |
| `eor_ps` | `0.0012 × (1 + 0.3 sin(...))` | EoR delay power spectrum |
| `sys_amps_true` | `[1+4j, 2+3j, 3+2j, 4+1j]` | True systematic amplitudes |

### Outputs

Saved to `op_dir`:

| File | Shape | Description |
|---|---|---|
| `gain_true.npy` | (Ntimes, Nfreqs) | True systematic gain model |
| `eor_true.npy` | (Ntimes, Nfreqs) | True EoR visibilities |
| `fg_true.npy` | (Ntimes, Nfreqs) | True foreground visibilities |
| `data_true.npy` | (Ntimes, Nfreqs) | Total data = EoR + FG + systematics |
| `fgmodes.npy` | (Nfreqs, Nfgmodes) | Foreground basis matrix |
| `gcr-eor.npy` | (Niter, Ntimes, Nfreqs) | Sampled EoR visibilities |
| `dps-eor.npy` | (Niter, Nfreqs) | Sampled EoR delay power spectrum |
| `fg-amps.npy` | (Niter, Nfgmodes) | Sampled foreground amplitudes |
| `b-sys.npy` | (Niter, Nsys_modes) | Sampled systematic amplitudes |
| `chisq.npy` | (Niter,) | χ² per iteration |
| `ln-post.npy` | (Niter,) | Log-posterior per iteration |

---

## Running on real HERA data — `run-hydra-pspec.py`

MPI-based runner for processing real HERA visibility files (`.uvh5`/`.miriad`).
Baselines are distributed across MPI ranks; within each rank, per-time GCR
solves can be further parallelised with `--nproc`.

```bash
mpirun -n <nranks> conda run -n py10 python run-hydra-pspec.py \
    --file_paths data/*.uvh5 \
    --out_dir    results/ \
    --Niter      1000 \
    --Nfgmodes   8 \
    --freq_range 100-200 \
    --nproc      4
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--file_paths` | *(required)* | Input UVData files |
| `--out_dir` | `./out/` | Output directory |
| `--Niter` | 100 | Gibbs iterations |
| `--Nfgmodes` | 8 | Foreground modes |
| `--freq_range` | all | Frequency selection in MHz: `'100'`, `'100-200'`, `'100,120,150'` |
| `--ant_str` | all | Antenna / baseline selection |
| `--nproc` | 1 | Multiprocessing workers per MPI rank |
| `--clobber` | False | Overwrite existing per-baseline output directories |
| `--map_estimate` | False | Return MAP estimate only (no full posterior sampling) |

Per-baseline results are written to `out_dir/<ant1>-<ant2>/` using the same
file naming convention as `sohini_test.py`.

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

## HERA validation workflow — `tools/hera_val/`

### 1. Extract .npy arrays from raw uvh5

```bash
conda run -n py10 python scripts/extract_hera_val_data.py
```

Loads `res/test_data/*.uvh5`, forms pseudo-Stokes I, runs a 2D delay–fringe-rate
transform, fits a Gaussian mask to identify systematic modes, and writes all
arrays to `res/hera_val_npy/`.

### 2. Run the validation notebook

`tools/hera_val/test-3.2.0.ipynb` loads directly from `res/hera_val_npy/` and
produces four-panel waterfall plots (freq–time, freq–fringe-rate, delay–time,
delay–fringe-rate) for:
- Clean FG + EoR visibilities
- Corrupted visibilities (FG + EoR + injected systematics)
- Injected systematics (corrupted − clean)

### Systematics injection parameters (`tools/hera_val/sim_config.yaml`)

| Component | Parameters |
|---|---|
| `reflection_spectrum` | 20 reflections, delays 200–1200 ns, amps 10⁻³–10⁻⁴ |
| `reflections` | 2 discrete at 200 ns (amp 0.03) and 1200 ns (amp 0.008) |
| `xtalk` | 10 cross-coupling copies, delays 900–1300 ns, amps 10⁻⁴–10⁻⁶ |

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
| `gcr_fg_and_signal(...)` | Parallelised GCR solver for EoR + FG joint draw |
| `sample_pspec(...)` | Draw from inverse-gamma power spectrum posterior |
| `covariance_from_pspec(...)` | Convert delay PS to freq–freq covariance |
| `goodness_of_fit_statistics(...)` | χ² and log-posterior |
| `data_dly_fr(...)` | 2D FFT: freq–time → delay–fringe-rate |

### `sys_solver.py` — systematics

| Function | Description |
|---|---|
| `fourier_mode_2d(freqs_Hz, times_sec, modes)` | Build 2D Fourier basis from (delay, fr) mode pairs |
| `sys_modes(freqs_Hz, times_sec, modes)` | Systematic mode operator (Nfreqs×Ntimes, Nmodes) |
| `gcr_systematics(...)` | GCR draw for multiplicative systematic amplitudes |
| `cholesky_inverse(A)` | Invert positive-definite matrix via Cholesky decomposition |

### `utils.py` — utilities

| Function | Description |
|---|---|
| `fourier_operator(n)` | Dense n×n DFT matrix (fftshift convention) |
| `naive_pspec(data, ...)` | Quick delay power spectrum via FFT |
| `write_numpy_files(fp, ...)` | Write all sampler output arrays to `fp/` |
| `append_gibbs_sample_h5(fp, ...)` | Stream Gibbs samples to HDF5 (avoids large in-memory arrays) |
| `get_git_version_info()` | Retrieve current git hash and branch |
| `form_pseudo_stokes_vis(uvd)` | UVData XX+YY → pseudo-Stokes I |
| `filter_freqs(freq_str, freqs)` | Parse frequency selection string to Quantity array |

### `dpss.py` and `lssa.py` — foreground fitting

| Function | Description |
|---|---|
| `dpss_fit_modes(d, w, freqs, cov, ...)` | Weighted DPSS fit to masked complex 1D data |
| `lssa_fit_modes(d, freqs, invcov, ...)` | Weighted LSSA sinusoid fit to masked data |

---

## Known issues

- **`tests/solver_comps.py`, `tests/filtered_and_masked_run.py`**: Use the path
  `'npy_data/fg_true.npy'` (missing `res/` prefix). Will fail unless run from a
  directory that contains a symlink or copy.
- **`tests/example.py`**: Uses the deprecated `gibbs_sample_with_fg` API and a
  hardcoded absolute path to a UVData file.
- **`tests/phil_test.py`**: Hardcoded absolute path to a `.uvh5` file.
