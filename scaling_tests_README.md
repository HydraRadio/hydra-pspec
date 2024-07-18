# Strong scaling tests

Clone this repo on DIRAC, then from within the `hydra-pspec` dir,
set up the python venv:

```
module load python/3.12.4
python3 -m venv .venv
source .venv/bin/activate
module load intel_comp/2020-update2
module load intel_mpi/2020-update2
pip install .
```

Each baseline is identical, so we use a script to duplicate and
organise the data.
The test data used for this is all in one directory e.g. ~/scaling-data
where `ls` would give:

```
or-cov.npy      fake-telescope.yaml  noise-cov.npy  obsparams-select.yaml  vis-eor-fgs.uvh5
fake-antpos.csv  fgmodes.npy          noise.npy      README                 vis-eor.uvh5
```

Then to duplicate and structure the data in a way that `hydra-pspec` is expecting, run:

```
python set_up_scaling_data.py --data_dir=${HOME}/scaling-data --num_baselines=4096 --dest_dir=scaling-data
```

The slurm jobscripts are created, and submitted like this,
using 256 ranks (cores) as an example:

```
bash create_jobscript.sh 256
sbatch jobscript_256ranks_strongscaling.sh
```

The results are saved in a subdirectory in `results/strong_scaling/`.

Then to create scaling plots
```
source .venv/bin/activate
python plot_speed_up.py --results_dir=results/strong_scaling
```
