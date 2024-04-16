# Example using MPI and 3 baselines
The data for 3 baselines are in the directory `test-data`
(not under version control).

## How to run
- Activate your python environment e.g. `source .venv/bin/activate`
- `mpirun -n 3 python run-hydra-pspec.py --config config-3bl-mpi.yaml`
