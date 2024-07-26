# hydra-pspec
Gibbs sampler-based power spectrum estimation code with foreground filtering and in-painting capabilities

## Running hydra-psec
This code is designed to be run using MPI

```
mpirun -n <number_of_ranks> run-hydra-pspec.py --config <config_file.yaml>
```

A python virtual environment should first be created using
```
python -m venv .venv
source activate .venv/bin/activate
pip install .
```

