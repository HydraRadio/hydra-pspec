# hydra-pspec
hydra-pspec is a Gibbs sampler-based power spectrum estimation code with foreground filtering and in-painting capabilities.  For more details on the underlying mathematics and demonstrations on simulated data please see [Kennedy+2023](https://ui.adsabs.harvard.edu/abs/2023ApJS..266...23K/abstract) and/or [Burba+2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535..793B/abstract).


## Python Dependencies

hydra-pspec is written in Python and has the following dependencies:

- astropy
- jsonargparse
- matplotlib
- mpi4py
- multiprocess
- numpy
- pyuvdata
- scipy
- setuptools
- setuptools_scm

If you wish to install all of these dependencies with `conda`/`mamba`, you can do so using the included `environment.yaml` file via
```
conda env create -f environment.yaml
```

hydra-pspec can then be installed via
```
pip install .
```


## Running hydra-pspec

hydra-pspec can be run using the provided driver script `run-hydra-pspec.py`.  This code is designed to be run using MPI via e.g.
```
mpirun -n <number_of_ranks> run-hydra-pspec.py --config <config_file.yaml>
```

There are several input parameters which are required to run a hydra-pspec analysis.  Using `jsonargparse`, these input parameters can be specified via a configuration yaml file or directly via the command line.  For a full list of available input parameters, run
```
python run-hydra-pspec.py --help
```

Please see `test_data/config.yaml` for an example of a configuration yaml file containing the minimum required parameters.  Please also see the `jsonargparse` [documentation](https://jsonargparse.readthedocs.io/en/stable/) for more details.


## Citation

Users of the code are requested to cite the following papers:

- [Kennedy+2023](https://ui.adsabs.harvard.edu/abs/2023ApJS..266...23K/abstract)
- [Burba+2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535..793B/abstract)


## How to contribute

hydra-pspec is an open source project which is being actively developed.  If you would like to make a contribution or suggest a feature, you are very welcome to do so in the form of an issue and/or pull request.

