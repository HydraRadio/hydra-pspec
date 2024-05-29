# Script to run run-hydra-pspec.py with different numbers of ranks
nranks=$1
results_dir="temp/3bl_${nranks}ranks"
mpirun -n ${nranks} python run-hydra-pspec.py --config config-3bl-mpi.yaml --dirname=${results_dir}
cp $0 ${results_dir}
