# Script to create slurm jobscripts for run-hydra-pspec.py with different numbers of ranks
# Usage:
# bash create_jobscript.sh number_of_ranks

n_ranks=$1
job_name="strongscaling_${n_ranks}ranks"
sed "s/n_ranks_placeholder/${n_ranks}/g; s/job_name_placeholder/${job_name}/g" jobscript.sh.template > jobscript_${job_name}.sh