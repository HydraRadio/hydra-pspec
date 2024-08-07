# Script to create slurm jobscripts for run-hydra-pspec.py with different numbers of ranks
# Usage:
# bash create_jobscript.sh number_of_ranks

n_ranks=$1
job_name="${n_ranks}ranks_strongscaling"
results_dir=results/strong_scaling
sed "s/@n_ranks@/${n_ranks}/g; s/@job_name@/${job_name}/g; s#@out_dir@#"${results_dir}"#g" \
    jobscript.sh.template > jobscript_${job_name}.sh
mkdir -p slurm-logs
