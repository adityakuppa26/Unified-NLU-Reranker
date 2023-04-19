#!/usr/bin/zsh

# Load module
tmux new -s throw "module load conda; conda activate ./alexa_env; sbatch --account=pi_adrozdov_umass_edu -c 24 --mem=10g --time=00:10:00 --partition=gypsum-titanx python 'throw.py'"


# An example of srun job
# srun -c 1 --account=pi_adrozdov_umass_edu --partition=gypsum-titanx python 'throw.py'