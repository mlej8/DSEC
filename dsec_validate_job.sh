#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=8G                  # memory (per node)
#SBATCH --time=01:00:00           # A time limit of zero requests that no time limit be imposed. Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".
#SBATCH --job-name=dsec         
#SBATCH --output=logs/%x-validation-%j.out
#SBATCH --mail-user=er.li@mail.mcgill.ca
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# confirm gpu available
nvidia-smi

# validate if a parameter is provided 
if [ $# -ne 1 ] 
then 
    echo "Usage $0 <filename>"
else
    # run the command
    source dsec-env/bin/activate
    python $1
fi