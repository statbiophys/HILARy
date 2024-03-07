#! /bin/bash
#
#SBATCH --job-name=AF

#SBATCH --output=logs_AF/AF_%A_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2048
#
#SBATCH --array=0-40%40
#

declare -a COMMANDS
let count=0
while read -r line; do
    COMMANDS[$count]=$line
    ((count++));
done < commands_synthetic.txt

srun ${COMMANDS[$SLURM_ARRAY_TASK_ID]}
