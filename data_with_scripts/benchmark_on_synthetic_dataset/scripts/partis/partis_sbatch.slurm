#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --exclude=zuzia,node02,node03 # uses only 1 node
#SBATCH --time=100:00:00
#SBATCH --job-name=partis_18_30
#SBATCH --error=/home/dupic/job.%J.out
#SBATCH --output=/home/dupic/job.%J.out


cd /home/dupic/
source /home/dupic/mambaforge/etc/profile.d/conda.sh
conda deactivate
conda activate partis
export CPLUS_INCLUDE_PATH=${HOME}/.local/include/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH${HOME}/.local/lib:

for fasta_file in HILARy/data/fasta_benchmark/families_cdr3l_{18,21,24,27,30}_1e5_*.csv.fasta;
do
    base_name=$(basename -- "$fasta_file" .fasta);
    new_file="data/partis_benchmark/${base_name}.yaml";
    partis/bin/partis partition --infname "$fasta_file" --outfname "$new_file";
done


