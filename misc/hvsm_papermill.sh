#!/bin/bash
#SBATCH --job-name=notebook-job-hvsm
#SBATCH --account=stats_dept1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --output=papermill_output_hvsm.txt

module load python3.11-anaconda/2024.02 # load your env as needed
papermill hvsm_colab.ipynb hvsm_colab_output.ipynb
