#!/bin/bash
#SBATCH --job-name=notebook-job-hvsm
#SBATCH --account=stats_dept1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --output=papermill_output_hvsm.txt

# 1) Load the intended Python
module load python3.11-anaconda/2024.02

# 2) Make sure the job's Python is visible (good for debugging)
echo "Python: $(python -c 'import sys; print(sys.executable)')"
echo "Papermill: $(which papermill)"

# 3) Register a kernelspec tied to THIS Python (scoped by job id so it won't clash)
KNAME="hvsm-${SLURM_JOB_ID}"
python -m ipykernel install --user --name "${KNAME}" --display-name "Python 3.11 (${KNAME})"

# Optional: show kernels so you can confirm in the job log
jupyter kernelspec list

# 4) Run papermill using that kernel
papermill -k "${KNAME}" hvsm_colab.ipynb hvsm_colab_output.ipynb

# 5) (Optional) Clean up the kernelspec after the run
jupyter kernelspec remove -y "${KNAME}" || true
