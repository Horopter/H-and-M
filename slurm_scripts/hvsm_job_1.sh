#!/bin/bash
#SBATCH --job-name=hvsm_job_1
#SBATCH --account=si670f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH --time=15:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail
set -o errtrace
umask 077

module purge
module load python3.11-anaconda/2024.02

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"


mkdir -p logs .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"

export WORK_DIR="${SLURM_TMPDIR:-$PWD}"
export DATA_DIR="${PROJECT_ROOT}/data"
export PYTHONPATH="${PROJECT_ROOT}/tools:${PYTHONPATH:-}"
export HVSM_ENABLE_RMM=0


export VENV_DIR="${HVSM_VENV_DIR:-$PROJECT_ROOT/hvsm_venv_shared}"
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
  rm -f "$VENV_DIR"/bin/Activate.ps1 2>/dev/null || true
fi
source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV_DISABLE_PROMPT=1

mkdir -p "$WORK_DIR/runs" "$WORK_DIR/weights" "$WORK_DIR/.cache"
ln -snf "$WORK_DIR/runs" runs || true
ln -snf "$WORK_DIR/weights" weights || true

export HF_HOME="$WORK_DIR/.cache/hf"
mkdir -p "$HF_HOME"

export MALLOC_TRIM_THRESHOLD_=134217728
export MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:128,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_SERVICE_FORCE_INTEL=1

echo "Host:        $(hostname)"
echo "Date:        $(date -Is)"
echo "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
echo "CUDA_HOME:   ${CUDA_HOME:-none}"

LOCK_FILE="$VENV_DIR/.pip.lock"
if command -v flock >/dev/null; then
  exec 200>"$LOCK_FILE"
  flock -x 200
fi
if [ ! -f "$VENV_DIR/.deps_installed" ]; then
python - <<'PY'
import os, sys, subprocess, shutil
def run(*args): print("+", *args, flush=True); subprocess.check_call(list(args))
run(sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "pip>=24.0", "setuptools>=68.0", "wheel>=0.41.0", "packaging>=24.0")
BASE_REQS = [
    "papermill==2.6.0","ipykernel==6.29.5","matplotlib==3.8.4",
    "pandas>=2.2.2,<3","numpy>=1.26,<2","scikit-learn==1.5.2",
    "xgboost==2.1.1","tqdm==4.66.4","joblib==1.4.2","scipy==1.11.4",
    "seaborn==0.13.2","ftfy==6.2.3","emoji==2.14.0","textstat==0.7.4",
    "wordfreq==3.1.1","spacy==3.7.5","transformers==4.44.2",
    "datasets>=2.20,<4","pyarrow>=14.0.0",
]
run(sys.executable, "-m", "pip", "install", "-q", *BASE_REQS)
run(sys.executable, "-m", "pip", "install", "-q",
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl")
cuda_tag = None
idx = "https://download.pytorch.org/whl/cpu"
run(sys.executable, "-m", "pip", "install", "-q", "--index-url", idx,
    "torch", "torchvision", "torchaudio")
run(sys.executable, "-m", "pip", "check")
for m in ["papermill","ipykernel","transformers","datasets","sklearn",
          "xgboost","pandas","numpy","matplotlib","spacy",
          "en_core_web_sm","torch"]:
    __import__(m)
print("All critical imports succeeded.", flush=True)
PY
  touch "$VENV_DIR/.deps_installed"
fi
if command -v flock >/dev/null; then
  flock -u 200
  exec 200>&-
fi

KNAME="hvsm-${SLURM_JOB_ID:-$$}"
python -m ipykernel install --user --name "${KNAME}"   --display-name "HVSM (${KNAME})"
trap 'jupyter kernelspec remove -y "${KNAME}" 2>/dev/null || true' EXIT

NOTEBOOK_IN="notebooks/hvsm_prod_1.ipynb"
STAMP="$(date +%Y%m%d-%H%M%S)"
NOTEBOOK_OUT="$WORK_DIR/runs/hvsm_executed_${STAMP}.ipynb"

PM_ARGS=(
  -k "${KNAME}"
  -p DATA_DIR "${DATA_DIR}"
  -p TRAIN_CSV "${DATA_DIR}/train.csv"
  -p VAL_CSV "${DATA_DIR}/val.csv"
  -p TEST_CSV "${DATA_DIR}/test.csv"
)

for f in "${DATA_DIR}/train.csv" "${DATA_DIR}/val.csv" "${DATA_DIR}/test.csv"; do
  if [ ! -f "$f" ]; then
    if ls "${f}.part"* >/dev/null 2>&1; then
      echo "Missing $f; chunk parts found (will reassemble in notebook)."
    else
      echo "Missing required data file: $f (no chunk parts found)." >&2
      exit 1
    fi
  fi
done

echo "Running papermill -> ${NOTEBOOK_OUT}"
papermill --log-output "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" "${PM_ARGS[@]}"

echo "Done. Output: ${NOTEBOOK_OUT}"
