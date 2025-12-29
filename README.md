# HVSM: Human vs Machine Text Classification

![Human vs Machine Generated Text classification](Human%20vs%20Machine%20Generated%20Text%20classification.png)

## Overview
This repo contains multiple end-to-end pipelines for classifying human vs machine-generated text for the SI670 Kaggle competition. It includes:
- CPU-first baselines and tuned variants.
- GPU-first rewrites using Polars + cuDF + cuML (a/b/c variants).
- SLURM job scripts for Great Lakes.
- Logging/diagnostics and weight saving for reproducibility.

## Repository layout
- `notebooks/`: all training/inference notebooks
- `slurm_scripts/`: SLURM job scripts (`hvsm_job*.sh`)
- `data/`: input CSVs (`train.csv`, `val.csv`, `test.csv`) **not committed**
- `outputs/`: submissions (created at runtime)
- `weights/`: saved models/vectorizers (created at runtime)
- `logs/`: SLURM stdout/err and papermill logs
- `runs/`: executed notebook outputs from papermill

## Approaches

### CPU pipelines (baseline + tuned)
1) **Baseline TF-IDF + XGBoost + Logistic Regression**
   - Notebook: `notebooks/hvsm_prod.ipynb`
   - Feature engineering + TF-IDF (1-3 grams)
   - Trains XGBoost + LR and calibrates with Platt scaling
   - Produces `outputs/submission_hvsm_prod.csv`

2) **CV + tuning pipeline**
   - Notebook: `notebooks/hvsm_prod_1.ipynb`
   - Stratified CV, randomized hyperparameter search
   - Heavier diagnostics and threshold tuning
   - Produces `outputs/submission_hvsm_prod_1.csv`

3) **Binary rules + prevalence match**
   - Notebook: `notebooks/hvsm_prod_2.ipynb`
   - Adds binary/text-structure features and prevalence matching
   - RandomizedSearchCV for LR/XGB
   - Produces `outputs/submission_hvsm_prod_2.csv`

### GPU pipelines (Polars + cuDF + cuML)
These mirror the CPU pipelines but run GPU-first with cuML/Polars.

A) **GPU baseline mirror**
   - Notebook: `notebooks/hvsm_prod_a.ipynb`
   - Polars feature engineering, cuML LR + NB
   - Produces `outputs/submission_hvsm_prod_a.csv`

B) **GPU tuned mirror**
   - Notebook: `notebooks/hvsm_prod_b.ipynb`
   - Random-search tuning on validation split
   - Produces `outputs/submission_hvsm_prod_b.csv`

C) **GPU binary rules + prevalence match**
   - Notebook: `notebooks/hvsm_prod_c.ipynb`
   - Binary rule sweep + prevalence head
   - Produces `outputs/submission_hvsm_prod_c.csv`

## Data
Place the competition CSVs in `data/`:
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

`test.csv` must include an `id` column and **no** `label`.

## Running locally
You can run any notebook from the repo root. Paths are relative to the root:

- Example (papermill):
  ```bash
  papermill notebooks/hvsm_prod_2.ipynb runs/hvsm_executed_local.ipynb
  ```

## Running on Great Lakes (SLURM)
Job scripts live in `slurm_scripts/`. Each script installs dependencies, sets up a temporary venv, and runs the matching notebook with papermill. Example:

```bash
sbatch slurm_scripts/hvsm_job_2.sh
```

## Outputs and weights
- Submissions go to `outputs/` with unique names per notebook.
- Model artifacts and vectorizers are saved to `weights/` (override with `WEIGHTS_DIR`).
- Logs are written to `logs/` (including step/iteration timestamps).

## Notes on logging
Each notebook prints START/END timestamps for major sections. Long-running steps also emit per-iteration or per-fit logs for visibility during long SLURM runs.
