#!/bin/bash
# LeanNPE vs dynesty theta_jn information test, one event per array task.
#   sbatch --array=0-5 scripts/slurm/inclination_dynesty.sbatch
# Runs on CPU nodes other than ant2 (kept for GPU training jobs' loaders).
#SBATCH --job-name=incl_dyn
#SBATCH --partition=gpu-stud
#SBATCH --exclude=ant2,ant8
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/incl_dyn-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/PosteriFlow}"
${PY:-$HOME/miniforge3/envs/ahsd/bin/python} -u scripts/inclination_vs_dynesty.py \
    --index "$SLURM_ARRAY_TASK_ID" --npool "$SLURM_CPUS_PER_TASK"
