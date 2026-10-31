#!/bin/bash
# Full dataset audit; exits non-zero on any hard failure, so jobs submitted
# with --dependency=afterok:<this job> only start on clean data.
#   sbatch --export=ALL,DATA=data/dataset_v3[,DIST="8 6000"] scripts/slurm/audit_dataset.sbatch
#SBATCH --job-name=audit
#SBATCH --partition=gpu-stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/audit-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/PosteriFlow}"
${PY:-$HOME/miniforge3/envs/ahsd/bin/python} -u scripts/audit_dataset.py "${DATA:?set DATA}" --dist-range ${DIST:-8 6000}
