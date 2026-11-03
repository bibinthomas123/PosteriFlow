#!/bin/bash
# DINGO on one prepared event: GPU sampling (GNPE), then importance sampling
# on CPUs, each stage timed.  sbatch --export=ALL,EV=<event> scripts/dingo_h2h/run.sbatch
#SBATCH --job-name=dingo
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dingo-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/PosteriFlow}/results/dingo_h2h"
export PATH=$HOME/miniforge3/envs/dingo/bin:$PATH
S="$EV/submit/bash_$EV.sh"
t0=$(date +%s); bash "$S" sampling
t1=$(date +%s); bash "$S" importance_sampling_part0
t2=$(date +%s); bash "$S" importance_sampling_merge || true
t3=$(date +%s)
printf '{"event": "%s", "sampling_s": %d, "importance_s": %d, "merge_s": %d, "node": "%s"}\n' \
  "$EV" $((t1 - t0)) $((t2 - t1)) $((t3 - t2)) "$(hostname)" > "$EV/timing.json"
cat "$EV/timing.json"
