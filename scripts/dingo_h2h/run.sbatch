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
# the script selects steps by SUBSTRING of the job label, so "sampling" alone
# would also run the importance-sampling steps: pass the full labels
L=$(grep -oE "${EV}_data0_[0-9-]+_sampling\"" "$S" | head -1 | tr -d '"')
P=${L%_sampling}
t0=$(date +%s); bash "$S" "$L"
t1=$(date +%s); bash "$S" "${P}_importance_sampling_part0"
t2=$(date +%s); bash "$S" "${P}_importance_sampling_merge" || true
t3=$(date +%s)
printf '{"event": "%s", "sampling_s": %d, "importance_s": %d, "merge_s": %d, "node": "%s"}\n' \
  "$EV" $((t1 - t0)) $((t2 - t1)) $((t3 - t2)) "$(hostname)" > "$EV/timing.json"
cat "$EV/timing.json"
