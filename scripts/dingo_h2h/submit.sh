#!/bin/bash
# Submit DINGO for every prepared event of a run folder that has no result yet:
# GPU sampling (GNPE, small network inference: an A40 suffices) followed by
# importance sampling on a CPU node (vl-parcio, 48 cores, < 6 h). Steps are
# selected by their FULL labels: the generated script matches by substring,
# so "sampling" alone would also run the importance-sampling steps.
#   bash scripts/dingo_h2h/submit.sh [run_folder]      (default dingo_h2h)
set -u
cd "$(dirname "$0")/../.."
RUN=${1:-dingo_h2h}
cd "results/$RUN"
B=$HOME/miniforge3/envs/dingo/bin
LOGS=$HOME/PosteriFlow/logs
n=0
for d in */data/.ok; do
  ev=${d%%/*}
  ls "$ev"/result/*importance_sampling*.hdf5 >/dev/null 2>&1 && continue
  S=$ev/submit/bash_$ev.sh
  L=$(grep -oE "${ev}_data0_[0-9-]+_sampling\"" "$S" | head -1 | tr -d '"')
  I=$(grep -oE "${ev}_data0_[0-9-]+_importance_sampling(_part0)?\"" "$S" | head -1 | tr -d '"')
  M=$(grep -oE "${ev}_data0_[0-9-]+_importance_sampling_merge\"" "$S" | head -1 | tr -d '"')
  [ -z "$L" ] || [ -z "$I" ] && { echo "$ev: step labels not found"; continue; }
  s=$(sbatch --parsable -p gpu-stud --gres=gpu:a40:1 -c 8 --mem=48G -t 03:00:00 -J dS_$ev \
      -o "$LOGS/dingoS-$RUN-$ev-%j.out" \
      --wrap "cd $PWD && export PATH=$B:\$PATH && t0=\$(date +%s) && bash $S $L && echo \$((\$(date +%s)-t0)) > $ev/timing_sampling.txt")
  sbatch --parsable --dependency=afterok:$s -p vl-parcio -c 48 --mem=96G -t 05:55:00 -J dI_$ev \
      -o "$LOGS/dingoI-$RUN-$ev-%j.out" \
      --wrap "cd $PWD && export PATH=$B:\$PATH && t0=\$(date +%s) && bash $S $I ${M:+&& bash $S $M}; echo \$((\$(date +%s)-t0)) > $ev/timing_is.txt" >/dev/null
  n=$((n + 1))
done
echo "submitted $n events from results/$RUN"
