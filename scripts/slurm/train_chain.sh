#!/bin/bash
# Train beyond the 24 h job limit: submit SEGMENTS chained jobs of
# scripts/slurm/train_lean_npe.sbatch; each resumes from the previous
# segment's model/$NAME/last_model.pth (trainer --resume: weights, optimizer,
# LR schedule, epoch, best-model bookkeeping) when it exists. A segment killed
# at the time limit loses only its unfinished epoch. Certification runs at
# the end of every segment (the last one certifies the final model).
#
#   NAME=lean_npe_bbh_v6 DATA=data/dataset_bbh_v6big EPOCHS=30 \
#   MODEL_FLAGS="--mc_bank ... --pivot" EXTRA="--dist_range 80 5000 ..." \
#   bash scripts/slurm/train_chain.sh 3 [extra sbatch args, e.g. --gres=gpu:h100:1]
set -euo pipefail
cd "$(dirname "$0")/../.."
SEGMENTS=${1:?number of segments}; shift
: "${NAME:?}" "${DATA:?}" "${EPOCHS:?}"
# values go through the environment (--export=ALL): sbatch --export=K=V splits
# on commas, which once silently truncated a run's bank options
export NAME DATA EPOCHS WORKERS="${WORKERS:-12}" MODEL_FLAGS="${MODEL_FLAGS:-none}" EXTRA_BASE="${EXTRA:-}"
# AFTER=<jobid>: the first segment waits for that job to succeed (e.g. the
# dataset audit); later segments follow their predecessor whatever its state
prev=""
for i in $(seq 1 "$SEGMENTS"); do
  dep=${prev:+--dependency=afterany:$prev}
  [ -z "$prev" ] && [ -n "${AFTER:-}" ] && dep="--dependency=afterok:$AFTER"
  prev=$(sbatch --parsable $dep -p gpu-stud -c 16 --mem=64G -t 23:55:00 -J "${NAME#lean_npe_}_s$i" \
    -o "logs/train-${NAME#lean_npe_}-s$i-%j.out" "$@" \
    --export=ALL \
    --wrap "cd $PWD && R=''; [ -f model/$NAME/last_model.pth ] && R='--resume model/$NAME/last_model.pth'; EXTRA=\"\$EXTRA_BASE \$R\" bash scripts/slurm/train_lean_npe.sbatch")
  echo "segment $i: job $prev"
done
