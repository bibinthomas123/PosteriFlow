#!/bin/bash
# LeanNPE training (mf_scratch recipe + widened distance support) on a slice
# of an H100, followed by full certification. Measured peak 5.7 GB VRAM ->
# 2 shards; data remixing is CPU-bound -> 16 CPUs / 12 loader workers.
#   sbatch --export=ALL,NAME=lean_npe_v3,DATA=data/dataset_v3,EPOCHS=40 scripts/slurm/train_lean_npe.sbatch
# EXTRA appends trainer flags (a later --dist_range overrides the default);
# VALIDATE_EXTRA appends validate_checkpoint flags (e.g. --skip_gw150914 for
# BNS/NSBH-only models: the six smoke-test events are all BBH).
# GWOSC is unreachable from compute nodes: prefetch the 6 certification
# events on the login node first (gwpy caches them in ~/.astropy/cache).
#SBATCH --job-name=lean_npe
#SBATCH --partition=gpu-stud
#SBATCH --gres=shard:2
#SBATCH --exclude=ant7,ant8
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/PosteriFlow}"
PY=${PY:-$HOME/miniforge3/envs/ahsd/bin/python}
OUT=model/${NAME:?set NAME}
DATA=${DATA:?set DATA}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "[$(date)] train $NAME on $DATA ($(hostname))"
$PY -u experiments/train_lean_npe.py --data "$DATA" --remix \
    --epochs "${EPOCHS:-40}" --batch 128 --lr 3e-4 --workers 12 --outdir "$OUT" \
    --dist_range 8 6000 \
    --noise_bank data/noise_bank --real_noise_prob 0.9 --det_dropout 0.2 \
    --psd_cond --encoder_type coherent --mass_param mc_q --mf_tokens \
    --geom_dropout 0.25 --delay_aug 0.5 --aux_mass 1.0 --ckpt_every 2 ${EXTRA:-} 2>&1 | tee "$OUT/train.log"
echo "[$(date)] certify"
$PY scripts/validate_checkpoint.py --model "$OUT/best_model.pth" --data "$DATA" \
    --device cuda --outdir "analysis/ci/$NAME" ${VALIDATE_EXTRA:-} 2>&1 | tee "$OUT/validate.log" || true
echo "[$(date)] done"
