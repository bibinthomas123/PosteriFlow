#!/bin/bash
# Trainer throughput benchmark (BBH recipe): batch size x bf16 x torch.compile.
# Each line of the output is one PROFILE line from train_lean_npe.py --profile_steps.
#   sbatch scripts/slurm/speed_benchmark.sbatch
#SBATCH --job-name=speed_bench
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/speed_bench-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/PosteriFlow}"
PY=${PY:-$HOME/miniforge3/envs/ahsd/bin/python}
DATA=${DATA:-data/dataset_bbh_v3}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COMMON="--data $DATA --remix --epochs 1 --lr 3e-4 --workers 12 --outdir /tmp/speed_bench_$SLURM_JOB_ID \
  --dist_range 80 5000 --noise_bank data/noise_bank --real_noise_prob 0.9 --det_dropout 0.2 \
  --psd_cond --encoder_type coherent --mass_param mc_q --mf_tokens --mf_bank mf_bank_bbh.npz --pol_features \
  --geom_dropout 0.25 --delay_aug 0.5 --aux_mass 1.0"
for cfg in "--batch 128" "--batch 128 --amp bf16" "--batch 128 --amp bf16 --compile" \
           "--batch 512" "--batch 512 --amp bf16 --compile"; do
    steps=200; [[ $cfg == *512* ]] && steps=80
    $PY -u experiments/train_lean_npe.py $COMMON $cfg --profile_steps $steps 2>&1 | grep -a "PROFILE\|Error\|error:" | tail -3
done
