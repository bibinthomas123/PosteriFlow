#!/bin/bash
# Parallel dataset generation on a CPU node: one shard per core, merged into a
# single dataset (see src/ahsd/data/scripts/generate_parallel.py).
#
#   sbatch --export=ALL,N=200000,OUT=data/dataset_v3 scripts/slurm/generate_parallel.sbatch
#
# Defaults: configs/data_config_fixed.yaml, seed 2026, one shard per allocated CPU.
#SBATCH --job-name=gen_parallel
#SBATCH --partition=all
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gen_parallel-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/PosteriFlow}"
PY=${PY:-$HOME/miniforge3/envs/ahsd/bin/python}
CONFIG=${CONFIG:-configs/data_config_fixed.yaml}
N=${N:?set N, e.g. --export=ALL,N=200000,OUT=data/dataset_v3}
OUT=${OUT:?set OUT}
SEED=${SEED:-2026}
SHARDS=${SHARDS:-$SLURM_CPUS_PER_TASK}
echo "[$(date)] $N samples -> $OUT with $SHARDS shards on $(hostname)"
$PY -u src/ahsd/data/scripts/generate_parallel.py --config "$CONFIG" \
    --n-samples "$N" --output-dir "$OUT" --shards "$SHARDS" --seed "$SEED" --validate
echo "[$(date)] building memmap cache"
$PY - "$OUT" <<'PYEOF'
import sys
sys.path.insert(0, "."); sys.path.insert(0, "src")
from experiments.remix_data import build_memmap_cache
d = sys.argv[1]
for split in ("train", "validation"):
    info = build_memmap_cache(d, split, f"{d}/memmap/{split}")
    print(split, "noise", info["n_noise"], "signals", info["n_signals"], "events", len(info["events"]))
PYEOF
echo "[$(date)] done"
