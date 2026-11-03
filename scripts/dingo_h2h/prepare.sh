#!/bin/bash
# Build each event's dingo_pipe workflow and download its data. Run on the
# LOGIN node (compute nodes have no internet); downloading is light.
#   bash scripts/dingo_h2h/prepare.sh [event ...]      (default: events.txt)
set -u
cd "$(dirname "$0")/../.."
ROOT=$PWD; OUT=$ROOT/results/dingo_h2h; mkdir -p "$OUT/ini"
M=${DINGO_MODEL:-$HOME/dingo_models/aligned/quasicircular_aligned_spin/6000mpc_hm_HL}
export PATH=$HOME/miniforge3/envs/dingo/bin:$PATH
EVENTS=${*:-$(grep -v '^#' "$ROOT/scripts/dingo_h2h/events.txt")}
cd "$OUT"
for ev in $EVENTS; do
  [ -f "$ev/data/.ok" ] && { echo "$ev: ready"; continue; }
  rm -rf "$ev"
  cat > "ini/$ev.ini" <<INI
accounting = local.dev.headtohead.dingo
model = $M/model_main.pt
model-init = $M/model_init.pt
device = cuda
num-samples = 50000
batch-size = 50000
recover-log-prob = True
importance-sample = True
importance-sampling-settings = {phase_marginalization:{approximation_22_mode:True}}
request-cpus-importance-sampling = 30
n-parallel = 1
trigger-time = $ev
label = $ev
outdir = $ev
channel-dict = {H1:GWOSC, L1:GWOSC}
detectors = [H1, L1]
psd-length = ${PSD_LENGTH:-128}
INI
  if nice -n 10 dingo_pipe "ini/$ev.ini" > "ini/$ev.build.log" 2>&1 \
     && nice -n 10 bash "$ev/submit/bash_$ev.sh" generation > "ini/$ev.gen.log" 2>&1 \
     && ls "$ev"/data/*event_data.hdf5 > /dev/null 2>&1; then
    touch "$ev/data/.ok"; echo "$ev: ready"
  else
    echo "$ev: SKIPPED ($(grep -hE "Error|error" ini/$ev.*.log | tail -1 | cut -c1-120))"
  fi
done
