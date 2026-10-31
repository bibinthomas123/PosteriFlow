#!/usr/bin/env python3
"""Run the (regenerate -> audit ->) retrain -> certify -> summarize pipeline
unattended on the cluster.

Run on the LOGIN node (it needs internet for GWOSC; compute nodes have none):

    nohup python scripts/run_pipeline.py [--regenerate] [--wait-pid PID] \
        [--v1-o2-add 80] [--after-jobs 2008094] > logs/pipeline.log 2>&1 &

Steps, each logged to logs/pipeline.log:
  1. optionally wait for a running process (e.g. a noise-bank download) to end
  2. optionally top up the noise bank with O2-era Virgo segments
  3. prefetch the certification GWTC events into the shared gwpy cache
  4. remove leftover run folders of the models about to be retrained
  5. with --regenerate: delete each model's dataset and submit a generation
     job (scripts/slurm/generate_parallel.sbatch) and an audit job
     (afterok: generation; the audit exits non-zero on any hard failure)
  6. submit one training job per model (scripts/slurm/train_lean_npe.sbatch,
     afterok: its audit; each certifies its checkpoint, incl. the real events
     of its classes)
  7. submit a summary job (afterany: all trainings + --after-jobs) that writes
     analysis/RUN_SUMMARY.md via scripts/summarize_runs.py
Job ids are written to logs/pipeline_jobs.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PY = sys.executable

# Matched-filter template tokens + the cross-detector polarization features
# built on them. Only for models with BBH in the data: a template bank coarse
# enough for the encoder matches BNS/NSBH inspirals poorly (fitting factor
# ~0.16), so those models train without MF tokens.
MF_BBH = "--mf_tokens --mf_bank mf_bank_bbh.npz --pol_features"

# name, dataset, generation config, epochs, distance support (Mpc),
# model flags, generation node, generation seed
MODELS = [
    ("lean_npe_v3_e80", "data/dataset_v3", "configs/data_config_fixed.yaml", 80, (8, 6000), MF_BBH, "ant2", 2026),
    ("lean_npe_bbh_v3", "data/dataset_bbh_v3", "configs/data_config_bbh_v3.yaml", 40, (80, 5000), MF_BBH, "ant2", 2027),
    ("lean_npe_nsbh_v3", "data/dataset_nsbh_v3", "configs/data_config_nsbh_v3.yaml", 40, (15, 1600), "", "ant7", 2028),
    ("lean_npe_bns_v3", "data/dataset_bns_v3", "configs/data_config_bns_v3.yaml", 40, (7, 550), "", "ant7", 2029),
]
N_SAMPLES = 200000
# two generation jobs per node (ant1 is often ReqNodeNotAvail; ant2 has 64 CPUs)
GEN_CPUS = {"ant2": 32, "ant7": 60}
CERT_EVENTS = ["GW150914", "GW151226", "GW170104", "GW170608", "GW170729", "GW170814",
               "GW170817", "GW190425", 1262276684.0, "GW200115_042309"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sbatch(*args):
    out = subprocess.run(["sbatch", "--parsable", *args], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"sbatch {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip().split(";")[0]


ap = argparse.ArgumentParser()
ap.add_argument("--wait-pid", type=int, default=None, help="wait for this process to exit first")
ap.add_argument("--v1-o2-add", type=int, default=0, help="O2-era Virgo noise segments to add first")
ap.add_argument("--after-jobs", nargs="*", default=[], help="extra job ids the summary waits for")
ap.add_argument("--models", nargs="*", default=None, help="subset of model names (default: all)")
ap.add_argument("--regenerate", action="store_true",
                help="delete, regenerate and audit each model's dataset before training")
a = ap.parse_args()
models = [m for m in MODELS if a.models is None or m[0] in a.models]
Path("logs").mkdir(exist_ok=True)

if a.wait_pid:
    log(f"waiting for pid {a.wait_pid}")
    while Path(f"/proc/{a.wait_pid}").exists():
        time.sleep(30)
    log("process finished")

if a.v1_o2_add:
    log(f"adding {a.v1_o2_add} O2-era V1 noise segments")
    subprocess.run([PY, "scripts/download_gwosc_noise_bank.py", "--dets", "V1", "--runs", "O2",
                    "--add", str(a.v1_o2_add), "--outdir", "data/noise_bank"], check=False)

n_v1_o2 = sum(1 for f in Path("data/noise_bank").glob("V1_*_strain.npy") if int(f.name.split("_")[1]) < 1187733618)
log(f"noise bank: {n_v1_o2} O2-era V1 segments")

log("prefetching certification events (gwpy cache, shared with compute nodes)")
sys.path.insert(0, str(ROOT / "src"))
from ahsd.inference.preprocessing import fetch_gwosc  # noqa: E402
for ev in CERT_EVENTS:
    try:
        fetch_gwosc(ev)
    except Exception as e:  # certification reports the failure per event
        log(f"  prefetch {ev} failed: {str(e)[:80]}")

ids, data_jobs = {}, {}
for name, data, config, epochs, (lo, hi), flags, node, seed in models:
    short = name.replace("lean_npe_", "")
    for d in (Path("model") / name, Path("analysis/ci") / name):
        if d.exists():
            shutil.rmtree(d)
    dep = []
    if a.regenerate:
        if Path(data).exists():
            log(f"deleting {data}")
            shutil.rmtree(data)
        gen = sbatch("-p", "gpu-stud", "-w", node, f"--cpus-per-task={GEN_CPUS[node]}", "--mem=120G",
                     f"--job-name=gen_{short}",
                     f"--export=ALL,N={N_SAMPLES},OUT={data},CONFIG={config},SEED={seed}",
                     "scripts/slurm/generate_parallel.sbatch")
        audit = sbatch(f"--dependency=afterok:{gen}", f"--job-name=audit_{short}",
                       f"--export=ALL,DATA={data},DIST={lo} {hi}", "scripts/slurm/audit_dataset.sbatch")
        data_jobs[name] = {"generate": gen, "audit": audit}
        dep = [f"--dependency=afterok:{audit}"]
        log(f"submitted generation {gen} ({config} -> {data} on {node}) and audit {audit}")
    ids[name] = sbatch(*dep, "--gres=gpu:h100:1", f"--job-name={short}",
                       f"--export=ALL,NAME={name},DATA={data},EPOCHS={epochs},"
                       f"MODEL_FLAGS={flags or 'none'},EXTRA=--dist_range {lo} {hi}",
                       "scripts/slurm/train_lean_npe.sbatch")
    log(f"submitted {name} ({data}, {epochs} epochs, {lo}-{hi} Mpc, flags '{flags}') as {ids[name]}")

deps = ":".join(list(ids.values()) + [str(j) for j in a.after_jobs])
summary = sbatch(f"--dependency=afterany:{deps}", "-p", "gpu-stud", "-c", "1", "--mem=4G", "-t", "20",
                 "-J", "summary", "-o", "logs/summary-%j.out",
                 "--wrap", f"cd {ROOT} && {PY} scripts/summarize_runs.py --models {' '.join(ids)}")
log(f"submitted summary job {summary} (after {deps})")
json.dump({"trainings": ids, "data": data_jobs, "summary": summary, "after_jobs": a.after_jobs,
           "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}, open("logs/pipeline_jobs.json", "w"), indent=1)
log("pipeline submitted; results will appear in analysis/RUN_SUMMARY.md")
