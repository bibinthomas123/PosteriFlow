#!/usr/bin/env python3
"""Collect certification reports and the inclination test into one page.

    python scripts/summarize_runs.py --models lean_npe_v3_e80 lean_npe_bbh_v3 ... \
        [--out analysis/RUN_SUMMARY.md]

Reads analysis/ci/<model>/report.json (validate_checkpoint) and
analysis/inclination_info/runs/*/summary.json (edgeon_vs_dynesty). Missing
inputs are listed, not fatal, so a partly failed pipeline still reports.
"""
import argparse
import glob
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--out", default="analysis/RUN_SUMMARY.md")
a = ap.parse_args()

L = [f"# Run summary ({time.strftime('%Y-%m-%d %H:%M')})", ""]
missing = []
L += ["## Models", "", "| model | epoch | val NLL | gates passed | failed gates |", "|---|---|---|---|---|"]
details = []
for m in a.models:
    f = Path("analysis/ci") / m / "report.json"
    if not f.exists():
        missing.append(f"{m}: no certification report ({f})")
        L.append(f"| `{m}` | – | – | – | **no report** |")
        continue
    r = json.load(open(f))
    gates = r.get("gates", [])
    failed = [g["gate"].split("(")[0].strip() for g in gates if not g["pass"]]
    L.append(f"| `{m}` | {r['meta'].get('epoch')} | {r['meta'].get('val_nll', float('nan')):.3f} | "
             f"{sum(g['pass'] for g in gates)}/{len(gates)} | {'; '.join(failed) or 'none'} |")
    details.append((m, r))

for m, r in details:
    L += ["", f"### `{m}`", ""]
    bt = (r.get("gaussian") or {}).get("distance_by_type") or {}
    if bt:
        L.append("distance by class (true/pred, cov90): " + ", ".join(
            f"{k} x{v['true_over_pred']:.2f} / {v['cov90']:.2f}" for k, v in bt.items()))
    ev = r.get("gw150914") or {}
    if ev:
        L += ["", "| event | result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in ev.items()]

runs = sorted(glob.glob("analysis/inclination_info/runs/ev*/summary.json"))
L += ["", "## Inclination information: LeanNPE vs dynesty (network SNR 25, design noise)", ""]
if runs:
    L += ["| event | true θ_jn | dynesty \\|cos θ\\| 90% | NPE \\|cos θ\\| 90% | dynesty P(edge-on) | NPE P(edge-on) | d_L true / dynesty / NPE |",
          "|---|---|---|---|---|---|---|"]
    for f in runs:
        s = json.load(open(f)); d, n = s["dynesty"], s["npe"]
        L.append(f"| ev{s['index']:02d} | {s['truth']['theta_jn']:.2f} | [{d['abs_cos_90'][0]:.2f}, {d['abs_cos_90'][1]:.2f}] | "
                 f"[{n['abs_cos_90'][0]:.2f}, {n['abs_cos_90'][1]:.2f}] | {d['p_edge_on']:.2f} | {n['p_edge_on']:.2f} | "
                 f"{s['truth']['luminosity_distance']:.0f} / {d['dL_med']:.0f} / {n['dL_med']:.0f} |")
    L += ["", "If dynesty's interval is much narrower than the NPE's for the same event, the model is losing "
          "inclination information the data contains."]
else:
    missing.append("inclination test: no summaries in analysis/inclination_info/runs/")
if missing:
    L += ["", "## Missing", ""] + [f"- {x}" for x in missing]
Path(a.out).write_text("\n".join(L) + "\n")
print(f"wrote {a.out}" + (f" ({len(missing)} missing inputs)" if missing else ""))
