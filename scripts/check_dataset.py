#!/usr/bin/env python3
"""
Dataset QA for PriorityNet
- Validates presence and scale of critical fields (network_snr, edge_type_id)
- Checks overlap-size distribution and edge ID variance
- Audits target priority distribution and calibration range
- Verifies normalization ranges in _detections_to_tensor
- Computes quick rank correlations vs simple baselines (e.g., SNR-desc)
- Outputs actionable warnings and a JSON report

Usage:
  python scripts/check_dataset.py --data_dir data/dataset --split validation
"""

import sys, json, math, logging, argparse
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from experiments.train_priority_net import (
    PriorityNetDataset, ChunkedGWDataLoader
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
L = logging.getLogger("dataset_check")

def pctl(x, q):
    return float(np.percentile(x, q)) if len(x) else float('nan')

def safe_mean(x):
    return float(np.mean(x)) if len(x) else float('nan')

def rank_spearman(y_true, y_score):
    if len(y_true) < 2:
        return np.nan
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(y_true, y_score)
        return float(rho)
    except Exception:
        return np.nan

def check_edge_ids(scenarios, max_samples=2000):
    edge_ids = []
    for i, sc in enumerate(scenarios[:max_samples]):
        eid = sc.get('edge_type_id', None)
        if eid is not None:
            edge_ids.append(int(eid))
    edge_ids = np.array(edge_ids)
    var = float(edge_ids.var()) if edge_ids.size else float('nan')
    unique = np.unique(edge_ids).tolist() if edge_ids.size else []
    L.info(f"edge_type_id variance={var:.3e} unique={unique[:10]}{'...' if len(unique)>10 else ''}")
    issues = []
    if not edge_ids.size:
        issues.append("Missing edge_type_id in scenarios")
    elif var < 1e-6:
        issues.append("edge_type_id variance is ~0; edge conditioning bypassed. Assign varied IDs per overlap context.")
    return {'variance': var, 'unique': unique, 'issues': issues}

def check_snr_presence_scale(scenarios, max_samples=5000):
    snrs, missing = [], 0
    for i, sc in enumerate(scenarios[:max_samples]):
        for d in sc['detections']:
            if 'network_snr' in d and d['network_snr'] is not None:
                snrs.append(float(d['network_snr']))
            else:
                missing += 1
    snrs = np.array(snrs, float)
    L.info(f"SNR count={snrs.size}, missing={missing}, range=[{np.min(snrs) if snrs.size else np.nan:.2f}, {np.max(snrs) if snrs.size else np.nan:.2f}]")
    issues = []
    if missing > 0:
        issues.append(f"{missing} detections missing network_snr; ensure it is computed and stored.")
    if snrs.size:
        if np.max(snrs) < 12 or np.min(snrs) > 8:
            issues.append("SNR dynamic range is narrow; model may become insensitive. Target a spread from ~6 to ~35.")
    return {'count': int(snrs.size), 'missing': int(missing), 'min': float(np.min(snrs)) if snrs.size else None, 'max': float(np.max(snrs)) if snrs.size else None, 'issues': issues}

def check_overlap_distribution(scenarios, max_samples=100000):
    sizes = [len(sc['detections']) for sc in scenarios[:max_samples]]
    sizes = np.array(sizes)
    hist = {int(k): int((sizes==k).sum()) for k in sorted(np.unique(sizes).tolist())}

    L.info(f"Overlap size histogram: {hist}")
    issues = []
    frac_5p = float((sizes>=5).mean()) if sizes.size else 0.0
    if frac_5p < 0.2:
        issues.append(f"Low proportion of 5+ overlaps ({frac_5p*100:.1f}%). Increase to 25–35% for robustness.")
    return {'hist': hist, 'frac_5plus': frac_5p, 'issues': issues}

def check_priority_targets(ds: PriorityNetDataset, max_items=2000):
    y = []
    for i in range(min(max_items, len(ds))):
        sc = ds[i]
        y.extend(sc['priorities'].numpy().tolist())
    y = np.array(y, float)
    L.info(f"Priority targets: mean={safe_mean(y):.3f} std={float(np.std(y)) if y.size else float('nan'):.3f} range=[{np.min(y) if y.size else float('nan'):.3f}, {np.max(y) if y.size else float('nan'):.3f}]")
    issues = []
    if y.size:
        if np.max(y) < 0.8:
            issues.append("Target max < 0.8; check target scaling. Model may learn compressed outputs.")
        if np.std(y) < 0.1:
            issues.append("Target std < 0.1; label spread too low; expand calibration/priority spread.")
    return {'mean': safe_mean(y), 'std': float(np.std(y)) if y.size else None, 'min': float(np.min(y)) if y.size else None, 'max': float(np.max(y)) if y.size else None, 'issues': issues}

def check_feature_normalization(scenarios, to_tensor_fn=None, max_samples=256):
    # If you can import your _detections_to_tensor, pass it as to_tensor_fn for exact checks.
    # Here we fallback to heuristic checks over raw values.
    vals = {'mass_1':[], 'mass_2':[], 'luminosity_distance':[], 'network_snr':[]}
    for sc in scenarios[:max_samples]:
        for d in sc['detections']:
            for k in vals:
                if k in d and d[k] is not None:
                    vals[k].append(float(d[k]))
    stats = {k: {'min': float(np.min(v)) if v else None, 'max': float(np.max(v)) if v else None, 'mean': float(np.mean(v)) if v else None, 'std': float(np.std(v)) if v else None, 'count': len(v)} for k,v in vals.items()}
    for k,s in stats.items():
        L.info(f"{k:>20}: n={s['count']:4d} mean={s['mean'] if s['mean'] is not None else float('nan'):.3f} std={s['std'] if s['std'] is not None else float('nan'):.3f} range=[{s['min'] if s['min'] is not None else float('nan'):.3f}, {s['max'] if s['max'] is not None else float('nan'):.3f}]")
    issues = []
    if stats['network_snr']['count'] > 0 and stats['network_snr']['std'] is not None and stats['network_snr']['std'] < 1.0:
        issues.append("network_snr variance is low; avoid z-scoring that collapses small deltas. Consider min(snr,35)/35.0 scaling.")
    if stats['luminosity_distance']['count'] > 0 and stats['luminosity_distance']['max'] is not None and stats['luminosity_distance']['max'] < 200:
        issues.append("luminosity_distance max < 200 Mpc; expand to higher distances for dynamic range.")
    return {'stats': stats, 'issues': issues}

def check_baseline_rank(ds: PriorityNetDataset, max_items=256):
    from scipy.stats import spearmanr
    rhos = []
    for i in range(min(max_items, len(ds))):
        sc = ds[i]
        y = sc['priorities'].numpy()
        snrs = np.array([d.get('network_snr', 0.0) for d in sc['detections']], float)
        if len(y) >= 2 and np.std(snrs) > 1e-6:
            rho, _ = spearmanr(y, snrs)
            rhos.append(float(rho))
    if rhos:
        L.info(f"Baseline SNR-desc Spearman: mean={np.nanmean(rhos):.3f} median={np.nanmedian(rhos):.3f} n={len(rhos)}")
    else:
        L.info("Baseline SNR-desc Spearman: insufficient variance to compute")
    return {'snr_spearman_mean': float(np.nanmean(rhos)) if rhos else None, 'n': len(rhos)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split', default='validation', choices=['train','validation','test'])
    ap.add_argument('--max_samples', type=int, default=2000, help='Scenarios to inspect')
    ap.add_argument('--report', default='data/test/dataset_report.json')
    args = ap.parse_args()

    L.info(f"Loading {args.split} from {args.data_dir}")
    loader = ChunkedGWDataLoader(args.data_dir, split=args.split, max_samples=args.max_samples, verbose=False)
    scenarios = loader.convert_to_priority_scenarios(create_overlaps=False)
    ds = PriorityNetDataset(scenarios, args.split)
    L.info(f"Scenarios: {len(ds)}")

    report = {}
    report['edge_ids'] = check_edge_ids(scenarios, max_samples=args.max_samples)
    report['snr'] = check_snr_presence_scale(scenarios, max_samples=args.max_samples)
    report['overlaps'] = check_overlap_distribution(scenarios, max_samples=args.max_samples)
    report['targets'] = check_priority_targets(ds, max_items=args.max_samples)
    report['features'] = check_feature_normalization(scenarios, max_samples=min(512, args.max_samples))
    report['baseline'] = check_baseline_rank(ds, max_items=min(512, args.max_samples))

    # Print issues summary
    issues = []
    for k, v in report.items():
        if isinstance(v, dict) and 'issues' in v and v['issues']:
            for it in v['issues']:
                L.error(f"[{k}] {it}")
                issues.append(f"{k}: {it}")

    # Save JSON report
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    L.info(f"Report written to {args.report}")

    # Exit non-zero if critical issues found
    critical = [it for it in issues if any(key in it.lower() for key in ['missing edge_type_id','variance is ~0','missing network_snr'])]
    if critical:
        L.error(f"❌ Critical dataset issues found: {len(critical)}")
        sys.exit(1)
    else:
        L.info("✅ Dataset checks passed (no critical blockers)")

if __name__ == "__main__":
    main()
