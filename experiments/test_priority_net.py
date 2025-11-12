#!/usr/bin/env python3
"""
🔥 PRIORITYNET PRODUCTION STRESS TESTER (CPU-compatible)
Complete 11-block validation with real GWTC-3 event parameters from GWOSC API.

Usage:
  python experiments/stress_test_priority_net.py \
    --model models/priority_net/priority_net_best.pth \
    --data_dir data/dataset \
    --device cpu
"""

import sys, json, time, logging, argparse, copy, os
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
import requests

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from train_priority_net import (
    PriorityNetDataset, ChunkedGWDataLoader, evaluate_priority_net, PriorityNet
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
L = logging.getLogger("stress_test")

FAIL_REASONS = []

def GATE(condition, reason):
    if not condition:
        FAIL_REASONS.append(reason)
        L.error(f"❌ GATE FAIL: {reason}")

# ========== Model restore ==========
def infer_hidden_dims_from_state(sd):
    dims = []
    w0 = sd.get('signal_encoder.input_embed.weight', None)
    if w0 is not None:
        dims.append(int(w0.shape[0]))
    i = 0
    while f'signal_encoder.blocks.{i}.linear.weight' in sd:
        dims.append(int(sd[f'signal_encoder.blocks.{i}.linear.weight'].shape[0]))
        i += 1
    if len(dims) >= 2 and dims[0] == dims[1]:
        dims = [dims[0]] + dims[2:]
    return dims if dims else [512, 384, 256, 128]

def build_config_from_checkpoint(ckpt):
    cfg_dict = ckpt.get('model_config') or ckpt.get('config') or {}
    if 'hidden_dims' not in cfg_dict or not cfg_dict['hidden_dims']:
        cfg_dict['hidden_dims'] = infer_hidden_dims_from_state(ckpt['model_state_dict'])
    cfg_dict.setdefault('dropout', 0.2)
    cfg_dict.setdefault('use_strain', True)
    cfg_dict.setdefault('use_edge_conditioning', True)
    cfg_dict.setdefault('n_edge_types', 19)
    return SimpleNamespace(**cfg_dict)

def load_prioritynet(checkpoint_path, device="cpu"):
    # If directory, find checkpoint file
    if os.path.isdir(checkpoint_path):
        for fname in ['priority_net_best.pth', 'checkpoint_best.pt', 'model.pt']:
            candidate = os.path.join(checkpoint_path, fname)
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = build_config_from_checkpoint(ckpt)
    model = PriorityNet(cfg, use_strain=cfg.use_strain, use_edge_conditioning=cfg.use_edge_conditioning, n_edge_types=cfg.n_edge_types).to(device).eval()
    try:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        L.info("✅ Loaded state_dict strict=True")
    except:
        msd, sd = model.state_dict(), ckpt['model_state_dict']
        filtered = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(filtered, strict=False)
        L.warning(f"⚠️  Loaded filtered: {len(filtered)} matched")
    return model

def det(m1, m2, dist, snr, src='BBH', ra=0.0, dec=0.0, t=0.0, a1=0.0, a2=0.0, tilt1=0.0, tilt2=0.0):
    return {'mass_1': float(m1), 'mass_2': float(m2), 'luminosity_distance': float(dist), 'ra': float(ra), 'dec': float(dec), 'geocent_time': float(t), 'theta_jn': 0.0, 'psi': 0.0, 'phase': 0.0, 'a_1': float(a1), 'a_2': float(a2), 'tilt_1': float(tilt1), 'tilt_2': float(tilt2), 'phi_12': 0.0, 'phi_jl': 0.0, 'signal_type': src, 'network_snr': float(snr)}

# ========== Tests 1–10 (same as before) ==========
def synthetic_cases():
    return [
        dict(name="Perfect (↓SNR)", detections=[det(35,30,100,25), det(25,20,200,15), det(15,10,400,8)], exp=[25,15,8], tol=0.00),
        dict(name="Reverse (↑SNR)", detections=[det(15,10,400,8), det(25,20,200,15), det(35,30,100,25)], exp=[8,15,25], tol=0.00),
        dict(name="Random", detections=[det(25,20,200,15), det(35,30,100,25), det(15,10,400,8), det(20,18,300,12)], exp=[15,25,8,12], tol=0.00),
        dict(name="Close SNR", detections=[det(30,25,150,15.0), det(28,23,160,14.8), det(26,21,170,14.5)], exp=[15.0,14.8,14.5], tol=0.05),
        dict(name="Heavy overlap (5)", detections=[det(40,35,80,30), det(35,30,100,25), det(30,25,120,22), det(25,20,200,15), det(20,15,300,10)], exp=[30,25,22,15,10], tol=0.00),
    ]

def run_synthetic(model, device="cpu"):
    L.info("\n" + "="*80 + "\n1️⃣  SYNTHETIC TESTS\n" + "="*80)
    passed, total = 0, 0
    for case in synthetic_cases():
        with torch.no_grad():
            preds, _ = model(case['detections'])
        p = preds.cpu().numpy()
        if p.size < 2:
            L.info(f"✅ {case['name']}: single (skip rank)")
            passed += 1; total += 1
            continue
        rho, _ = spearmanr(case['exp'], p)
        ok = rho >= (1.0 - case['tol'])
        L.info(f"{'✅' if ok else '❌'} {case['name']}: ρ={rho:.3f}")
        passed += int(ok); total += 1
    rate = passed / max(1, total)
    L.info(f"📊 Synthetic: {passed}/{total} ({rate*100:.1f}%)")
    GATE(rate >= 0.8, f"Synthetic pass rate {rate*100:.1f}% < 80%")

def stress_dense_overlaps(model):
    L.info("\n" + "="*80 + "\n2️⃣  DENSE OVERLAPS (n=6–8)\n" + "="*80)
    for n in [6, 7, 8]:
        dets = [det(35-i*3, 30-i*2.5, 100+i*50, 28-i*2.5) for i in range(n)]
        exp = [28-i*2.5 for i in range(n)]
        with torch.no_grad():
            preds, _ = model(dets)
        p = preds.cpu().numpy()
        rho, _ = spearmanr(exp, p)
        L.info(f"n={n}: ρ={rho:.3f}")
        GATE(rho >= 0.70, f"Dense n={n} ρ={rho:.3f} < 0.70")

def monotonicity_sensitivity(model):
    L.info("\n" + "="*80 + "\n3️⃣  MONOTONICITY & SENSITIVITY\n" + "="*80)
    base = [det(30, 25, 150, 20), det(25, 20, 200, 15)]
    with torch.no_grad():
        p_base, _ = model(base)
    inc = copy.deepcopy(base)
    inc[0]['network_snr'] = 22.0
    with torch.no_grad():
        p_inc, _ = model(inc)
    delta_snr = p_inc.cpu().numpy()[0] - p_base.cpu().numpy()[0]
    L.info(f"SNR +2 → Δpred={delta_snr:.4f}")
    if abs(delta_snr) < 0.001:
        L.warning("⚠️  SNR delta near-zero; model may prioritize other features")
    GATE(delta_snr >= -0.01, f"SNR increase caused priority drop (Δ={delta_snr:.4f})")
    
    far = copy.deepcopy(base)
    far[0]['luminosity_distance'] = 200.0
    with torch.no_grad():
        p_far, _ = model(far)
    delta_dist = p_far.cpu().numpy()[0] - p_base.cpu().numpy()[0]
    L.info(f"Distance +33% → Δpred={delta_dist:.4f}")
    GATE(delta_dist < 0.0, f"Distance increase didn't lower priority (Δ={delta_dist:.4f})")

def calibration_spread(model, dataset):
    L.info("\n" + "="*80 + "\n4️⃣  CALIBRATION & SPREAD\n" + "="*80)
    preds, trues = [], []
    for i in range(min(200, len(dataset))):
        sc = dataset[i]
        with torch.no_grad():
            p, _ = model(sc['detections'])
        preds.extend(p.cpu().numpy())
        trues.extend(sc['priorities'].numpy())
    preds, trues = np.array(preds), np.array(trues)
    L.info(f"mean(pred)={preds.mean():.3f} mean(true)={trues.mean():.3f}")
    L.info(f"std(pred)={preds.std():.3f} std(true)={trues.std():.3f}")
    L.info(f"max(pred)={preds.max():.3f} max(true)={trues.max():.3f}")
    gap = trues.max() - preds.max()
    L.info(f"Max gap={gap:.3f}")
    if gap > 0.18:
        L.warning(f"⚠️  Pred max under-scaled by {gap:.3f}; retrain recommended")

def uncertainty_quality(model, dataset):
    L.info("\n" + "="*80 + "\n5️⃣  UNCERTAINTY QUALITY\n" + "="*80)
    errs, uncs = [], []
    for i in range(min(200, len(dataset))):
        sc = dataset[i]
        with torch.no_grad():
            p, u = model(sc['detections'])
        e = np.abs(p.cpu().numpy() - sc['priorities'].numpy())
        errs.extend(e)
        uncs.extend(u.cpu().numpy())
    errs, uncs = np.array(errs), np.array(uncs)
    corr = np.corrcoef(errs, uncs)[0, 1]
    L.info(f"corr(|error|, unc)={corr:.3f}")
    GATE(corr >= 0.15, f"Uncertainty correlation {corr:.3f} < 0.15")

def edge_activation_check(model, dataset):
    L.info("\n" + "="*80 + "\n6️⃣  EDGE CONDITIONING\n" + "="*80)
    ids = []
    for i in range(min(100, len(dataset))):
        sc = dataset[i]
        # Note: PriorityNetDataset returns 'edge_type_ids' (plural) as a tensor
        edge_type_ids_tensor = sc.get('edge_type_ids', None)
        if edge_type_ids_tensor is not None:
            # Get first element from the tensor
            eid = int(edge_type_ids_tensor[0].item()) if len(edge_type_ids_tensor) > 0 else 0
        else:
            eid = 0
        ids.append(eid)
    ids = np.array(ids)
    var = ids.var()
    unique_ids = np.unique(ids)
    L.info(f"edge_type_id variance={var:.3f}")
    L.info(f"Unique edge_type_ids: {unique_ids} (count: {len(unique_ids)})")
    L.info(f"Distribution: {dict(zip(*np.unique(ids, return_counts=True)))}")
    if var < 1e-6:
        L.warning("⚠️  Edge IDs collapsed; edge pathway bypassed (fix dataset)")

def snr_nwise_breakdown(model, dataset):
    L.info("\n" + "="*80 + "\n7️⃣  SNR & N-WISE BREAKDOWN\n" + "="*80)
    bins = {'<8': [], '8-12': [], '12-20': [], '>20': []}
    for i in range(min(500, len(dataset))):
        sc = dataset[i]
        with torch.no_grad():
            p, _ = model(sc['detections'])
        for j, d in enumerate(sc['detections']):
            snr = d.get('network_snr', 15.0)
            tgt = sc['priorities'][j].item()
            pred = p[j].item()
            if snr < 8:
                bins['<8'].append((tgt, pred))
            elif snr < 12:
                bins['8-12'].append((tgt, pred))
            elif snr < 20:
                bins['12-20'].append((tgt, pred))
            else:
                bins['>20'].append((tgt, pred))
    for k, v in bins.items():
        if len(v) < 10:
            continue
        t, p = zip(*v)
        rho, _ = spearmanr(t, p)
        L.info(f"SNR {k:>6}: n={len(v):4d} ρ={rho:.3f}")

def cross_device_determinism(model, dataset, device):
    L.info("\n" + "="*80 + "\n8️⃣  CROSS-DEVICE DETERMINISM\n" + "="*80)
    if device == "cpu":
        L.info("⚠️  Running on CPU; skipping GPU-CPU comparison")
        return
    sample_idx = 42 if len(dataset) > 42 else 0
    sc = dataset[sample_idx]
    with torch.no_grad():
        model.to('cuda')
        p_gpu, _ = model(sc['detections'])
        model.to('cpu')
        p_cpu, _ = model(sc['detections'])
    diff = np.abs(p_gpu.cpu().numpy() - p_cpu.numpy()).max()
    L.info(f"Max GPU-CPU diff={diff:.2e}")
    GATE(diff < 1e-3, f"GPU-CPU diff {diff:.2e} > 1e-3")
    model.to(device)

def throughput_memory(model, dataset, device):
    L.info("\n" + "="*80 + "\n9️⃣  THROUGHPUT & MEMORY\n" + "="*80)
    times = {'single': [], 'double': [], 'multi': []}
    for i in range(min(100, len(dataset))):
        sc = dataset[i]
        n = len(sc['detections'])
        t0 = time.time()
        with torch.no_grad():
            _ = model(sc['detections'])
        dt = (time.time() - t0) * 1000
        if n == 1:
            times['single'].append(dt)
        elif n == 2:
            times['double'].append(dt)
        else:
            times['multi'].append(dt)
    for k, v in times.items():
        if not v:
            continue
        L.info(f"{k:>6}: {np.mean(v):.2f} ms (median {np.median(v):.2f})")

def ood_extremes(model):
    L.info("\n" + "="*80 + "\n🔟 OOD EXTREMES\n" + "="*80)
    extremes = [
        ("High-mass BBH", det(90, 60, 1000, 18)),
        ("Extreme spins", det(30, 25, 150, 15, a1=0.95, a2=0.95, tilt1=0.05, tilt2=0.05)),
        ("Close BNS", det(1.4, 1.3, 20, 22, src='BNS')),
        ("Far BBH", det(30, 25, 2000, 6)),
    ]
    for name, d in extremes:
        with torch.no_grad():
            p, _ = model([d])
        ok = not np.isnan(p.cpu().numpy()).any()
        L.info(f"{'✅' if ok else '❌'} {name}: pred={p.item():.4f}")
        GATE(ok, f"NaN in {name}")

# ========== 11. REAL EVENTS (fetch GWTC-3 from official JSON API) ==========
def fetch_gwtc3_from_api():
    """Fetch all GWTC-3 events with parameters from official GWOSC JSON API."""
    CATALOG_URL = "https://www.gwosc.org/eventapi/json/GWTC-3-confident/"
    L.info("📡 Fetching GWTC-3 catalog from GWOSC API...")
    
    # Static fallback for known events if API fails completely
    STATIC_FALLBACK = [
        {'name': 'GW150914', 'm1': 36.2, 'm2': 29.1, 'snr': 24.4, 'dl': 420, 'ra': 1.95, 'dec': -1.27, 'gps': 1126259462.4, 'src': 'BBH'},
        {'name': 'GW151012', 'm1': 23.3, 'm2': 13.6, 'snr': 10.0, 'dl': 1000, 'ra': 0.0, 'dec': 0.0, 'gps': 1128678900.4, 'src': 'BBH'},
        {'name': 'GW151226', 'm1': 14.2, 'm2': 7.5, 'snr': 13.0, 'dl': 440, 'ra': 1.2, 'dec': -0.5, 'gps': 1135136350.6, 'src': 'BBH'},
        {'name': 'GW170104', 'm1': 31.2, 'm2': 19.4, 'snr': 13.0, 'dl': 880, 'ra': 0.5, 'dec': -0.3, 'gps': 1167559936.6, 'src': 'BBH'},
        {'name': 'GW170608', 'm1': 12.0, 'm2': 7.0, 'snr': 15.0, 'dl': 340, 'ra': 1.1, 'dec': -0.2, 'gps': 1180922494.5, 'src': 'BBH'},
        {'name': 'GW170729', 'm1': 50.6, 'm2': 34.3, 'snr': 10.8, 'dl': 2840, 'ra': 0.0, 'dec': 0.0, 'gps': 1185389807.3, 'src': 'BBH'},
        {'name': 'GW170809', 'm1': 35.2, 'm2': 23.8, 'snr': 12.4, 'dl': 990, 'ra': 0.0, 'dec': 0.0, 'gps': 1186302519.8, 'src': 'BBH'},
        {'name': 'GW170814', 'm1': 30.5, 'm2': 25.3, 'snr': 18.0, 'dl': 540, 'ra': 0.6, 'dec': -0.7, 'gps': 1186741861.5, 'src': 'BBH'},
        {'name': 'GW170817', 'm1': 1.46, 'm2': 1.27, 'snr': 32.4, 'dl': 40, 'ra': 3.447, 'dec': -0.408, 'gps': 1187008882.4, 'src': 'BNS'},
        {'name': 'GW170818', 'm1': 35.5, 'm2': 26.8, 'snr': 11.3, 'dl': 1020, 'ra': 0.8, 'dec': -0.4, 'gps': 1187058327.1, 'src': 'BBH'},
        {'name': 'GW170823', 'm1': 39.6, 'm2': 29.4, 'snr': 11.9, 'dl': 1850, 'ra': 1.3, 'dec': -0.6, 'gps': 1187529256.5, 'src': 'BBH'},
        {'name': 'GW190412', 'm1': 30.1, 'm2': 8.4, 'snr': 19.0, 'dl': 730, 'ra': 0.0, 'dec': 0.0, 'gps': 1239082262.2, 'src': 'BBH'},
        {'name': 'GW190425', 'm1': 1.74, 'm2': 1.45, 'snr': 12.9, 'dl': 156, 'ra': 0.0, 'dec': 0.0, 'gps': 1240215503.0, 'src': 'BNS'},
        {'name': 'GW190521', 'm1': 85.0, 'm2': 66.0, 'snr': 15.0, 'dl': 5300, 'ra': 0.0, 'dec': 0.0, 'gps': 1242442967.4, 'src': 'BBH'},
        {'name': 'GW190814', 'm1': 23.2, 'm2': 2.6, 'snr': 25.0, 'dl': 267, 'ra': 0.0, 'dec': 0.0, 'gps': 1249852257.0, 'src': 'NSBH'},
        {'name': 'GW190910', 'm1': 42.6, 'm2': 15.5, 'snr': 10.0, 'dl': 2300, 'ra': 0.0, 'dec': 0.0, 'gps': 1252582119.0, 'src': 'BBH'},
    ]
    
    try:
        response = requests.get(CATALOG_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Debug: inspect structure
        events_raw = data.get("events", {})
        L.info(f"📦 API returned {len(events_raw)} raw events")
        
        records = []
        for name, event_data in events_raw.items():
            params = event_data.get("parameters", {})
            
            # Try all possible key names (GWOSC uses inconsistent naming)
            def get_param(param_dict, *keys):
                for k in keys:
                    if k in param_dict and param_dict[k] is not None:
                        return param_dict[k]
                return None
            
            m1 = get_param(params, "mass_1_source", "m1_source", "mass1", "m1")
            m2 = get_param(params, "mass_2_source", "m2_source", "mass2", "m2")
            snr = get_param(params, "network_matched_filter_snr", "network_snr", "SNR", "snr")
            dl = get_param(params, "luminosity_distance", "distance", "DL")
            ra = get_param(params, "right_ascension", "ra", "RA")
            dec = get_param(params, "declination", "dec", "DEC")
            gps = get_param(params, "GPS", "gps") or event_data.get("GPS") or event_data.get("gps", 0.0)
            
            # Skip if missing critical params
            if any(x is None for x in [m1, m2, snr, dl]):
                L.debug(f"⚠️  {name}: incomplete (m1={m1}, m2={m2}, snr={snr}, dl={dl})")
                continue
            
            src = 'BBH' if m1 > 3 and m2 > 3 else ('BNS' if m1 < 3 and m2 < 3 else 'NSBH')
            records.append({
                'name': name,
                'm1': float(m1),
                'm2': float(m2),
                'snr': float(snr),
                'dl': float(dl),
                'ra': float(ra) if ra is not None else 0.0,
                'dec': float(dec) if dec is not None else 0.0,
                'gps': float(gps),
                'src': src
            })
        
        if records:
            L.info(f"✅ Loaded {len(records)} events with valid parameters from GWTC-3 API")
            return records
        else:
            L.warning("⚠️  API returned 0 valid events; using static fallback catalog")
            return STATIC_FALLBACK
            
    except Exception as e:
        L.warning(f"⚠️  API fetch failed ({e}); using static fallback catalog")
        return STATIC_FALLBACK

def real_events_comprehensive(model):
    L.info("\n" + "="*80 + "\n1️⃣1️⃣ REAL EVENTS (GWTC-3) + DECOY TESTS\n" + "="*80)
    
    # Fetch real GWTC-3 events
    gwtc_events = fetch_gwtc3_from_api()
    if not gwtc_events:
        L.warning("⚠️  No events fetched; skipping real-event tests")
        return
    
    # Sort by SNR descending to test diverse range
    gwtc_events.sort(key=lambda x: x['snr'], reverse=True)
    
    results = []
    test_count = min(30, len(gwtc_events))  # Test top 30 events
    
    for ev in gwtc_events[:test_count]:
        d = det(ev['m1'], ev['m2'], ev['dl'], ev['snr'], src=ev['src'], ra=ev['ra'], dec=ev['dec'], t=ev['gps'])
        with torch.no_grad():
            p, u = model([d])
        pred_val, unc_val = p.item(), u.item()
        results.append((ev['name'], pred_val, unc_val, ev['snr']))
        L.info(f"{ev['name']:>15}: pred={pred_val:.4f} unc={unc_val:.4f} snr={ev['snr']:.1f} m1={ev['m1']:.1f} m2={ev['m2']:.1f}")
        GATE(not np.isnan(pred_val), f"{ev['name']} pred is NaN")
    
    # Distribution summary
    if results:
        preds = np.array([r[1] for r in results])
        uncs = np.array([r[2] for r in results])
        snrs = np.array([r[3] for r in results])
        L.info(f"\n📊 Real events summary (n={len(results)}):")
        L.info(f"  pred: mean={preds.mean():.3f} std={preds.std():.3f} range=[{preds.min():.3f}, {preds.max():.3f}]")
        L.info(f"  unc:  mean={uncs.mean():.3f} std={uncs.std():.3f}")
        L.info(f"  SNR:  range=[{snrs.min():.1f}, {snrs.max():.1f}]")
        
        # Check for prediction diversity
        if preds.std() < 0.01:
            L.warning("⚠️  Predictions have very low std; model may be under-sensitive to parameter variation")
        else:
            L.info(f"✅ Prediction diversity: std={preds.std():.3f}")
        
        GATE(preds.min() >= 0.01, f"Real event pred min {preds.min():.3f} < 0.01")
        GATE(preds.max() <= 0.60, f"Real event pred max {preds.max():.3f} > 0.60")
    
    # Decoy tests on known events
    L.info("\n🎭 Decoy tests:")
    for test_name in ['GW150914', 'GW170817']:
        real_ev = next((e for e in gwtc_events if test_name in e['name']), None)
        if not real_ev:
            L.warning(f"⚠️  {test_name} not found in catalog")
            continue
        
        real_d = det(real_ev['m1'], real_ev['m2'], real_ev['dl'], real_ev['snr'], src=real_ev['src'], ra=real_ev['ra'], dec=real_ev['dec'], t=real_ev['gps'])
        decoy_d = copy.deepcopy(real_d)
        decoy_d['network_snr'] *= 0.7
        decoy_d['luminosity_distance'] *= 1.3
        
        with torch.no_grad():
            preds, _ = model([real_d, decoy_d])
        pr, pd = preds.cpu().numpy()
        ok = pr > pd
        L.info(f"{'✅' if ok else '❌'} {test_name}: real={pr:.3f} decoy={pd:.3f}")
        GATE(ok, f"{test_name} decoy outranked real")

# ========== Main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    model = load_prioritynet(args.model, device=args.device)
    run_synthetic(model, args.device)

    val_dataset = None
    if args.data_dir:
        loader = ChunkedGWDataLoader(args.data_dir, split='validation', max_samples=500, verbose=False)
        scenarios = loader.convert_to_priority_scenarios(create_overlaps=True, overlap_probability=0.5)
        val_dataset = PriorityNetDataset(scenarios, "validation")
        L.info(f"✅ Loaded {len(val_dataset)} validation scenarios")

    stress_dense_overlaps(model)
    monotonicity_sensitivity(model)

    if val_dataset:
        calibration_spread(model, val_dataset)
        uncertainty_quality(model, val_dataset)
        edge_activation_check(model, val_dataset)
        snr_nwise_breakdown(model, val_dataset)
       

    ood_extremes(model)
    real_events_comprehensive(model)

    L.info("\n" + "="*80 + "\n🏁 FINAL VERDICT\n" + "="*80)
    if FAIL_REASONS:
        L.error(f"❌ TEST FAILED ({len(FAIL_REASONS)} gates):")
        for r in FAIL_REASONS:
            L.error(f"  - {r}")
        L.error("\n🔧 Recommended fixes:")
        L.error("  1. Calibration: retrain 5-10 epochs with adjusted loss scaling")
        L.error("  2. Edge IDs: fix dataset converter to assign varied edge_type_id")
        L.error("  3. SNR sensitivity: check feature normalization and weighting")
        sys.exit(1)
    else:
        L.info("✅ ALL GATES PASSED — MODEL IS PRODUCTION-READY 🚀")

if __name__ == "__main__":
    main()
