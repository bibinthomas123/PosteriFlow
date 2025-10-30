import sys, os
# Prefer local src over installed packages
sys.path.insert(0, os.path.abspath('src'))

from ahsd.data.parameter_sampler import ParameterSampler
from ahsd.data import GWDatasetGenerator
from collections import Counter
import glob, pickle, shutil

OUT_BASE = 'data/quick_calibration'
BASELINE_DIR = os.path.join(OUT_BASE, 'baseline')
CAL_DIR = os.path.join(OUT_BASE, 'calibrated')


def aggregate_counts(output_dir):
    batch_dir = os.path.join(output_dir, 'batches')
    files = sorted(glob.glob(os.path.join(batch_dir, '*.pkl')))
    sig_counts = Counter()
    snr_counts = Counter()
    total_signals = 0
    for bf in files:
        with open(bf, 'rb') as f:
            data = pickle.load(f)
            samples = data.get('samples', data) if isinstance(data, dict) else data
            for s in samples:
                params = s.get('parameters')
                if not params:
                    continue
                if isinstance(params, list):
                    for p in params:
                        if not isinstance(p, dict):
                            continue
                        t = p.get('type') or p.get('event_type') or 'unknown'
                        sig_counts[t] += 1
                        total_signals += 1
                        if 'target_snr' in p:
                            snr = float(p['target_snr'])
                            if snr < 10:
                                snr_counts['weak'] += 1
                            elif snr < 15:
                                snr_counts['low'] += 1
                            elif snr < 25:
                                snr_counts['medium'] += 1
                            elif snr < 40:
                                snr_counts['high'] += 1
                            else:
                                snr_counts['loud'] += 1
                elif isinstance(params, dict):
                    p = params
                    t = p.get('type') or p.get('event_type') or 'unknown'
                    sig_counts[t] += 1
                    total_signals += 1
                    if 'target_snr' in p:
                        snr = float(p['target_snr'])
                        if snr < 10:
                            snr_counts['weak'] += 1
                        elif snr < 15:
                            snr_counts['low'] += 1
                        elif snr < 25:
                            snr_counts['medium'] += 1
                        elif snr < 40:
                            snr_counts['high'] += 1
                        else:
                            snr_counts['loud'] += 1
    return total_signals, dict(sig_counts), dict(snr_counts)


def ensure_clean(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(os.path.join(dirpath, 'batches'), exist_ok=True)


if __name__ == '__main__':
    # Baseline generation (no calibration)
    ensure_clean(BASELINE_DIR)
    print('Generating baseline (500 samples) ...')
    gen_baseline = GWDatasetGenerator(output_dir=BASELINE_DIR, output_format='pkl', config={'debug_snr_diagnostic': False})
    gen_baseline.generate_dataset(n_samples=500, overlap_fraction=0.1, edge_case_fraction=0.1, save_batch_size=100, add_glitches=False, preprocess=False, create_splits=False)
    print('Aggregating baseline counts...')
    b_total, b_sig, b_snr = aggregate_counts(BASELINE_DIR)

    # Calibrate sampler
    print('Empirically calibrating sampler (5000 forward samples) ...')
    sampler = ParameterSampler()
    cond = sampler.empirical_calibrate(n_samples=5000, random_seed=42)
    print('Calibration result (per-event-type SNR regime fractions):')
    from pprint import pprint
    pprint(cond)

    # Calibrated generation
    ensure_clean(CAL_DIR)
    print('Generating calibrated dataset (1000 samples) using calibrated sampler ...')
    gen_cal = GWDatasetGenerator(output_dir=CAL_DIR, output_format='pkl', config={'debug_snr_diagnostic': False}, parameter_sampler=sampler)
    gen_cal.generate_dataset(n_samples=1000, overlap_fraction=0.1, edge_case_fraction=0.1, save_batch_size=100, add_glitches=False, preprocess=False, create_splits=False)

    print('Aggregating calibrated counts...')
    c_total, c_sig, c_snr = aggregate_counts(CAL_DIR)

    print('\n=== SUMMARY ===')
    print('Baseline signals:', b_total)
    print('Baseline signal-type counts:')
    pprint(b_sig)
    print('Baseline SNR regime counts:')
    pprint(b_snr)

    print('\nCalibrated signals:', c_total)
    print('Calibrated signal-type counts:')
    pprint(c_sig)
    print('Calibrated SNR regime counts:')
    pprint(c_snr)

    # Save calibration map for inspection
    with open(os.path.join(CAL_DIR, 'calibration_map.pkl'), 'wb') as f:
        pickle.dump(cond, f)
    print('\nCalibration map saved to', os.path.join(CAL_DIR, 'calibration_map.pkl'))
