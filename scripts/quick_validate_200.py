import sys, os
# Ensure local `src` is preferred over any installed `ahsd` package
sys.path.insert(0, os.path.abspath('src'))
from ahsd.data import GWDatasetGenerator
from ahsd.data.config import EVENT_TYPE_DISTRIBUTION, SNR_DISTRIBUTION
from collections import Counter


def aggregate_counts(output_dir='data/quick_validate'):
    import glob, pickle
    batch_files = glob.glob(output_dir + '/batches/*.pkl')
    sig_counts = Counter()
    snr_counts = Counter()
    total_samples = 0
    for bf in batch_files:
        with open(bf, 'rb') as f:
            data = pickle.load(f)
            samples = data.get('samples', data) if isinstance(data, dict) else data
            for s in samples:
                total_samples += 1
                params = s.get('parameters')
                if not params:
                    continue
                if isinstance(params, list):
                    for p in params:
                        t = p.get('type') or p.get('event_type') or 'unknown'
                        sig_counts[t] += 1
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
                else:
                    p = params
                    t = p.get('type') or p.get('event_type') or 'unknown'
                    sig_counts[t] += 1
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
    return total_samples, sig_counts, snr_counts


if __name__ == '__main__':
    gen = GWDatasetGenerator(output_dir='data/quick_validate', output_format='pkl', config={'debug_snr_diagnostic': False})
    gen.generate_dataset(n_samples=200, overlap_fraction=0.1, edge_case_fraction=0.1, save_batch_size=100, add_glitches=False, preprocess=False, create_splits=False)
    total, sig_counts, snr_counts = aggregate_counts('data/quick_validate')
    print('Total samples (batches):', total)
    print('Signal type counts:')
    print(dict(sig_counts))
    print('SNR regime counts:')
    print(dict(snr_counts))
