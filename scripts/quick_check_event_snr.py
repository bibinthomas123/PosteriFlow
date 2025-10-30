from ahsd.data import GWDatasetGenerator
from ahsd.data.config import EVENT_TYPE_DISTRIBUTION, SNR_DISTRIBUTION
import pprint

def main():
    gen = GWDatasetGenerator(output_dir='data/quick_test', output_format='pkl', config={'debug_snr_diagnostic': False})
    summary = gen.generate_dataset(n_samples=100, overlap_fraction=0.1, edge_case_fraction=0.1, save_batch_size=100, add_glitches=False, preprocess=False, create_splits=False)
    print('Summary:')
    pprint.pprint(summary)

if __name__ == '__main__':
    main()
