import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_generate_dataset_config_mapping():
    """Load configs/data_config.yaml and assert the generator receives the expected mapped args."""
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / 'configs' / 'data_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Patch the GWDatasetGenerator so we don't run heavy initialization
    with patch('ahsd.data.scripts.generate_dataset.GWDatasetGenerator') as MockGen:
        mock_instance = MagicMock()
        # Return a simple summary so generate_dataset_from_config can continue
        mock_instance.generate_dataset.return_value = {'n_samples': config.get('n_samples', 1000), 'output_dir': config.get('output_dir')}
        MockGen.return_value = mock_instance

        # Import here so the module-level imports resolve under the patched name
        from ahsd.data.scripts.generate_dataset import generate_dataset_from_config

        summary = generate_dataset_from_config(config)

        # Assert the generator class was constructed once with expected kwargs
        MockGen.assert_called_once()
        gen_call_kwargs = MockGen.call_args.kwargs

        assert gen_call_kwargs['output_dir'] == config.get('output_dir', 'data/dataset')
        assert gen_call_kwargs['sample_rate'] == config.get('sample_rate', 4096)
        assert gen_call_kwargs['duration'] == config.get('duration', 4.0)
        assert gen_call_kwargs['detectors'] == config.get('detectors', ['H1', 'L1', 'V1'])
        assert gen_call_kwargs['output_format'] == config.get('output_format', 'pkl')
        assert gen_call_kwargs['config'] == config

        # Assert generate_dataset was invoked with expected mapped args
        mock_instance.generate_dataset.assert_called_once()
        gen_kwargs = mock_instance.generate_dataset.call_args.kwargs

        assert gen_kwargs['n_samples'] == config.get('n_samples', 1000)
        assert gen_kwargs['overlap_fraction'] == config.get('overlap_fraction')
        assert gen_kwargs['edge_case_fraction'] == config.get('edge_case_fraction')
        assert gen_kwargs['chunk_size'] == config.get('chunk_size')
        assert gen_kwargs['noise_augmentation_k'] == config.get('noise_augmentation_k')

        # Verify summary returned the mocked value
        assert summary['n_samples'] == config.get('n_samples', 1000)
