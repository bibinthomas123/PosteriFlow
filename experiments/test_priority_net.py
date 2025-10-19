#!/usr/bin/env python3
"""
Comprehensive Test Suite for PriorityNet
Tests functionality, edge cases, performance, and robustness
"""

import sys
from pathlib import Path
import torch
import numpy as np
import logging
import json
import time
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from train_priority_net import (
    PriorityNetDataset,
    ChunkedGWDataLoader,
    evaluate_priority_net,
    PriorityNet
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# SYNTHETIC TEST CASE GENERATORS
# ============================================================================

class TestCaseGenerator:
    """Generate synthetic test cases for PriorityNet."""
    
    @staticmethod
    def create_detection(mass_1, mass_2, distance, snr, source_type='BBH'):
        """Create a detection dictionary."""
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
        return {
            'mass_1': float(mass_1),
            'mass_2': float(mass_2),
            'chirp_mass': float(chirp_mass),
            'luminosity_distance': float(distance),
            'network_snr': float(snr),
            'signal_type': source_type,
            'a_1': 0.0,
            'a_2': 0.0,
            'tilt_1': 0.0,
            'tilt_2': 0.0,
            'phi_12': 0.0,
            'phi_jl': 0.0,
            'theta_jn': 0.0,
            'phase': 0.0,
            'ra': 0.0,
            'dec': 0.0,
            'psi': 0.0,
            'geocent_time': 0.0
        }
    
    @staticmethod
    def test_perfect_ordering():
        """Test case: Perfect descending SNR order."""
        return {
            'name': 'Perfect Ordering (High→Low SNR)',
            'detections': [
                TestCaseGenerator.create_detection(35, 30, 100, 25.0),
                TestCaseGenerator.create_detection(25, 20, 200, 15.0),
                TestCaseGenerator.create_detection(15, 10, 400, 8.0)
            ],
            'expected_priorities': [25.0, 15.0, 8.0],
            'expected_ranking': [0, 1, 2],  # Indices in priority order
            'test_type': 'basic_ranking',
            'expected_correlation': 1.0
        }
    
    @staticmethod
    def test_reverse_ordering():
        """Test case: Reverse SNR order."""
        return {
            'name': 'Reverse Ordering (Low→High SNR)',
            'detections': [
                TestCaseGenerator.create_detection(15, 10, 400, 8.0),
                TestCaseGenerator.create_detection(25, 20, 200, 15.0),
                TestCaseGenerator.create_detection(35, 30, 100, 25.0)
            ],
            'expected_priorities': [8.0, 15.0, 25.0],
            'expected_ranking': [2, 1, 0],
            'test_type': 'basic_ranking',
            'expected_correlation': 1.0
        }
    
    @staticmethod
    def test_random_ordering():
        """Test case: Random SNR order."""
        return {
            'name': 'Random Ordering',
            'detections': [
                TestCaseGenerator.create_detection(25, 20, 200, 15.0),
                TestCaseGenerator.create_detection(35, 30, 100, 25.0),
                TestCaseGenerator.create_detection(15, 10, 400, 8.0),
                TestCaseGenerator.create_detection(20, 18, 300, 12.0)
            ],
            'expected_priorities': [15.0, 25.0, 8.0, 12.0],
            'expected_ranking': [1, 0, 3, 2],
            'test_type': 'basic_ranking',
            'expected_correlation': 1.0
        }
    
    @staticmethod
    def test_close_snrs():
        """Test case: Very close SNR values (subtle ranking)."""
        return {
            'name': 'Close SNRs (Subtle Ranking)',
            'detections': [
                TestCaseGenerator.create_detection(30, 25, 150, 15.0),
                TestCaseGenerator.create_detection(28, 23, 160, 14.8),
                TestCaseGenerator.create_detection(26, 21, 170, 14.5)
            ],
            'expected_priorities': [15.0, 14.8, 14.5],
            'expected_ranking': [0, 1, 2],
            'test_type': 'subtle_ranking',
            'expected_correlation': 1.0,
            'tolerance': 0.05  # Allow for small errors
        }
    
    @staticmethod
    def test_equal_priorities():
        """Test case: Equal SNRs (tie-breaking)."""
        return {
            'name': 'Equal Priorities (Tie-Breaking)',
            'detections': [
                TestCaseGenerator.create_detection(20, 15, 200, 12.0),
                TestCaseGenerator.create_detection(20, 15, 200, 12.0)
            ],
            'expected_priorities': [12.0, 12.0],
            'expected_ranking': None,  # Any order acceptable
            'test_type': 'tie_breaking',
            'expected_correlation': None  # Undefined for equal values
        }
    
    @staticmethod
    def test_bbh_vs_bns():
        """Test case: Different source types."""
        return {
            'name': 'BBH vs BNS (Mixed Sources)',
            'detections': [
                TestCaseGenerator.create_detection(35, 30, 100, 20.0, 'BBH'),
                TestCaseGenerator.create_detection(1.4, 1.3, 100, 15.0, 'BNS'),
                TestCaseGenerator.create_detection(10, 1.4, 100, 18.0, 'NSBH')
            ],
            'expected_priorities': [20.0, 15.0, 18.0],
            'expected_ranking': [0, 2, 1],
            'test_type': 'source_type',
            'expected_correlation': 1.0
        }
    
    @staticmethod
    def test_extreme_masses():
        """Test case: Extreme mass values."""
        return {
            'name': 'Extreme Masses',
            'detections': [
                TestCaseGenerator.create_detection(100, 90, 500, 18.0, 'BBH'),  # High mass
                TestCaseGenerator.create_detection(5, 3, 50, 10.0, 'BBH'),      # Low mass
                TestCaseGenerator.create_detection(1.2, 1.1, 20, 8.0, 'BNS')    # Very low mass
            ],
            'expected_priorities': [18.0, 10.0, 8.0],
            'expected_ranking': [0, 1, 2],
            'test_type': 'extreme_params',
            'expected_correlation': 0.8
        }
    
    @staticmethod
    def test_extreme_distances():
        """Test case: Extreme distance values."""
        return {
            'name': 'Extreme Distances',
            'detections': [
                TestCaseGenerator.create_detection(30, 25, 10, 30.0, 'BBH'),    # Very close
                TestCaseGenerator.create_detection(30, 25, 500, 15.0, 'BBH'),   # Moderate
                TestCaseGenerator.create_detection(30, 25, 2000, 5.0, 'BBH')    # Very far
            ],
            'expected_priorities': [30.0, 15.0, 5.0],
            'expected_ranking': [0, 1, 2],
            'test_type': 'extreme_params',
            'expected_correlation': 1.0
        }
    
    @staticmethod
    def test_single_detection():
        """Test case: Single detection (edge case)."""
        return {
            'name': 'Single Detection',
            'detections': [
                TestCaseGenerator.create_detection(30, 25, 150, 20.0)
            ],
            'expected_priorities': [20.0],
            'expected_ranking': [0],
            'test_type': 'edge_case',
            'expected_correlation': None  # Undefined for single value
        }
    
    @staticmethod
    def test_heavy_overlap():
        """Test case: Many overlapping signals."""
        return {
            'name': 'Heavy Overlap (5 signals)',
            'detections': [
                TestCaseGenerator.create_detection(40, 35, 80, 30.0),
                TestCaseGenerator.create_detection(35, 30, 100, 25.0),
                TestCaseGenerator.create_detection(30, 25, 120, 22.0),
                TestCaseGenerator.create_detection(25, 20, 200, 15.0),
                TestCaseGenerator.create_detection(20, 15, 300, 10.0)
            ],
            'expected_priorities': [30.0, 25.0, 22.0, 15.0, 10.0],
            'expected_ranking': [0, 1, 2, 3, 4],
            'test_type': 'stress_test',
            'expected_correlation': 1.0
        }
    
    @staticmethod
    def get_all_test_cases():
        """Get all synthetic test cases."""
        return [
            TestCaseGenerator.test_perfect_ordering(),
            TestCaseGenerator.test_reverse_ordering(),
            TestCaseGenerator.test_random_ordering(),
            TestCaseGenerator.test_close_snrs(),
            TestCaseGenerator.test_equal_priorities(),
            TestCaseGenerator.test_bbh_vs_bns(),
            TestCaseGenerator.test_extreme_masses(),
            TestCaseGenerator.test_extreme_distances(),
            TestCaseGenerator.test_single_detection(),
            TestCaseGenerator.test_heavy_overlap()
        ]


# ============================================================================
# TEST SUITE CLASSES
# ============================================================================

class TestSuite:
    """Comprehensive test suite for PriorityNet."""
    
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        self.results = {}
    
    def run_all_tests(self, dataset=None):
        """Run all test suites."""
        logger.info("\n" + "="*80)
        logger.info("RUNNING COMPREHENSIVE PRIORITYNET TEST SUITE")
        logger.info("="*80)
        
        # Test 1: Synthetic edge cases
        self.results['synthetic'] = self.test_synthetic_scenarios()
        
        # Test 2: Real dataset (if provided)
        if dataset is not None:
            self.results['real_dataset'] = self.test_real_dataset(dataset)
        
        # Test 3: Performance benchmarks
        if dataset is not None:
            self.results['performance'] = self.test_performance(dataset)
        
        # Test 4: Robustness tests
        self.results['robustness'] = self.test_robustness()
        
        # Test 5: Edge cases
        self.results['edge_cases'] = self.test_edge_cases()
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    def test_synthetic_scenarios(self):
        """Test 1: Synthetic edge case scenarios."""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Synthetic Edge Cases")
        logger.info("="*80)
        
        test_cases = TestCaseGenerator.get_all_test_cases()
        results = []
        
        for test_case in test_cases:
            result = self._run_single_test(test_case)
            results.append(result)
            
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            logger.info(f"{status}: {test_case['name']}")
            
            if not result['passed'] and result.get('reason'):
                logger.info(f"  Reason: {result['reason']}")
        
        pass_rate = sum(r['passed'] for r in results) / len(results) if results else 0.0
        logger.info(f"\nSynthetic Test Pass Rate: {pass_rate*100:.1f}% ({sum(r['passed'] for r in results)}/{len(results)})")
        
        return {
            'n_tests': len(results),
            'passed': sum(r['passed'] for r in results),
            'failed': sum(not r['passed'] for r in results),
            'pass_rate': pass_rate,
            'individual_results': results
        }
    
    def _run_single_test(self, test_case):
        """Run a single test case."""
        try:
            with torch.no_grad():
                pred_priorities, uncertainties = self.model(test_case['detections'])
            
            pred_array = pred_priorities.cpu().numpy()
            expected_priorities = np.array(test_case['expected_priorities'])
            
            # Handle single detection case
            if len(pred_array) == 1:
                return {
                    'name': test_case['name'],
                    'passed': True,
                    'type': test_case['test_type'],
                    'reason': 'Single detection - cannot rank'
                }
            
            # Handle equal priorities
            if test_case.get('expected_correlation') is None:
                return {
                    'name': test_case['name'],
                    'passed': True,
                    'type': test_case['test_type'],
                    'reason': 'Equal priorities - tie-breaking acceptable'
                }
            
            # Compute correlation
            rho, _ = spearmanr(expected_priorities, pred_array)
            
            # Check ranking order
            expected_ranking = test_case.get('expected_ranking')
            pred_ranking = np.argsort(pred_array)[::-1]  # Descending order
            
            tolerance = test_case.get('tolerance', 0.0)
            passed = False
            reason = None
            
            if expected_ranking is not None:
                # Check if ranking matches
                if np.array_equal(pred_ranking, expected_ranking):
                    passed = True
                else:
                    # Check correlation threshold
                    if rho >= (test_case['expected_correlation'] - tolerance):
                        passed = True
                    else:
                        passed = False
                        reason = f"Correlation {rho:.3f} < threshold {test_case['expected_correlation'] - tolerance:.3f}"
            else:
                passed = True
            
            return {
                'name': test_case['name'],
                'passed': passed,
                'type': test_case['test_type'],
                'correlation': float(rho) if np.isfinite(rho) else None,
                'predicted_ranking': pred_ranking.tolist(),
                'expected_ranking': expected_ranking,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error in test {test_case['name']}: {e}")
            return {
                'name': test_case['name'],
                'passed': False,
                'type': test_case['test_type'],
                'reason': f"Exception: {str(e)}"
            }
    
    def test_real_dataset(self, dataset):
        """Test 2: Real dataset evaluation."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Real Dataset Evaluation")
        logger.info("="*80)
        
        results = evaluate_priority_net(
            self.model,
            dataset,
            split_name="test",
            debug_plots=False,
            out_dir=str(self.output_dir)
        )
        
        return results
    
    def test_performance(self, dataset):
        """Test 3: Performance benchmarking."""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Performance Benchmarking")
        logger.info("="*80)
        
        # Sample scenarios
        n_samples = min(100, len(dataset))
        sample_indices = np.random.choice(len(dataset), n_samples, replace=False)
        
        single_times = []
        double_times = []
        multi_times = []
        
        for idx in sample_indices:
            scenario = dataset[idx]
            n_det = len(scenario['detections'])
            
            start = time.time()
            with torch.no_grad():
                _ = self.model(scenario['detections'])
            elapsed = (time.time() - start) * 1000  # ms
            
            if n_det == 1:
                single_times.append(elapsed)
            elif n_det == 2:
                double_times.append(elapsed)
            else:
                multi_times.append(elapsed)
        
        results = {}
        
        for name, times in [('single', single_times), ('double', double_times), ('multi', multi_times)]:
            if times:
                results[name] = {
                    'n_samples': len(times),
                    'mean_ms': float(np.mean(times)),
                    'std_ms': float(np.std(times)),
                    'median_ms': float(np.median(times)),
                    'min_ms': float(np.min(times)),
                    'max_ms': float(np.max(times))
                }
                logger.info(f"{name.capitalize()} signals ({len(times)} samples):")
                logger.info(f"  Mean: {results[name]['mean_ms']:.2f} ± {results[name]['std_ms']:.2f} ms")
                logger.info(f"  Median: {results[name]['median_ms']:.2f} ms")
                logger.info(f"  Range: [{results[name]['min_ms']:.2f}, {results[name]['max_ms']:.2f}] ms")
        
        return results
    
    def test_robustness(self):
        """Test 4: Robustness to input variations."""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Robustness Testing")
        logger.info("="*80)
        
        results = {}
        
        # Test 4a: Noisy inputs
        base_detection = TestCaseGenerator.create_detection(30, 25, 150, 20.0)
        correlations = []
        
        for noise_level in [0.01, 0.05, 0.1, 0.2]:
            test_dets = []
            true_pris = []
            
            for i in range(5):
                snr = 20.0 - i * 3.0
                det = TestCaseGenerator.create_detection(
                    30 + np.random.randn() * noise_level * 30,
                    25 + np.random.randn() * noise_level * 25,
                    150 + np.random.randn() * noise_level * 150,
                    snr
                )
                test_dets.append(det)
                true_pris.append(snr)
            
            with torch.no_grad():
                preds, _ = self.model(test_dets)
            
            rho, _ = spearmanr(true_pris, preds.cpu().numpy())
            correlations.append(float(rho))
        
        results['noise_robustness'] = {
            'noise_levels': [0.01, 0.05, 0.1, 0.2],
            'correlations': correlations
        }
        
        logger.info("Noise Robustness:")
        for level, corr in zip(results['noise_robustness']['noise_levels'], correlations):
            logger.info(f"  Noise={level:.2f}: ρ={corr:.3f}")
        
        return results
    
    def test_edge_cases(self):
        """Test 5: Edge cases and boundary conditions."""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Edge Cases")
        logger.info("="*80)
        
        results = {}
        
        # Test 5a: Empty input handling
        try:
            with torch.no_grad():
                _ = self.model([])
            results['empty_input'] = {'passed': False, 'reason': 'Should raise error'}
        except:
            results['empty_input'] = {'passed': True, 'reason': 'Correctly raises error'}
        
        logger.info(f"Empty input: {'✅ PASS' if results['empty_input']['passed'] else '❌ FAIL'}")
        
        # Test 5b: Very large number of detections
        try:
            large_dets = [TestCaseGenerator.create_detection(30, 25, 150, 20.0 - i) for i in range(10)]
            with torch.no_grad():
                preds, _ = self.model(large_dets)
            results['large_input'] = {'passed': True, 'n_detections': 10}
            logger.info("Large input (10 signals): ✅ PASS")
        except Exception as e:
            results['large_input'] = {'passed': False, 'reason': str(e)}
            logger.info(f"Large input: ❌ FAIL - {e}")
        
        return results
    
    def generate_report(self):
        """Generate comprehensive test report."""
        report_path = self.output_dir / 'test_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n✅ Test report saved to {report_path}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TEST SUITE SUMMARY")
        logger.info("="*80)
        
        if 'synthetic' in self.results:
            logger.info(f"Synthetic Tests: {self.results['synthetic']['pass_rate']*100:.1f}% pass rate")
        
        if 'real_dataset' in self.results:
            logger.info(f"Real Dataset: ρ={self.results['real_dataset'].get('avg_correlation', 0):.3f}")
        
        if 'performance' in self.results and 'multi' in self.results['performance']:
            logger.info(f"Performance: {self.results['performance']['multi']['mean_ms']:.2f} ms (multi-signal)")
        
        logger.info("="*80)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive PriorityNet Testing')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', default=None, help='Dataset directory (optional)')
    parser.add_argument('--output_dir', default='outputs/comprehensive_tests', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    saved_config = checkpoint.get('config', {})
    use_strain = saved_config.get('use_strain', any('strain_encoder' in k for k in checkpoint['model_state_dict'].keys()))
    
    model = PriorityNet(use_strain=use_strain)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    
    # Load dataset if provided
    dataset = None
    if args.data_dir:
        logger.info(f"Loading dataset from {args.data_dir}")
        test_loader = ChunkedGWDataLoader(args.data_dir, split='test')
        test_scenarios = test_loader.convert_to_priority_scenarios(create_overlaps=False)
        dataset = PriorityNetDataset(test_scenarios, "test")
        logger.info(f"Loaded {len(dataset)} scenarios")
    
    # Run test suite
    test_suite = TestSuite(model, output_dir)
    results = test_suite.run_all_tests(dataset)
    
    logger.info(f"\n✅ All tests complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()