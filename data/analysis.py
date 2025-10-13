#!/usr/bin/env python3
"""
Enhanced GW Dataset Analysis for NeuralPE and Priority Network Training
Validates physics correctness, overlap quality, and parameter independence
Usage: python analyze_dataset_enhanced.py --data_dir data/data
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ks_2samp
import warnings
warnings.filterwarnings('ignore')


def load_dataset(data_dir):
    """Load all samples from train/validation/test splits."""
    all_samples = []
    data_path = Path(data_dir)

    for split in ['train', 'validation', 'test']:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"⚠️  Split directory not found: {split_dir}")
            continue

        chunk_files = sorted(split_dir.glob('chunk_*.pkl'))
        print(f"   Found {len(chunk_files)} chunk(s) in {split}/")

        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'rb') as f:
                    samples = pickle.load(f)
                    all_samples.extend(samples)
                    print(f"   ✅ Loaded {len(samples)} samples from {chunk_file.name}")
            except Exception as e:
                print(f"   ❌ Error loading {chunk_file.name}: {e}")

    return all_samples


def extract_parameters(samples):
    """Extract physical parameters from all samples including overlaps."""
    params_list = []

    for sample in samples:
        metadata = sample['metadata']

        # Skip noise-only
        if metadata.get('event_type') == 'noise':
            continue

        # Get signal parameters (handle both single and overlap)
        sig_params_list = metadata.get('signal_parameters', [{}])
        if not isinstance(sig_params_list, list):
            sig_params_list = [sig_params_list]

        is_overlap = metadata.get('is_overlap', False)
        num_signals = len(sig_params_list) if is_overlap else 1

        # Extract first signal parameters
        sig_params = sig_params_list[0] if sig_params_list else {}

        params = {
            'mass_1': sig_params.get('mass_1', np.nan),
            'mass_2': sig_params.get('mass_2', np.nan),
            'total_mass': sig_params.get('total_mass', np.nan),
            'chirp_mass': sig_params.get('chirp_mass', np.nan),
            'mass_ratio': sig_params.get('mass_ratio', np.nan),
            'luminosity_distance': sig_params.get('luminosity_distance', np.nan),
            'redshift': sig_params.get('redshift', np.nan),
            'comoving_distance': sig_params.get('comoving_distance', np.nan),
            'a1': sig_params.get('a1', np.nan),
            'a2': sig_params.get('a2', np.nan),
            'effective_spin': sig_params.get('effective_spin', np.nan),
            'tilt1': sig_params.get('tilt1', np.nan),
            'tilt2': sig_params.get('tilt2', np.nan),
            'theta_jn': sig_params.get('theta_jn', np.nan),
            'lambda_1': sig_params.get('lambda_1', np.nan),
            'lambda_2': sig_params.get('lambda_2', np.nan),
            'event_type': metadata.get('event_type', 'unknown'),
            'is_overlap': is_overlap,
            'num_signals': num_signals,
            'network_snr': metadata.get('network_snr', np.nan)
        }

        params_list.append(params)

    return pd.DataFrame(params_list)


def check_physics_correctness(df):
    """Validate physics relationships in the dataset."""
    print("\n" + "="*80)
    print("🔬 PHYSICS CORRECTNESS CHECKS")
    print("="*80)

    issues = []

    # 1. Check inclination distribution (cos(θ) should be uniform)
    print("\n1️⃣  Checking inclination isotropy...")
    theta_jn = df['theta_jn'].dropna()
    if len(theta_jn) > 10:
        cos_theta = np.cos(theta_jn)
        _, p_value = ks_2samp(cos_theta, np.random.uniform(-1, 1, len(cos_theta)))
        if p_value < 0.05:
            issues.append(f"   ❌ Inclination not isotropic (p={p_value:.3f})")
        else:
            print(f"   ✅ Inclination is isotropic (p={p_value:.3f})")

    # 2. Check distance-SNR independence (within event type)
    print("\n2️⃣  Checking distance-SNR independence...")
    for event_type in ['BBH', 'BNS', 'NSBH']:
        mask = (df['event_type'] == event_type) & (~df['is_overlap'])
        if mask.sum() < 10:
            continue

        d = df.loc[mask, 'luminosity_distance'].dropna()
        snr = df.loc[mask, 'network_snr'].dropna()
        if len(d) > 10 and len(snr) > 10:
            # Align indices
            common_idx = d.index.intersection(snr.index)
            if len(common_idx) > 10:
                corr, _ = pearsonr(d[common_idx], snr[common_idx])
                if abs(corr) > 0.3:
                    issues.append(f"   ❌ {event_type}: Distance-SNR correlated (r={corr:.3f})")
                else:
                    print(f"   ✅ {event_type}: Distance-SNR independent (r={corr:.3f})")

    # 3. Check mass-distance independence
    print("\n3️⃣  Checking mass-distance independence...")
    for event_type in ['BBH', 'BNS', 'NSBH']:
        mask = (df['event_type'] == event_type) & (~df['is_overlap'])
        if mask.sum() < 10:
            continue

        m_tot = df.loc[mask, 'total_mass'].dropna()
        d = df.loc[mask, 'luminosity_distance'].dropna()
        common_idx = m_tot.index.intersection(d.index)
        if len(common_idx) > 10:
            corr, _ = pearsonr(m_tot[common_idx], d[common_idx])
            if abs(corr) > 0.3:
                issues.append(f"   ❌ {event_type}: Mass-distance correlated (r={corr:.3f})")
            else:
                print(f"   ✅ {event_type}: Mass-distance independent (r={corr:.3f})")

    # 4. Check effective spin calculation
    print("\n4️⃣  Checking effective spin physics...")
    mask = df[['a1', 'a2', 'tilt1', 'tilt2', 'mass_1', 'mass_2', 'effective_spin']].notna().all(axis=1)
    if mask.sum() > 10:
        sample_rows = df[mask].head(20)
        max_error = 0
        for _, row in sample_rows.iterrows():
            expected = ((row['a1'] * np.cos(row['tilt1']) * row['mass_1'] + 
                        row['a2'] * np.cos(row['tilt2']) * row['mass_2']) / 
                       (row['mass_1'] + row['mass_2']))
            error = abs(expected - row['effective_spin'])
            max_error = max(max_error, error)

        if max_error > 0.01:
            issues.append(f"   ❌ Effective spin calculation error (max={max_error:.4f})")
        else:
            print(f"   ✅ Effective spin correctly computed (max error={max_error:.4f})")

    # 5. Check cosmology consistency
    print("\n5️⃣  Checking cosmology (z, d_L, d_C relationship)...")
    mask = df[['redshift', 'luminosity_distance', 'comoving_distance']].notna().all(axis=1)
    if mask.sum() > 10:
        z = df.loc[mask, 'redshift']
        d_L = df.loc[mask, 'luminosity_distance']
        d_C = df.loc[mask, 'comoving_distance']

        # Check d_C < d_L (always true)
        if not (d_C < d_L).all():
            issues.append("   ❌ Comoving distance > luminosity distance (unphysical)")
        else:
            print("   ✅ d_C < d_L for all samples")

        # Check approximate relation d_C ≈ d_L/(1+z)
        d_C_approx = d_L / (1 + z)
        rel_error = np.abs(d_C - d_C_approx) / d_C
        if rel_error.mean() > 0.2:
            issues.append(f"   ❌ Cosmology inconsistent (mean error={rel_error.mean():.2%})")
        else:
            print(f"   ✅ Cosmology consistent (mean error={rel_error.mean():.2%})")

    # Summary
    print("\n" + "-"*80)
    if issues:
        print("\n❌ PHYSICS ISSUES FOUND:")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ ALL PHYSICS CHECKS PASSED!")
    print("="*80)


def check_overlap_quality(df):
    """Validate overlap-specific properties."""
    overlap_df = df[df['is_overlap'] == True]

    if len(overlap_df) == 0:
        print("\n⚠️  No overlap samples found, skipping overlap analysis")
        return

    print("\n" + "="*80)
    print("🔄 OVERLAP DATASET QUALITY")
    print("="*80)

    print(f"\n📊 Overlap Statistics:")
    print(f"   Total overlaps: {len(overlap_df)}")
    print(f"   Signals distribution: {overlap_df['num_signals'].value_counts().to_dict()}")

    # Check SNR distribution in overlaps
    print(f"\n   SNR range: {overlap_df['network_snr'].min():.1f} - {overlap_df['network_snr'].max():.1f}")
    print(f"   SNR mean: {overlap_df['network_snr'].mean():.1f}")

    # Event type mix in overlaps
    print(f"\n   Event types in overlaps: {overlap_df['event_type'].value_counts().to_dict()}")

    print("="*80)


def analyze_parameter_independence(df):
    """Check parameter independence within event types."""
    print("\n" + "="*80)
    print("🔗 PARAMETER INDEPENDENCE ANALYSIS")
    print("="*80)

    # Expected physical correlations (OK to have)
    expected = [
        ('mass_1', 'total_mass'), ('mass_2', 'total_mass'),
        ('mass_1', 'chirp_mass'), ('mass_2', 'chirp_mass'),
        ('luminosity_distance', 'redshift'),
        ('luminosity_distance', 'comoving_distance'),
        ('redshift', 'comoving_distance'),
        ('lambda_1', 'lambda_2')
    ]

    for event_type in ['BBH', 'BNS', 'NSBH']:
        event_df = df[(df['event_type'] == event_type) & (~df['is_overlap'])]

        if len(event_df) < 10:
            print(f"\n⚠️  Skipping {event_type}: insufficient samples")
            continue

        print(f"\n{event_type} ({len(event_df)} samples):")
        print("-"*40)

        numeric_cols = event_df.select_dtypes(include=[np.number]).columns
        corr = event_df[numeric_cols].corr()

        problematic = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                param1, param2 = corr.columns[i], corr.columns[j]

                # Skip expected correlations
                if (param1, param2) in expected or (param2, param1) in expected:
                    continue

                corr_val = corr.iloc[i, j]
                if abs(corr_val) > 0.3:
                    problematic.append((param1, param2, corr_val))

        if problematic:
            print("   ❌ Unexpected correlations:")
            for p1, p2, val in problematic:
                print(f"      {p1} ↔ {p2}: r={val:+.3f}")
        else:
            print("   ✅ No unexpected correlations (parameters independent)")

    print("="*80)


def plot_physics_checks(df, output_dir):
    """Create physics validation plots."""
    output_path = Path(output_dir)

    # 1. Inclination distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    theta_jn = df['theta_jn'].dropna()
    axes[0].hist(theta_jn, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('θ_jn (radians)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Inclination Distribution')
    axes[0].grid(True, alpha=0.3)

    cos_theta = np.cos(theta_jn)
    axes[1].hist(cos_theta, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axhline(len(cos_theta)/30, color='r', linestyle='--', label='Uniform expectation')
    axes[1].set_xlabel('cos(θ_jn)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('cos(θ) Distribution (should be flat)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'physics_inclination_check.png', dpi=150)
    print(f"✅ Saved: {output_path / 'physics_inclination_check.png'}")
    plt.close()

    # 2. Distance-SNR scatter per event type
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, event_type in enumerate(['BBH', 'BNS', 'NSBH']):
        mask = (df['event_type'] == event_type) & (~df['is_overlap'])
        event_df = df[mask]

        if len(event_df) > 5:
            axes[idx].scatter(event_df['luminosity_distance'], event_df['network_snr'], alpha=0.5)
            axes[idx].set_xlabel('Luminosity Distance (Mpc)')
            axes[idx].set_ylabel('Network SNR')
            axes[idx].set_title(f'{event_type} Distance-SNR')
            axes[idx].grid(True, alpha=0.3)

            # Add correlation
            d = event_df['luminosity_distance'].dropna()
            snr = event_df['network_snr'].dropna()
            common = d.index.intersection(snr.index)
            if len(common) > 5:
                corr, _ = pearsonr(d[common], snr[common])
                axes[idx].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[idx].transAxes,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig(output_path / 'physics_distance_snr_independence.png', dpi=150)
    print(f"✅ Saved: {output_path / 'physics_distance_snr_independence.png'}")
    plt.close()


def generate_summary_report(df, output_dir):
    """Generate comprehensive summary report."""
    output_path = Path(output_dir)

    report = []
    report.append("="*80)
    report.append("📊 GRAVITATIONAL WAVE DATASET ANALYSIS REPORT")
    report.append("="*80)
    report.append("")

    # Dataset statistics
    report.append("1. DATASET STATISTICS")
    report.append("-"*40)
    report.append(f"Total samples: {len(df)}")
    report.append(f"Single events: {(~df['is_overlap']).sum()}")
    report.append(f"Overlap events: {df['is_overlap'].sum()}")
    report.append("")

    # Event type distribution
    report.append("Event type distribution:")
    for event_type, count in df['event_type'].value_counts().items():
        pct = 100 * count / len(df)
        report.append(f"   {event_type}: {count} ({pct:.1f}%)")
    report.append("")

    # SNR statistics
    report.append("2. SNR DISTRIBUTION")
    report.append("-"*40)
    snr_stats = df['network_snr'].describe()
    report.append(f"Mean: {snr_stats['mean']:.2f}")
    report.append(f"Std: {snr_stats['std']:.2f}")
    report.append(f"Range: [{snr_stats['min']:.2f}, {snr_stats['max']:.2f}]")
    report.append("")

    # Parameter ranges
    report.append("3. PARAMETER RANGES")
    report.append("-"*40)
    for param in ['mass_1', 'mass_2', 'chirp_mass', 'luminosity_distance', 'redshift']:
        if param in df.columns:
            data = df[param].dropna()
            if len(data) > 0:
                report.append(f"{param}: [{data.min():.2f}, {data.max():.2f}]")

    # Save report
    report_text = "\n".join(report)
    with open(output_path / 'analysis_summary.txt', 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n✅ Summary saved to: {output_path / 'analysis_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced GW dataset analysis')
    parser.add_argument('--data_dir', type=str, default='data/data',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='analysis',
                       help='Output directory for plots')
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("📂 Loading dataset...")
    samples = load_dataset(args.data_dir)
    print(f"\n✅ Loaded {len(samples)} samples")

    print("\n🔍 Extracting parameters...")
    df = extract_parameters(samples)
    print(f"✅ Extracted {len(df)} samples")

    # Run all checks
    check_physics_correctness(df)
    check_overlap_quality(df)
    analyze_parameter_independence(df)

    # Generate visualizations
    print("\n📈 Creating visualizations...")
    plot_physics_checks(df, args.output_dir)

    # Generate summary
    generate_summary_report(df, args.output_dir)

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()