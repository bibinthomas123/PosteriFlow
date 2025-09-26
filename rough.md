




(ahsd) ubuntu@ip-172-31-40-99:~/PosteriFlow$ python experiments/real_data_test.py  --event GW150914     --phase3a outputs/phase3a_output_3/neural_pe_model_only.pth     --phase3b outputs/phase3b_production/phase3b_working_output.pth     --save_dir outputs/real_data_analysis    --whiten
🚀 AHSD REAL DATA ANALYSIS - SCALE-AWARE VERSION
============================================================
Event: GW150914
Analysis length: 4.0s at 4096 Hz
Output directory: outputs/real_data_analysis
============================================================
🔧 Bandpass filter: 20.0-400.0 Hz
📡 Fetching 30.0s around GW150914 (GPS=1126259462.4)
✅ LIGO data fetched successfully
🔧 Preparing 4.0s segments...
✅ Segment prepared: 16384 samples (4.00s)
✅ Segment prepared: 16384 samples (4.00s)
✅ Data prepared: shape=(2, 16384)
   H1 RMS: 2.60e+00
   L1 RMS: 1.30e+00
🔧 Using computation device: cpu
🔧 Loading Neural PE from: outputs/phase3a_output_3/neural_pe_model_only.pth
   Parameters: ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec', 'geocent_time', 'theta_jn', 'psi', 'phase']
✅ Neural PE loaded: 2,995,414 parameters
🔧 Loading Subtractor from: outputs/phase3b_production/phase3b_working_output.pth
✅ Subtractor loaded: 38,681,377 parameters
🧠 Running AHSD scale-aware inference on GW150914...
🔧 Processing segment: shape=(2, 16384), length=16384
🔧 Synthetic data: RMS=1.02e+00, scaling=3.47e-01
✅ Inference completed: output_shape=(2, 16384)
   Strength: 0.333373
   Data scale: synthetic
   Original H1/L1 std: 2.60e+00/1.30e+00
✅ AHSD SCALE-AWARE ANALYSIS COMPLETED!
   Processing time: 0.02 seconds
   Subtraction strength: 0.333373
   Data scale detected: synthetic
   H1 noise reduction: 77.4%
   L1 noise reduction: 62.7%
   SNR improvement H1: 12.9 dB
   SNR improvement L1: 8.6 dB

📊 PARAMETER ESTIMATION RESULTS:
   Primary mass (m₁): 28.3 M☉
   Secondary mass (m₂): 24.7 M☉
   Luminosity distance: 662 Mpc
   Inclination angle: 113°

🔬 COMPARISON WITH LIGO CATALOG (GW150914):
   Mass 1 - AHSD: 28.3 M☉, LIGO: 36 M☉
   Mass 2 - AHSD: 24.7 M☉, LIGO: 29 M☉
   Distance - AHSD: 662 Mpc, LIGO: 410 Mpc
   Parameter agreement: m₁±21%, m₂±15%, D±61%

💾 Saving scale-aware analysis results...
📊 Generating scale-aware analysis plots...
✅ Saved ASD plot: outputs/real_data_analysis/GW150914_fs4096_seg4s_H1_asd.png
✅ Saved ASD plot: outputs/real_data_analysis/GW150914_fs4096_seg4s_L1_asd.png
✅ Saved Q-transform: outputs/real_data_analysis/GW150914_fs4096_seg4s_H1_raw_qtransform.png
✅ Saved Q-transform: outputs/real_data_analysis/GW150914_fs4096_seg4s_H1_cleaned_qtransform.png
✅ Saved Q-transform: outputs/real_data_analysis/GW150914_fs4096_seg4s_L1_raw_qtransform.png
✅ Saved Q-transform: outputs/real_data_analysis/GW150914_fs4096_seg4s_L1_cleaned_qtransform.png

🎉 AHSD SCALE-AWARE REAL DATA ANALYSIS COMPLETE!
============================================================
📁 Results saved to: outputs/real_data_analysis
📊 Files generated:
   • GW150914_fs4096_seg4s_*.npy (raw data, cleaned data, contamination pattern)
   • GW150914_fs4096_seg4s_analysis_results.json (complete metadata)
   • GW150914_fs4096_seg4s_*_asd.png (ASD comparison plots)
   • GW150914_fs4096_seg4s_*_qtransform.png (Q-transform spectrograms)
============================================================
🏆 AHSD successfully analyzed real GW150914 data with scale-aware processing!
   Signal subtraction: 70.1% average noise reduction
   Processing speed: 173.0x real-time
   Data scale: synthetic (2.60e+00 RMS)
============================================================