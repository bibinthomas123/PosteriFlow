#!/usr/bin/env python3
import re, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DG = Path("/home/bibinathomas/PosteriFlow/src/ahsd/data/dataset_generator.py")
WG = Path("/home/bibinathomas/PosteriFlow/src/ahsd/data/waveform_generator.py")

def backup(p: Path):
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = p.with_suffix(p.suffix + f".bak_{ts}")
    shutil.copy2(p, dst)
    print(f"✓ Backed up {p} -> {dst}")

def read(p: Path) -> str:
    s = p.read_text()
    return s

def write(p: Path, s: str):
    p.write_text(s)
    print(f"✓ Wrote {p}")

def patch_param_based_priority(s: str) -> str:
    # Replace computed-priority from _compute_snr_from_params(...) with param-based
    s = re.sub(
        r"priority\s*=\s*self\._compute_snr_from_params\([^)]*\)",
        'priority = float(params.get("target_snr", 15.0))',
        s,
    )
    # Replace direct np.clip(max_snr, 7.0, 80.0) assignments into priority with param-based
    s = re.sub(
        r"priority\s*=\s*float\(\s*np\.clip\(\s*max_snr\s*,\s*7\.0\s*,\s*80\.0\s*\)\s*\)",
        'priority = float(params.get("target_snr", 15.0))',
        s,
    )
    s = re.sub(
        r"priority\s*=\s*np\.clip\(\s*max_snr\s*,\s*7\.0\s*,\s*80\.0\s*\)",
        'priority = float(params.get("target_snr", 15.0))',
        s,
    )
    # If any return path returns a clipped strain SNR, prefer param-based when available
    s = re.sub(
        r"return\s+np\.clip\(\s*max_snr\s*,\s*7\.0\s*,\s*80\.0\s*\)",
        'return float(params.get("target_snr", 15.0)) if "target_snr" in params else float(max_snr)',
        s,
    )
    # Remove 80 caps used on generic 'snr' returns, set to a sane bound only if truly needed
    s = re.sub(
        r"return\s+float\(\s*np\.clip\(\s*snr\s*,\s*7\.0\s*,\s*80\.0\s*\)\s*\)",
        'return float(np.clip(snr, 5.0, 100.0))',
        s,
    )
    return s

def patch_simulator_params_scope(s: str) -> str:
    # In simulator path, avoid 'params' NameError; derive from sample['parameters'] safely
    pattern = re.compile(
        r"priority\s*=\s*float\(\s*params\.get\(\s*[\"']target_snr[\"']\s*,\s*15\.0\s*\)\s*\)\s*"
    )
    repl = (
        'sig_params = (sample.get("parameters") or {}) if isinstance(sample.get("parameters"), dict) '
        'else ((sample.get("parameters")[0]) if (isinstance(sample.get("parameters"), list) and sample.get("parameters")) else {})\n'
        '                priority = float(sig_params.get("target_snr", 15.0))'
    )
    s, n = pattern.subn(repl, s)
    if n:
        print(f"✓ Fixed simulator params scope in {n} place(s)")
    return s

def replace_compute_snr_function(s: str) -> str:
    # Replace entire def _compute_snr_from_params(...) block with safe version
    header = re.compile(r"^(\s*)def\s+_compute_snr_from_params\s*\(\s*self\s*,\s*params\s*:\s*Dict\s*,\s*detector_data\s*:\s*Optional\[Dict]\s*=\s*None\s*\)\s*->\s*float\s*:\s*$", re.M)
    m = header.search(s)
    if not m:
        print("! _compute_snr_from_params header not found; skipping replacement")
        return s
    indent = m.group(1)
    start = m.start()
    # find end of block (next top-level def/class with same indent or EOF)
    body_start = m.end()
    tail = s[body_start:]
    next_def = re.search(rf"^{indent}def\s+|^{indent}class\s+", tail, re.M)
    end = body_start + (next_def.start() if next_def else len(tail))
    safe_impl = f"""{indent}def _compute_snr_from_params(self, params: Dict, detector_data: Optional[Dict] = None) -> float:
{indent}    \"\"\"
{indent}    Compute SNR for diagnostics or when target_snr is missing.
{indent}    Never overwrite params['target_snr'] here; callers decide whether to store it.
{indent}    Preference order:
{indent}      1) If params['target_snr'] exists, return it verbatim.
{indent}      2) Else, estimate from strain if detector_data is provided (clamped to [5, 100]).
{indent}      3) Else, fall back to distance/mass scaling (clamped to [5, 100]).
{indent}    \"\"\"
{indent}    ts = params.get("target_snr")
{indent}    if ts is not None:
{indent}        return float(ts)
{indent}
{indent}    try:
{indent}        if detector_data:
{indent}            max_snr = 0.0
{indent}            for det_name, data in detector_data.items():
{indent}                strain = data.get("strain")
{indent}                if strain is None or len(strain) == 0:
{indent}                    continue
{indent}                snr = estimate_snr_from_strain(
{indent}                    strain,
{indent}                    psd=data.get("psd"),
{indent}                    sampling_rate=self.sample_rate,
{indent}                )
{indent}                if np.isfinite(snr):
{indent}                    max_snr = max(max_snr, float(snr))
{indent}            if max_snr > 0:
{indent}                return float(np.clip(max_snr, 5.0, 100.0))
{indent}    except Exception:
{indent}        pass
{indent}
{indent}    distance = float(params.get("luminosity_distance", 400.0))
{indent}    m1 = float(params.get("mass_1", 30.0))
{indent}    m2 = float(params.get("mass_2", 30.0))
{indent}    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
{indent}    reference_snr = 15.0
{indent}    reference_mass = 30.0
{indent}    reference_distance = 400.0
{indent}    eps = 1e-9
{indent}    snr_est = reference_snr * (max(chirp_mass, eps) / reference_mass) ** (5.0 / 6.0) * (reference_distance / max(distance, eps))
{indent}    snr_est *= np.random.uniform(0.9, 1.1)
{indent}    return float(np.clip(snr_est, 5.0, 100.0))
"""
    new_s = s[:start] + safe_impl + s[end:]
    print("✓ Replaced _compute_snr_from_params with safe implementation")
    return new_s

def ensure_crude_rescale_consistency(s: str) -> str:
    # Ensure reference_snr = 15.0 in crude rescale path; do not add caps here
    s = re.sub(
        r"(reference_snr\s*=\s*)(?:100\.0|80\.0|30\.0)",
        r"\g<1>15.0",
        s,
    )
    return s

def main():
    assert DG.exists(), f"Missing {DG}"
    backup(DG)
    s = read(DG)
    s = patch_param_based_priority(s)
    s = patch_simulator_params_scope(s)
    s = replace_compute_snr_function(s)
    write(DG, s)

    if WG.exists():
        backup(WG)
        w = read(WG)
        w = ensure_crude_rescale_consistency(w)
        write(WG, w)
    else:
        print(f"ℹ waveform_generator.py not found at {WG}, skipped")

    print("\nNext steps:")
    print("  1) pip install -e .")
    print("  2) rm -rf data/dataset/*")
    print("  3) ahsd-generate --config configs/data_config.yaml")
    print("  4) python3 analyze_snr.py data/dataset")

if __name__ == "__main__":
    main()
