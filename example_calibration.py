#!/usr/bin/env python3
"""
example_calibration.py — Run Step 0 instrument calibration.

Reads parameters from config.yaml, fits the per-pixel dark-current model
and light-response curve, recovers the camera response function (CRF), and
saves four calibration arrays to the configured output directory:

    Sd.npy    — per-pixel dark-current slope
    b.npy     — per-pixel dark-current intercept
    Smax.npy  — per-pixel saturation threshold (sigmoid asymptote)
    crf.npy   — camera response function

These files are required inputs for example_radiance.py.
Calibration only needs to be run once per detector/imager.

Usage:
    python example_calibration.py
    python example_calibration.py --config path/to/config.yaml
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from config import load_config
import Step0_calibration as step0


def main():
    parser = argparse.ArgumentParser(description="Run SWIR_HDR Step 0 calibration.")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: repo root config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cal = cfg["calibration"]

    print("=" * 60)
    print("  SWIR_HDR — Step 0 Calibration")
    print("=" * 60)
    print(f"  dc_dir         : {cal['dc_dir']}")
    print(f"  reflectance_dir: {cal['reflectance_dir']}")
    print(f"  crf_data_dir   : {cal['crf_data_dir']}")
    print(f"  output_dir     : {cal['output_dir']}")
    print(f"  fit_method     : {cal['fit_method']}")
    print(f"  smoothing_lambda: {cal['smoothing_lambda']}")
    print()

    results = step0.run_calibration(
        dc_dir          = cal["dc_dir"],
        reflectance_dir = cal["reflectance_dir"],
        output_dir      = cal["output_dir"],
        crf_dir         = cal["crf_data_dir"],
        config = {
            "fit_method":       cal["fit_method"],
            "smoothing_lambda": cal["smoothing_lambda"],
        },
    )

    print()
    print("Calibration complete. Saved:")
    for key in ("Sd_path", "b_path", "Smax_path", "crf_path"):
        print(f"  {results[key]}")


if __name__ == "__main__":
    main()
