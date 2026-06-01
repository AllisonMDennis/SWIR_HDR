#!/usr/bin/env python3
"""
example_radiance.py — Import subject images and compute HDR radiance maps.

Reads parameters from config.yaml, loads calibration arrays produced by
example_calibration.py, preprocesses the subject H5 files (Step 1), and
computes a per-channel HDR radiance map (Step 2).

Output files are written to:
    {subject_dir}/{base_data_folder}/final_data/

Usage:
    python example_radiance.py
    python example_radiance.py --config path/to/config.yaml
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from config import load_config
import Step1_import as step1
import Step2_radiance as step2

_WEIGHTING_FUNCTIONS = {
    "debevec":   step2.debevec,
    "robertson": step2.robertson,
    "broadhat":  step2.broadhat,
    "vinegoni":  step2.vinegoni,
}


def main():
    parser = argparse.ArgumentParser(description="Run SWIR_HDR Steps 1-2: import and HDR fusion.")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: repo root config.yaml)")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    cal  = cfg["calibration"]
    imp  = cfg["import"]
    rad  = cfg["radiance"]
    subj = cfg["subject"]

    print("=" * 60)
    print("  SWIR_HDR — Steps 1-2: Import and HDR Radiance")
    print("=" * 60)
    # HDR pipeline output goes to output/<experiment_title>/ (git-ignored),
    # not into the subject's data directory.
    base = os.path.join(imp["output_dir"], subj["experiment_title"])

    print(f"  subject dir    : {subj['directory']}")
    print(f"  experiment     : {subj['experiment_title']}")
    print(f"  calibration dir: {cal['output_dir']}")
    print(f"  output dir     : {base}")
    print(f"  preprocessing  : {imp['preprocessing_ops']}")
    print(f"  weighting fn   : {rad['weighting_function']}")
    print(f"  method         : {rad['method']}")
    print()

    # --- Load calibration arrays ---
    calib_dir = cal["output_dir"]
    params = {}
    for key in ["Sd", "b", "Smax"]:
        path = os.path.join(calib_dir, f"{key}.npy")
        params[key] = np.load(path)
        print(f"  Loaded {key}.npy  shape={params[key].shape}  "
              f"range=[{params[key].min():.2f}, {params[key].max():.2f}]")

    crf = np.load(os.path.join(calib_dir, "crf.npy"))
    print(f"  Loaded crf.npy  shape={crf.shape}  "
          f"range=[{crf.min():.2f}, {crf.max():.2f}]")
    print()

    # --- Step 1: import and preprocess ---
    print("Step 1 — Import and preprocess subject H5 files")
    step1.process_and_save(
        directory        = subj["directory"],
        experiment_title = subj["experiment_title"],
        base_data_folder = base,
        operations       = imp["preprocessing_ops"],
        params           = params,
    )
    print()

    # --- Step 2: HDR radiance fusion ---
    print("Step 2 — HDR radiance map computation")
    weighting_fn = _WEIGHTING_FUNCTIONS[rad["weighting_function"]]

    processed = step2.process_hdr_images(
        directory          = subj["directory"],
        experiment_title   = subj["experiment_title"],
        base_data_folder   = base,
        coefficients_dict  = params,
        response_curve     = crf,
        weighting_function = weighting_fn,
        method             = rad["method"],
    )

    final_dir = os.path.join(base, "final_data")
    print(f"\nComplete — {len(processed)} radiance map(s) saved to {final_dir}")
    for item in processed:
        rm = item["radiance_map"]
        print(f"  {item['key']}: shape={rm.shape}  "
              f"range=[{np.nanmin(rm):.4f}, {np.nanmax(rm):.4f}]")


if __name__ == "__main__":
    main()
