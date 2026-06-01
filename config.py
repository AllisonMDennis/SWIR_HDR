"""
config.py — Load and resolve the SWIR_HDR pipeline configuration.

Usage:
    from config import load_config
    cfg = load_config()          # loads config.yaml from repo root
    cfg = load_config("path/to/config.yaml")
"""

import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Keys whose values are relative paths to be resolved against REPO_ROOT
_PATH_KEYS = {'dir', 'dc_dir', 'reflectance_dir', 'crf_data_dir', 'output_dir', 'directory'}


def load_config(path=None):
    """
    Load config.yaml and resolve relative paths against the repo root.

    Returns a nested dict mirroring the YAML structure, with all path-valued
    keys converted to absolute path strings.
    """
    if path is None:
        path = REPO_ROOT / "config.yaml"

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    _resolve_paths(cfg)
    return cfg


def _resolve_paths(node):
    """Recursively resolve path-valued keys to absolute strings."""
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if isinstance(value, dict):
            _resolve_paths(value)
        elif key in _PATH_KEYS and isinstance(value, str):
            node[key] = str(REPO_ROOT / value)
