# run_pipeline.py
# -----------------------------------------------------------------------------
# YAML-driven AutoClean runner.
# Usage:
#   python run_pipeline.py --config config.yaml
#
# This script:
#   1) Loads config.yaml (I/O paths, all processing options)
#   2) Builds CleanConfig
#   3) Fits + transforms the input dataset
#   4) Saves processed CSV and artifacts
# -----------------------------------------------------------------------------

import os
import argparse
import json
from typing import Any, Dict

import pandas as pd

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

from cleaning_pipeline import AutoCleaner, CleanConfig


# -----------------------------
# Logging helpers
# -----------------------------
def log(msg: str, level: str, desired: str):
    """Conditional print based on log level preference."""
    order = {"info": 1, "debug": 2}
    if order[level] <= order.get(desired, 1):
        print(f"[{level.upper()}] {msg}")


# -----------------------------
# Config utilities
# -----------------------------
def load_yaml_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Please install with: pip install pyyaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def validate_and_normalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate presence of top-level keys and fill sensible defaults.
    Returns a normalized dict with 'io', 'processing', 'run'.
    """
    cfg = cfg.copy()
    if "io" not in cfg:
        raise ValueError("config.yaml missing required 'io' section.")
    if "processing" not in cfg:
        raise ValueError("config.yaml missing required 'processing' section.")
    if "run" not in cfg:
        cfg["run"] = {}

    # IO defaults
    io = cfg["io"]
    required_io = ["input_path", "output_path", "artifacts_dir"]
    for k in required_io:
        if k not in io or io[k] in (None, ""):
            raise ValueError(f"'io.{k}' is required in config.yaml")

    io.setdefault("save_columns_list", False)
    io.setdefault("target_column", None)

    # run defaults
    run = cfg["run"]
    run.setdefault("log_level", "info")
    if run["log_level"] not in ("info", "debug"):
        raise ValueError("run.log_level must be 'info' or 'debug'")

    # processing defaults are handled by CleanConfig, but we ensure keys exist
    processing = cfg["processing"]
    # no strict validation of choices here—sklearn/AutoCleaner will raise clear errors if invalid

    return cfg


def build_clean_config(processing: Dict[str, Any], target_column: Any, artifacts_dir: str) -> CleanConfig:
    """
    Map the YAML 'processing' + some 'io' fields to CleanConfig.
    Missing keys will fall back to CleanConfig defaults.
    """
    kwargs = dict(processing)  # copy all processing options
    kwargs["target_column"] = target_column
    kwargs["save_artifacts_dir"] = artifacts_dir
    return CleanConfig(**kwargs)


# -----------------------------
# I/O helpers
# -----------------------------
def read_any_table(path: str, log_level: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    log(f"Reading input: {path}", "info", log_level)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext in (".csv",):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension for input_path: {ext}")


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# -----------------------------
# Main
# -----------------------------
def main():
    # CLI: only YAML is needed now
    parser = argparse.ArgumentParser(description="YAML-driven AutoClean runner")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # 1) Load + validate config
    raw_cfg = load_yaml_config(args.config)
    cfg = validate_and_normalize(raw_cfg)

    io = cfg["io"]
    processing = cfg["processing"]
    run = cfg["run"]
    log_level = run["log_level"]

    input_path = io["input_path"]
    output_path = io["output_path"]
    artifacts_dir = io["artifacts_dir"]
    save_columns_list = bool(io.get("save_columns_list", False))
    target_column = io.get("target_column", None)

    # 2) Build CleanConfig
    clean_cfg = build_clean_config(processing, target_column, artifacts_dir)

    # 3) Read input
    df = read_any_table(input_path, log_level)

    # 4) Fit + transform
    log("Fitting cleaner...", "info", log_level)
    cleaner = AutoCleaner(clean_cfg).fit(df)
    log("Transforming data...", "info", log_level)
    df_clean = cleaner.transform(df)

    # 5) Save output CSV
    ensure_parent_dir(output_path)
    df_clean.to_csv(output_path, index=False)
    log(f"Saved processed CSV: {output_path}", "info", log_level)

    # 6) Optionally save columns list (no target)
    if save_columns_list:
        os.makedirs(artifacts_dir, exist_ok=True)
        cols_path = os.path.join(artifacts_dir, "columns.txt")
        cols_iter = (
            df_clean.drop(columns=[target_column]).columns
            if (target_column is not None and target_column in df_clean.columns)
            else df_clean.columns
        )
        with open(cols_path, "w", encoding="utf-8") as f:
            for c in cols_iter:
                f.write(c + "\n")
        log(f"Wrote columns list: {cols_path}", "info", log_level)

    # 7) Log artifacts location for clarity
    log(f"Artifacts directory: {artifacts_dir}", "debug", log_level)
    log("Done.", "info", log_level)


if __name__ == "__main__":
    main()
