"""Small helpers shared by the experiment scripts."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


# Configuration and path handling.
def load_config(config_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    """Load YAML and resolve every entry under paths from the repository root."""
    config_path = Path(config_path)
    repo_root = Path(repo_root)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Copy config-example.yaml to config.yaml and update its paths."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    configured_paths = config.get("paths")
    if not isinstance(configured_paths, dict):
        raise ValueError("config.yaml must contain a 'paths' mapping.")

    resolved_paths: dict[str, Path] = {}
    for name, value in configured_paths.items():
        if not isinstance(value, str):
            raise TypeError(f"paths.{name} must be a string.")

        expanded = Path(os.path.expandvars(os.path.expanduser(value)))
        resolved_paths[name] = expanded if expanded.is_absolute() else repo_root / expanded

    return {**config, "paths": resolved_paths}


# Reproducibility across independent runs.
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for one independent run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Small result summaries used by every experiment.
def summarize_runs(
    results: Iterable[dict[str, float | int]],
    metric_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Return population mean and standard deviation for selected metrics."""
    rows = list(results)
    if not rows:
        raise ValueError("Cannot summarize an empty result list.")

    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.asarray([row[metric_name] for row in rows], dtype=np.float64)
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary


# Human-readable output for aggregate metrics and settings.
def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Write a small, human-readable JSON result file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
