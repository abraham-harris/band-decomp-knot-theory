"""Small helpers for creating and recording experiment runs.

This module is intentionally independent of the training code.  Importing it
does not create directories or write files; those actions happen only when its
functions are called.
"""

import csv
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def load_config(config_path):
    """Load an experiment configuration from a JSON file."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a JSON object")

    return config


def _slugify(name):
    """Return a filesystem-friendly experiment name."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())
    slug = slug.strip("_-")
    return slug or "experiment"


def create_run_directory(name, runs_root="runs"):
    """Create and return a unique timestamped directory for an experiment."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    run_directory = runs_root / f"{timestamp}_{_slugify(name)}"
    run_directory.mkdir(exist_ok=False)
    return run_directory


def save_config(config, run_directory):
    """Save the resolved configuration used by a run."""
    config_path = Path(run_directory) / "config.json"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    return config_path


def _git_information():
    """Return the current Git revision and dirty status when available."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
        dirty = None

    return {"git_commit": commit, "git_dirty": dirty}


def initialize_run(config, name=None, runs_root="runs"):
    """Create a run directory and write its config and initial metadata."""
    experiment_name = name or config.get("name", "experiment")
    git_information = _git_information()
    run_directory = create_run_directory(experiment_name, runs_root=runs_root)
    save_config(config, run_directory)

    run_info = {
        "name": experiment_name,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        **git_information,
    }
    _write_json(run_directory / "run_info.json", run_info)
    return run_directory


def update_run_status(run_directory, status, error=None):
    """Update completion information for an existing experiment run."""
    run_info_path = Path(run_directory) / "run_info.json"
    with run_info_path.open("r", encoding="utf-8") as run_info_file:
        run_info = json.load(run_info_file)

    run_info["status"] = status
    run_info["finished_at"] = None if status == "running" else datetime.now(timezone.utc).isoformat()
    if error is not None:
        run_info["error"] = str(error)
    else:
        run_info.pop("error", None)

    _write_json(run_info_path, run_info)


def save_metrics(rows, metrics_path):
    """Save a sequence of metric dictionaries as a CSV file."""
    rows = list(rows)
    if not rows:
        return None

    fieldnames = []
    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return metrics_path


def _write_json(path, value):
    """Replace run metadata atomically so an interrupted update remains readable."""
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}_", suffix=".tmp", delete=False,
        ) as json_file:
            temporary_path = Path(json_file.name)
            json.dump(value, json_file, indent=2)
            json_file.write("\n")
            json_file.flush()
            os.fsync(json_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
