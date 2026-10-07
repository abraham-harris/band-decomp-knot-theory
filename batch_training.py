"""Train PPO on CSV braids, independently or iteratively, saving completed results.

Run with --config, resume independent runs with --resume, or plot saved rows
with --plot-only. Iterative runs save final policies in their models directory.
"""

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time

from experiment_utils import initialize_run, load_config, update_run_status


RESULT_FIELDS = (
    "braid_id", "braid_index", "true_rank", "initial_length", "best_length",
    "rank_gap", "seed", "epochs_completed", "episodes_completed",
    "elapsed_seconds", "best_decomposition", "best_action_sequence", "best_found_at",
)
TRAINING_FIELDS = (
    "epochs", "env_samples", "max_actions", "max_num_bands", "learning_rate",
    "gamma", "batch_size", "epsilon", "policy_epochs",
)


def _resolve_config(config, trainer, name=None):
    allowed = set(TRAINING_FIELDS) | {"name", "mode", "seed", "data_path", "iterative", "braid_index"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
    if config.get("mode", "batch_specific_training") != "batch_specific_training":
        raise ValueError("mode must be 'batch_specific_training'")
    iterative = config.get("iterative", False)
    braid_index = config.get("braid_index")
    if type(iterative) is not bool:
        raise ValueError("iterative must be a boolean")
    if iterative and (type(braid_index) is not int or braid_index < 2):
        raise ValueError("iterative training requires braid_index >= 2 as an upper limit")
    if not iterative and braid_index is not None:
        raise ValueError("braid_index requires iterative=true")
    parameters = inspect.signature(trainer).parameters
    resolved = {
        "name": name if name is not None else config.get("name", "batch_braids"),
        "mode": "batch_specific_training",
        "data_path": config.get("data_path", "data/braids_with_ranks.csv"),
        "seed": config.get("seed"),
        "iterative": iterative,
        "braid_index": braid_index,
        **{key: config.get(key, parameters[key].default) for key in TRAINING_FIELDS},
    }
    seed = resolved["seed"]
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("seed must be null or an integer from 0 through 2**32 - 1")
    for key in ("epochs", "env_samples", "max_actions", "max_num_bands",
                "batch_size", "policy_epochs"):
        if type(resolved[key]) is not int or resolved[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "gamma", "epsilon"):
        value = resolved[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (key == "learning_rate" and value == 0)
                or (key == "gamma" and value > 1)):
            raise ValueError(f"Invalid {key}: {value}")
    return resolved


def _load_braids(dataset_bytes, max_num_bands):
    reader = csv.DictReader(io.StringIO(dataset_bytes.decode("utf-8-sig")))
    required = {"Braid index", "Braid ranks", "Braid word"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Dataset must contain columns: {', '.join(sorted(required))}")
    braids = []
    for braid_id, row in enumerate(reader):
        try:
            braid_index = int(row["Braid index"])
            true_rank = int(row["Braid ranks"])
            word_text = row["Braid word"].strip()
            if not (word_text.startswith("{") and word_text.endswith("}")):
                raise ValueError("Braid word must be an ordered word enclosed in braces")
            # Do not parse the braces as a Python set: order and repetitions matter.
            content = word_text[1:-1].strip()
            word = [int(item.strip()) for item in content.split(",")] if content else []
            if braid_index < 2 or true_rank < 0:
                raise ValueError("Invalid braid index or rank")
            if any(crossing == 0 or abs(crossing) >= braid_index for crossing in word):
                raise ValueError("Crossing is outside this braid's strand count")
            if len(word) > max_num_bands:
                raise ValueError(f"Word length {len(word)} exceeds max_num_bands={max_num_bands}")
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"Dataset braid {braid_id} (CSV line {braid_id + 2}): {error}") from error
        braids.append({"braid_id": braid_id, "braid_index": braid_index,
                       "true_rank": true_rank, "word": word})
    if not braids:
        raise ValueError("Dataset contains no braids")
    return braids


def _save_results(rows, path):
    """Replace the CSV atomically so an interruption cannot leave a partial row."""
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=".results_", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_results(path, braids):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(RESULT_FIELDS):
            raise ValueError("Saved results.csv has unexpected columns")
        rows = list(reader)
    seen = set()
    for row in rows:
        braid_id = int(row["braid_id"])
        if braid_id in seen or not 0 <= braid_id < len(braids):
            raise ValueError("Saved results contain duplicate or invalid braid IDs")
        seen.add(braid_id)
        braid = braids[braid_id]
        if (int(row["braid_index"]) != braid["braid_index"]
                or int(row["true_rank"]) != braid["true_rank"]
                or int(row["initial_length"]) != len(braid["word"])):
            raise ValueError(f"Saved result {braid_id} does not match the dataset")
    return rows


def _select_braids(braids, config):
    if not config["iterative"]:
        return braids
    selected = [braid for braid in braids if braid["braid_index"] <= config["braid_index"]]
    # Best achieved length is not known until after training.
    return sorted(selected, key=lambda braid: (
        braid["true_rank"], len(braid["word"]), braid["braid_id"]
    ))


def _save_policy_model(policy_state, path):
    """Atomically save final policy weights in a run's models directory."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.stem}_",
            suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            torch.save(policy_state, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def plot_batch_results(run_directory):
    """Plot completed rows only, including results from an interrupted run."""
    import matplotlib.pyplot as plt

    run_directory = Path(run_directory)
    with (run_directory / "results.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("No completed braids to plot yet")
    rows.sort(key=lambda row: tuple(int(row[key]) for key in
                                   ("true_rank", "initial_length", "best_length", "braid_id")))
    x_values = range(1, len(rows) + 1)
    figure, axis = plt.subplots(figsize=(13, 4))
    try:
        axis.scatter(x_values, [int(row["true_rank"]) for row in rows], label="True Rank")
        axis.scatter(x_values, [int(row["initial_length"]) for row in rows],
                     color="green", label="Initial Band Decomp Length")
        axis.scatter(x_values, [int(row["best_length"]) for row in rows],
                     marker="+", label="Best Band Decomp Length Achieved")
        matched = sum(int(row["best_length"]) == int(row["true_rank"]) for row in rows)
        axis.set_title(f"{len(rows)} completed braids; {matched} reached the known rank")
        axis.set_ylabel("Band Decomposition Length")
        axis.set_xlabel("Test Braid Identifier (sorted; completed braids only)")
        axis.legend()
        axis.grid()
        figure.tight_layout()
        plot_path = run_directory / "plot.png"
        figure.savefig(plot_path)
    finally:
        plt.close(figure)
    return plot_path


def run_batch_experiment(config_path=None, *, resume=None, name=None, runs_root="runs"):
    """Save every completed braid before starting the next; restart unfinished work."""
    import numpy as np
    import torch
    from main import ppo_single_braid

    if (config_path is None) == (resume is None):
        raise ValueError("Supply either config_path or resume")
    if resume is not None:
        if name is not None:
            raise ValueError("name applies only to new runs")
        run_directory = Path(resume)
        config = load_config(run_directory / "config.json")
        dataset_bytes = (run_directory / "dataset.csv").read_bytes()
        if hashlib.sha256(dataset_bytes).hexdigest() != config["dataset_sha256"]:
            raise ValueError("The run's dataset snapshot has changed")
        # Validate saved parameters using the same rules as a new experiment.
        resolved = _resolve_config({key: value for key, value in config.items()
                                    if key not in {"dataset_sha256", "num_braids"}},
                                   ppo_single_braid)
        config.update(iterative=resolved["iterative"], braid_index=resolved["braid_index"])
        braids = _load_braids(dataset_bytes, config["max_num_bands"])
        rows = _load_results(run_directory / "results.csv", braids)
    else:
        config = _resolve_config(load_config(config_path), ppo_single_braid, name)
        dataset_bytes = Path(config["data_path"]).read_bytes()
        braids = _load_braids(dataset_bytes, config["max_num_bands"])
        selected_count = len(_select_braids(braids, config))
        if not selected_count:
            raise ValueError(f"Dataset contains no braids up to braid_index={config['braid_index']}")
        config.update(dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
                      num_braids=selected_count)
        run_directory = initialize_run(config, runs_root=runs_root)
        (run_directory / "dataset.csv").write_bytes(dataset_bytes)
        rows = []
        _save_results(rows, run_directory / "results.csv")

    braids = _select_braids(braids, config)
    if not braids:
        raise ValueError(f"Dataset contains no braids up to braid_index={config['braid_index']}")
    if config["iterative"] and rows:
        raise ValueError("Iterative training cannot resume without optimizer state; start a new run")
    training_states = {}

    print(f"Run directory: {run_directory}", flush=True)
    completed = {int(row["braid_id"]) for row in rows}
    training_description = (f"one continuing policy per braid index up to {config['braid_index']}"
                            if config["iterative"] else "a fresh policy per braid")
    print(f"Completed: {len(completed)}/{len(braids)}; {training_description}.",
          flush=True)
    update_run_status(run_directory, "running")
    kwargs = {key: config[key] for key in TRAINING_FIELDS}
    try:
        for braid in braids:
            braid_id = braid["braid_id"]
            if braid_id in completed:
                continue
            seed = None if config["seed"] is None else (config["seed"] + braid_id) % 2**32
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
            print(f"Training braid {braid_id} ({len(rows) + 1}/{len(braids)}): "
                  f"{braid['braid_index']} strands, initial length {len(braid['word'])}", flush=True)
            start = time.monotonic()
            training_result = ppo_single_braid(
                band_decomposition=braid["word"], braid_index=braid["braid_index"],
                save_path=None, report_path=None, collect_logs=False, show_progress=False,
                training_state=training_states.get(braid["braid_index"]),
                return_training_state=config["iterative"],
                **kwargs,
            )
            report = training_result[4]
            row = {
                "braid_id": braid_id, "braid_index": braid["braid_index"],
                "true_rank": braid["true_rank"], "initial_length": report["initial_length"],
                "best_length": report["best_length"],
                "rank_gap": report["best_length"] - braid["true_rank"], "seed": seed,
                "epochs_completed": config["epochs"],
                "episodes_completed": config["epochs"] * config["env_samples"],
                "elapsed_seconds": round(time.monotonic() - start, 3),
                **{key: json.dumps(report[key]) for key in
                   ("best_decomposition", "best_action_sequence", "best_found_at")},
            }
            if config["iterative"]:
                training_states[braid["braid_index"]] = training_result[5]
            rows.append(row)
            _save_results(rows, run_directory / "results.csv")
            print(f"Saved braid {braid_id}: best length {row['best_length']}, "
                  f"known rank {row['true_rank']} ({len(rows)}/{len(braids)} completed)", flush=True)
            del training_result, report
        if config["iterative"]:
            for braid_index, state in sorted(training_states.items()):
                _save_policy_model(state["policy"],
                                   run_directory / "models" / f"braid_index_{braid_index}.pt")
        plot_batch_results(run_directory)
        update_run_status(run_directory, "completed")
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        update_run_status(run_directory, status, error=f"{type(error).__name__}: {error}")
        raise
    return run_directory


def _handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="JSON config for a new experiment")
    source.add_argument("--resume", help="Existing run directory; skip completed braids")
    source.add_argument("--plot-only", metavar="RUN_DIRECTORY", help="Plot saved results without training")
    parser.add_argument("--name", help="Override the name of a new experiment")
    parser.add_argument("--runs-root", default="runs", help="Parent directory for new runs (default: runs)")
    args = parser.parse_args()
    if args.name is not None and args.config is None:
        parser.error("--name requires --config")
    if args.plot_only is not None:
        print(f"Plot: {plot_batch_results(args.plot_only)}")
        return
    signal.signal(signal.SIGTERM, _handle_termination)
    try:
        run_batch_experiment(args.config, resume=args.resume, name=args.name, runs_root=args.runs_root)
    except KeyboardInterrupt:
        print("Interrupted. Saved results are preserved; resume the printed run directory.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
