"""Compare Meiners-style braid DP with saved batch RL decompositions.

Run with the completed copy of a batch run, for example::

    ../.knotenv/bin/python compare_dp_rl.py \
        --rl-run runs/dp_rl_historical_completion_47ac89b

Results are saved after each braid. Repeating the command resumes completed
cases; use --retry-failed after raising the per-instance time limit.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time

from dp_braid import braid_upper_bound, normal_form_key


FIELDS = (
    "braid_id", "braid_index", "true_rank", "initial_length",
    "rl_best_length", "rl_flat_word_length",
    "dp_original_status", "dp_original_bound", "dp_original_decomposition",
    "dp_original_certificates", "dp_original_max_band_length",
    "dp_original_seconds", "dp_original_error",
    "dp_after_rl_status", "dp_after_rl_bound", "dp_after_rl_decomposition",
    "dp_after_rl_certificates", "dp_after_rl_max_band_length",
    "dp_after_rl_seconds", "dp_after_rl_error",
    "hybrid_best_length", "hybrid_source", "best_of_all_length",
)
STAGES = ("dp_original", "dp_after_rl")


class CaseTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise CaseTimeout("DP case exceeded its time limit")


def _parse_word(text):
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("dataset braid word must be enclosed in braces")
    middle = text[1:-1].strip()
    return [int(part.strip()) for part in middle.split(",")] if middle else []


def _load_inputs(rl_run, *, require_complete=True):
    config = json.loads((rl_run / "config.json").read_text(encoding="utf-8"))
    dataset_bytes = (rl_run / "dataset.csv").read_bytes()
    digest = hashlib.sha256(dataset_bytes).hexdigest()
    if digest != config.get("dataset_sha256"):
        raise ValueError("RL run dataset snapshot hash does not match its config")
    with (rl_run / "dataset.csv").open(newline="", encoding="utf-8-sig") as source:
        dataset = list(csv.DictReader(source))
    with (rl_run / "results.csv").open(newline="", encoding="utf-8") as source:
        rl_rows = list(csv.DictReader(source))
    by_id = {}
    for row in rl_rows:
        braid_id = int(row["braid_id"])
        if braid_id in by_id or braid_id < 0 or braid_id >= len(dataset):
            raise ValueError(f"duplicate or invalid RL braid ID {braid_id}")
        by_id[braid_id] = row
    if require_complete and set(by_id) != set(range(len(dataset))):
        missing = sorted(set(range(len(dataset))) - set(by_id))
        raise ValueError(f"RL run is incomplete; missing braid IDs: {missing}")
    return dataset, by_id, digest


def _load_or_create_output(output_dir, rl_run, dataset_digest):
    config_path = output_dir / "config.json"
    engine_digest = hashlib.sha256(
        (Path(__file__).parent / "dp_braid.py").read_bytes()).hexdigest()
    expected = {
        "rl_run": str(rl_run.resolve()),
        "dataset_sha256": dataset_digest,
        "dp_engine_sha256": engine_digest,
        "algorithm": "Meiners braid upper-bound DP; longest conjugation; free-reduced input",
        "max_environment_band_length": 13,
    }
    if output_dir.exists():
        if not config_path.exists():
            raise ValueError(f"Existing output directory lacks config.json: {output_dir}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != expected:
            raise ValueError("Comparison configuration or DP code changed; choose a new output directory")
    else:
        output_dir.mkdir(parents=True)
        config_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    results_path = output_dir / "results.csv"
    if not results_path.exists():
        _save_rows(results_path, [])
    with results_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Existing comparison results have unexpected columns")
        rows = {}
        for row in reader:
            braid_id = int(row["braid_id"])
            if braid_id in rows:
                raise ValueError(f"Duplicate comparison braid ID {braid_id}")
            rows[braid_id] = row
    return results_path, rows


def _save_rows(path, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=path.parent,
                prefix=".dp_results_", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _check_rl_row(dataset_row, rl_row, braid_id):
    braid_index = int(dataset_row["Braid index"])
    true_rank = int(dataset_row["Braid ranks"])
    original = _parse_word(dataset_row["Braid word"])
    bands = json.loads(rl_row["best_decomposition"])
    if (type(bands) is not list or
            any(type(band) is not list or any(type(x) is not int for x in band)
                for band in bands)):
        raise ValueError(f"RL braid {braid_id} lacks a list-of-lists decomposition")
    if (int(rl_row["braid_index"]) != braid_index or
            int(rl_row["true_rank"]) != true_rank or
            int(rl_row["initial_length"]) != len(original) or
            int(rl_row["best_length"]) != len(bands)):
        raise ValueError(f"RL braid {braid_id} metadata does not match its dataset row")
    flat = [crossing for band in bands for crossing in band]
    if normal_form_key(braid_index, original) != normal_form_key(braid_index, flat):
        raise ValueError(f"RL braid {braid_id} best decomposition changes the braid")
    return braid_index, true_rank, original, bands, flat


def _run_stage(word, braid_index, true_rank, timeout_seconds):
    started = time.perf_counter()
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    if timeout_seconds:
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        result = braid_upper_bound(word, braid_index)
        return {
            "status": "completed", "bound": result.upper_bound,
            "decomposition": json.dumps(result.decomposition, separators=(",", ":")),
            "certificates": json.dumps(result.certificates, separators=(",", ":")),
            "max_band_length": max(map(len, result.decomposition), default=0),
            "seconds": round(time.perf_counter() - started, 3),
            "error": ("Verified upper bound is below supplied known rank"
                      if result.upper_bound < true_rank else ""),
        }
    except Exception as error:
        return {
            "status": "timeout" if isinstance(error, CaseTimeout) else "error",
            "bound": "", "decomposition": "", "certificates": "",
            "max_band_length": "",
            "seconds": round(time.perf_counter() - started, 3),
            "error": f"{type(error).__name__}: {error}"[:500],
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _update_comparison(row):
    rl = int(row["rl_best_length"])
    after = (int(row["dp_after_rl_bound"])
             if row["dp_after_rl_status"] == "completed" else None)
    original = (int(row["dp_original_bound"])
                if row["dp_original_status"] == "completed" else None)
    hybrid = min(rl, after) if after is not None else rl
    row["hybrid_best_length"] = hybrid
    row["hybrid_source"] = "dp_after_rl" if after is not None and after < rl else "rl"
    row["best_of_all_length"] = min(hybrid, original) if original is not None else hybrid


def _summary(rows, dataset_count):
    values = list(rows.values())
    paired = [row for row in values if all(
        row[f"{stage}_status"] == "completed" for stage in STAGES)]
    return {
        "total_braids": dataset_count,
        "rows_saved": len(values),
        "paired_completed": len(paired),
        "dp_original_completed": sum(row["dp_original_status"] == "completed"
                                     for row in values),
        "dp_after_rl_completed": sum(row["dp_after_rl_status"] == "completed"
                                     for row in values),
        "timeout_or_error_cases": sorted(int(row["braid_id"]) for row in values
                                         if any(row[f"{stage}_status"] != "completed"
                                                for stage in STAGES)),
        "rl_reached_known_rank": sum(int(row["rl_best_length"]) == int(row["true_rank"])
                                     for row in values),
        "dp_original_reached_known_rank": sum(
            int(row["dp_original_bound"]) == int(row["true_rank"]) for row in values
            if row["dp_original_status"] == "completed"),
        "dp_after_rl_reached_known_rank": sum(
            int(row["dp_after_rl_bound"]) == int(row["true_rank"]) for row in values
            if row["dp_after_rl_status"] == "completed"),
        "hybrid_reached_known_rank": sum(
            int(row["hybrid_best_length"]) == int(row["true_rank"]) for row in values),
        "best_of_all_reached_known_rank": sum(
            int(row["best_of_all_length"]) == int(row["true_rank"]) for row in values),
        "hybrid_improved_over_rl": sum(
            int(row["hybrid_best_length"]) < int(row["rl_best_length"])
            for row in values),
        "dp_original_better_than_rl": sum(
            int(row["dp_original_bound"]) < int(row["rl_best_length"])
            for row in values if row["dp_original_status"] == "completed"),
        "dp_after_rl_better_than_dp_original_on_paired": sum(
            int(row["dp_after_rl_bound"]) < int(row["dp_original_bound"])
            for row in paired),
        "dp_after_rl_better_than_dp_original_ids": [
            int(row["braid_id"]) for row in paired
            if int(row["dp_after_rl_bound"]) < int(row["dp_original_bound"])],
        "dp_seconds_by_stage": {
            stage: round(sum(float(row[f"{stage}_seconds"]) for row in values
                             if row[f"{stage}_status"] == "completed"), 3)
            for stage in STAGES},
        "witnesses_over_environment_band_limit": sum(
            int(row[f"{stage}_max_band_length"]) > 13 for row in values
            for stage in STAGES if row[f"{stage}_status"] == "completed"),
        "bounds_below_supplied_rank": [
            {"braid_id": int(row["braid_id"]), "stage": stage,
             "bound": int(row[f"{stage}_bound"]), "supplied_rank": int(row["true_rank"])}
            for row in values for stage in STAGES
            if row[f"{stage}_status"] == "completed" and
            int(row[f"{stage}_bound"]) < int(row["true_rank"])],
    }


def _write_report(output_dir, summary):
    saved = summary["rows_saved"]
    original = summary["dp_original_completed"]
    after = summary["dp_after_rl_completed"]
    lines = [
        "# Dynamic programming and RL comparison",
        "",
        f"Evaluated {saved}/{summary['total_braids']} saved RL braids; "
        f"both DP inputs completed for {summary['paired_completed']}.",
        "",
        "| Method | Matched supplied rank | Evaluated |",
        "| --- | ---: | ---: |",
        f"| RL | {summary['rl_reached_known_rank']} | {saved} |",
        f"| DP on original word | {summary['dp_original_reached_known_rank']} | {original} |",
        f"| DP after RL | {summary['dp_after_rl_reached_known_rank']} | {after} |",
        f"| RL followed by DP, taking the better result | "
        f"{summary['hybrid_reached_known_rank']} | {saved} |",
        f"| Best of all three | {summary['best_of_all_reached_known_rank']} | {saved} |",
        "",
        f"DP after RL improved the RL band count for "
        f"{summary['hybrid_improved_over_rl']} braids. It beat DP on the "
        f"original word for {summary['dp_after_rl_better_than_dp_original_on_paired']} "
        f"braid{'' if summary['dp_after_rl_better_than_dp_original_on_paired'] == 1 else 's'}: "
        f"{summary['dp_after_rl_better_than_dp_original_ids']}.",
        "",
        f"{summary['witnesses_over_environment_band_limit']} completed DP "
        "witnesses contain a band longer than BandEnv's 13-crossing limit.",
        "",
        f"DP time on this machine: {summary['dp_seconds_by_stage']['dp_original']}s "
        "for original words and "
        f"{summary['dp_seconds_by_stage']['dp_after_rl']}s for RL words. "
        "Saved RL runtimes may come from a different machine.",
        "",
        f"Timeout or error IDs: {summary['timeout_or_error_cases']}.",
        f"Verified bounds below supplied ranks: "
        f"{summary['bounds_below_supplied_rank']}.",
        "",
        "Each decomposition in results.csv is a JSON list of band words "
        "(list of lists of signed generators).",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def compare(rl_run, output_dir, *, timeout_seconds=120, retry_failed=False,
            braid_ids=None):
    rl_run = Path(rl_run)
    output_dir = Path(output_dir)
    dataset, rl_rows, digest = _load_inputs(
        rl_run, require_complete=braid_ids is None)
    results_path, rows = _load_or_create_output(output_dir, rl_run, digest)
    selected = range(len(dataset)) if braid_ids is None else sorted(set(braid_ids))
    for braid_id in selected:
        if not 0 <= braid_id < len(dataset):
            raise ValueError(f"braid ID {braid_id} is outside the dataset")
        if braid_id not in rl_rows:
            raise ValueError(f"RL run has no saved result for braid ID {braid_id}")
        old = rows.get(braid_id)
        if old is not None and all(old[f"{stage}_status"] == "completed"
                                   for stage in STAGES):
            continue
        if old is not None and not retry_failed:
            continue
        braid_index, true_rank, original, bands, flat = _check_rl_row(
            dataset[braid_id], rl_rows[braid_id], braid_id)
        row = {field: "" for field in FIELDS}
        row.update(braid_id=braid_id, braid_index=braid_index,
                   true_rank=true_rank, initial_length=len(original),
                   rl_best_length=len(bands), rl_flat_word_length=len(flat))
        print(f"Braid {braid_id}/{len(dataset) - 1}: original {len(original)}, "
              f"RL {len(bands)} bands / {len(flat)} crossings", flush=True)
        for stage, word in (("dp_original", original), ("dp_after_rl", flat)):
            result = _run_stage(word, braid_index, true_rank, timeout_seconds)
            for key, value in result.items():
                row[f"{stage}_{key}"] = value
            print(f"  {stage}: {result['status']} "
                  f"{result['bound']} ({result['seconds']}s)", flush=True)
        _update_comparison(row)
        rows[braid_id] = row
        _save_rows(results_path, [rows[key] for key in sorted(rows)])
        (output_dir / "summary.json").write_text(
            json.dumps(_summary(rows, len(dataset)), indent=2) + "\n",
            encoding="utf-8")
    summary = _summary(rows, len(dataset))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-run", required=True, type=Path,
                        help="Completed copy of the saved batch RL run")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/dp_rl_comparison_100_epochs"))
    parser.add_argument("--timeout-seconds", type=float, default=120,
                        help="Time limit for each DP input; 0 disables it")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Recompute cases with a timeout or error")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--braid-ids", type=int, nargs="+",
                        help="Optional subset for a pilot; later runs resume the rest")
    subset.add_argument("--through-id", type=int,
                        help="Process available saved RL rows from ID 0 through this ID")
    args = parser.parse_args()
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be nonnegative")
    if args.through_id is not None and args.through_id < 0:
        parser.error("--through-id must be nonnegative")
    selected = (list(range(args.through_id + 1))
                if args.through_id is not None else args.braid_ids)
    summary = compare(args.rl_run, args.output_dir,
                      timeout_seconds=args.timeout_seconds,
                      retry_failed=args.retry_failed, braid_ids=selected)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
