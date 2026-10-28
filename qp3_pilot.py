"""Reproducible 3-braid quasipositivity witness pilot.

The model receives only cases_public.json.  cases_private.json records how each
known quasipositive case was made, for auditing after the benchmark.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import torch

from band_env import BandEnv
from experiment_utils import initialize_run, load_config, update_run_status
from garside import left_normal_form


BUCKETS = {"2-3": (2, 3), "4-6": (4, 6), "7-10": (7, 10)}
CSV_FIELDS = ("case_id", "model_input_word", "known_positive_band_decomposition",
              "model_found_decomposition", "status", "elapsed_seconds",
              "actions_taken", "length", "exponent_sum", "negative_crossings",
              "inserted_cancelling_pairs")


def inverse(word):
    return [-crossing for crossing in reversed(word)]


def flatten(bands):
    return [crossing for band in bands for crossing in band]


def exponent_sum(word):
    return sum(1 if crossing > 0 else -1 for crossing in word)


def normal_form_key(word):
    power, factors = left_normal_form(3, word)
    return power, tuple(tuple(factor) for factor in factors)


def validate_public_cases(cases, max_word_length=24, expected_count=60):
    if len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} cases, found {len(cases)}")
    seen = set()
    counts = {bucket: 0 for bucket in BUCKETS}
    for case in cases:
        word = case["word"]
        if not word or len(word) > max_word_length:
            raise ValueError(f"Case {case['case_id']} has invalid word length")
        if any(type(crossing) is not int or crossing not in (-2, -1, 1, 2)
               for crossing in word):
            raise ValueError(f"Case {case['case_id']} is not a 3-braid word")
        key = normal_form_key(word)
        if key in seen:
            raise ValueError(f"Case {case['case_id']} duplicates a braid element")
        seen.add(key)
        if key[0] >= 0:
            raise ValueError(f"Case {case['case_id']} is a positive braid element")
        negatives = sum(crossing < 0 for crossing in word)
        bucket = case["negative_bucket"]
        if bucket not in BUCKETS or not BUCKETS[bucket][0] <= negatives <= BUCKETS[bucket][1]:
            raise ValueError(f"Case {case['case_id']} is in the wrong bucket")
        if (case["length"] != len(word) or case["exponent_sum"] != exponent_sum(word)
                or case["negative_crossings"] != negatives):
            raise ValueError(f"Case {case['case_id']} has inconsistent metadata")
        counts[bucket] += 1
    if any(count != expected_count // len(BUCKETS) for count in counts.values()):
        raise ValueError(f"Corpus buckets are not balanced: {counts}")


def _one_candidate(rng, max_word_length):
    """Construct a positive-band product, scramble it, and return its provenance."""
    band_count = rng.randint(2, 4)
    base_bands = []
    for _ in range(band_count):
        conjugator = [rng.choice((-2, -1, 1, 2)) for _ in range(rng.randint(0, 2))]
        base_bands.append(conjugator + [rng.choice((1, 2))] + inverse(conjugator))
    original_word = flatten(base_bands)
    env = BandEnv(band_decomposition=base_bands, braid_index=3,
                  max_num_bands=30, train_type="deterministic",
                  observation_type="lk_matrix")
    moves = []
    pair_count = rng.randint(0, 2)
    operations = ["create"] * pair_count + ["slide"] * rng.randint(0, 3)
    rng.shuffle(operations)
    try:
        for operation in operations:
            if operation == "create":
                crossing = rng.choice((-2, -1, 1, 2))
                position = rng.randrange(len(env.band_decomposition) + 1)
                env.create_bands(position, crossing)
                moves.append({"move": "create", "position": position, "index": crossing})
                continue
            direction = rng.choice(("left", "right"))
            position = rng.randrange(1, len(env.band_decomposition)) if direction == "left" else rng.randrange(len(env.band_decomposition) - 1)
            before = [band.copy() for band in env.band_decomposition]
            if direction == "left":
                env.left_slide(position)
            else:
                env.right_slide(position)
            if env.band_decomposition != before:
                moves.append({"move": direction, "position": position})
        mixed_word = flatten(env.band_decomposition)
    finally:
        env.close()
    global_conjugator = [rng.choice((-2, -1, 1, 2)) for _ in range(rng.randint(0, 1))]
    word = global_conjugator + mixed_word + inverse(global_conjugator)
    if len(word) > max_word_length:
        return None
    # Confirm that the environment moves preserved the braid element.
    if normal_form_key(original_word) != normal_form_key(mixed_word):
        return None
    hidden_witness = [global_conjugator + band + inverse(global_conjugator)
                      for band in base_bands]
    if normal_form_key(word) != normal_form_key(flatten(hidden_witness)):
        return None
    return word, {
        "positive_bands": base_bands,
        "hidden_positive_witness": hidden_witness,
        "environment_moves": moves,
        "inserted_cancelling_pairs": pair_count,
        "global_conjugator": global_conjugator,
    }


def generate_cases(seed=20261020, count=60, max_word_length=24,
                   max_candidates=100000):
    if count < 1 or count % len(BUCKETS):
        raise ValueError("case count must be positive and divisible by three")
    rng = random.Random(seed)
    public, private = [], []
    seen = set()
    targets = {bucket: count // len(BUCKETS) for bucket in BUCKETS}
    counts = {bucket: 0 for bucket in BUCKETS}
    for candidate_number in range(max_candidates):
        candidate_seed = rng.randrange(2**32)
        candidate = _one_candidate(random.Random(candidate_seed), max_word_length)
        if candidate is None:
            continue
        word, construction = candidate
        if not word or len(word) > max_word_length:
            continue
        negatives = sum(crossing < 0 for crossing in word)
        bucket = next((name for name, (minimum, maximum) in BUCKETS.items()
                       if minimum <= negatives <= maximum), None)
        if bucket is None or counts[bucket] >= targets[bucket]:
            continue
        key = normal_form_key(word)
        if key[0] >= 0 or key in seen:
            continue
        seen.add(key)
        case_id = f"qp3_{len(public):03d}"
        public.append({"case_id": case_id, "word": word, "length": len(word),
                       "exponent_sum": exponent_sum(word),
                       "negative_crossings": negatives, "negative_bucket": bucket})
        private.append({"case_id": case_id, "candidate_seed": candidate_seed,
                        "candidate_number": candidate_number, **construction})
        counts[bucket] += 1
        if len(public) == count:
            break
    if len(public) != count:
        raise RuntimeError(f"Could generate only {len(public)}/{count} cases: {counts}")
    validate_public_cases(public, max_word_length=max_word_length,
                          expected_count=count)
    return public, private


def is_positive_band_witness(bands, exponent):
    return (len(bands) == exponent and
            all(exponent_sum(band) == 1 for band in bands))


def verify_witness(input_word, bands):
    """Check a candidate after timing; band provenance comes from environment moves."""
    return (is_positive_band_witness(bands, exponent_sum(input_word)) and
            normal_form_key(input_word) == normal_form_key(flatten(bands)))


def _load_policy(model_path, max_num_bands):
    from main import PolicyNetwork

    torch.set_num_threads(1)
    probe = BandEnv(band_decomposition=[1], braid_index=3,
                    max_num_bands=max_num_bands, train_type="deterministic",
                    observation_type="lk_matrix")
    try:
        state_size = probe.get_state().size
        action_size = probe.max_num_actions
    finally:
        probe.close()
    model = PolicyNetwork(state_size, action_size).cpu()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        model(torch.zeros((1, state_size), dtype=torch.float32))
    return model


def search_one(case, model, *, max_num_bands=30, attempts=10,
               max_actions=150, policy_seed=1729):
    word = case["word"]
    exponent = exponent_sum(word)
    attempt_rows = []
    candidate_bands = None
    total_start = time.perf_counter()
    for attempt_index in range(attempts):
        attempt_seed = (policy_seed + int(case["case_id"].split("_")[-1]) * 1000
                        + attempt_index) % 2**32
        generator = torch.Generator(device="cpu").manual_seed(attempt_seed)
        attempt_start = time.perf_counter()
        env = BandEnv(band_decomposition=word, braid_index=3,
                      max_num_bands=max_num_bands, train_type="deterministic",
                      observation_type="lk_matrix")
        actions = []
        found = False
        try:
            state, _ = env.reset()
            with torch.inference_mode():
                for _ in range(max_actions):
                    probabilities = model(torch.from_numpy(state).unsqueeze(0)).squeeze(0)
                    action = int(torch.multinomial(probabilities, 1, generator=generator).item())
                    state, _, terminated, truncated, _ = env.step(action)
                    actions.append(action)
                    bands = env.band_decomposition
                    if is_positive_band_witness(bands, exponent):
                        candidate_bands = [band.copy() for band in bands]
                        found = True
                        break
                    if terminated or truncated:
                        break
        finally:
            env.close()
        attempt_rows.append({"attempt": attempt_index, "seed": attempt_seed,
                             "actions": actions, "actions_taken": len(actions),
                             "elapsed_seconds": time.perf_counter() - attempt_start,
                             "found_candidate": found})
        if found:
            break
    elapsed = time.perf_counter() - total_start
    # This check is outside the elapsed search interval.
    verified = verify_witness(word, candidate_bands) if candidate_bands is not None else None
    status = ("witness" if verified else
              "invalid_witness" if candidate_bands is not None else "budget_exhausted")
    return {"case_id": case["case_id"], "status": status,
            "success": status == "witness", "elapsed_seconds": elapsed,
            "actions_taken": sum(row["actions_taken"] for row in attempt_rows),
            "attempts_completed": len(attempt_rows), "attempts": attempt_rows,
            "witness": candidate_bands, "witness_verified": verified,
            "length": case["length"], "exponent_sum": case["exponent_sum"],
            "negative_crossings": case["negative_crossings"],
            "negative_bucket": case["negative_bucket"]}


def _summary(rows):
    def describe(items):
        successes = [item for item in items if item["success"]]
        elapsed = [item["elapsed_seconds"] for item in items]
        success_elapsed = [item["elapsed_seconds"] for item in successes]
        return {"cases": len(items), "witnesses": len(successes),
                "success_rate": len(successes) / len(items) if items else None,
                "invalid_witnesses": sum(item["status"] == "invalid_witness" for item in items),
                "budget_exhausted": sum(item["status"] == "budget_exhausted" for item in items),
                "median_elapsed_seconds_all": statistics.median(elapsed) if elapsed else None,
                "median_elapsed_seconds_successes": statistics.median(success_elapsed) if success_elapsed else None,
                "elapsed_seconds_all": sorted(elapsed),
                "elapsed_seconds_successes": sorted(success_elapsed),
                "mean_actions_all": statistics.mean(item["actions_taken"] for item in items) if items else None}
    by_bucket = {bucket: describe([row for row in rows if row["negative_bucket"] == bucket])
                 for bucket in BUCKETS}
    by_length = {str(length): describe([row for row in rows if row["length"] == length])
                 for length in sorted({row["length"] for row in rows})}
    by_exponent = {str(exponent): describe([row for row in rows if row["exponent_sum"] == exponent])
                   for exponent in sorted({row["exponent_sum"] for row in rows})}
    by_negative_count = {str(count): describe([row for row in rows if row["negative_crossings"] == count])
                         for count in sorted({row["negative_crossings"] for row in rows})}
    return {"overall": describe(rows), "by_negative_crossings": by_bucket,
            "by_exact_negative_crossing_count": by_negative_count,
            "by_word_length": by_length, "by_exponent_sum": by_exponent,
            "timing_note": "Model time includes environment setup, reset, policy sampling and steps; checkpoint loading, corpus generation and post-run witness checks are excluded."}


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _checkpoint_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_csv_row(case, private_case, result):
    """Present the input and both positive-band decompositions side by side."""
    return {
        "case_id": case["case_id"],
        "model_input_word": json.dumps(case["word"]),
        "known_positive_band_decomposition": json.dumps(private_case["hidden_positive_witness"]),
        "model_found_decomposition": json.dumps(result["witness"]) if result["witness"] is not None else "",
        "status": result["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "actions_taken": result["actions_taken"],
        "length": case["length"],
        "exponent_sum": case["exponent_sum"],
        "negative_crossings": case["negative_crossings"],
        "inserted_cancelling_pairs": private_case["inserted_cancelling_pairs"],
    }


def run_pilot(config_path, *, runs_root="runs", generate_only=False):
    config = load_config(config_path)
    required = {"name", "model_path", "generation_seed", "policy_seed",
                "cases", "max_word_length", "max_num_bands", "attempts", "max_actions"}
    if set(config) != required:
        raise ValueError(f"Config fields must be exactly {sorted(required)}")
    if config["cases"] != 60 or config["max_word_length"] > 24:
        raise ValueError("This pilot requires 60 cases and at most 24 crossings")
    if config["max_num_bands"] != 30 or config["attempts"] != 10 or config["max_actions"] != 150:
        raise ValueError("This pilot requires 30 band slots, 10 attempts and 150 actions")
    for key in ("generation_seed", "policy_seed"):
        if type(config[key]) is not int or not 0 <= config[key] < 2**32:
            raise ValueError(f"{key} must be an integer in [0, 2**32)")
    if not generate_only and not Path(config["model_path"]).is_file():
        raise FileNotFoundError(config["model_path"])
    run = initialize_run(config, runs_root=runs_root)
    print(f"Run directory: {run}", flush=True)
    try:
        cases, provenance = generate_cases(config["generation_seed"], config["cases"],
                                           config["max_word_length"])
        _write_json(run / "cases_public.json", cases)
        _write_json(run / "cases_private.json", provenance)
        with (run / "cases_review.csv").open("w", newline="", encoding="utf-8") as output:
            fields = ("case_id", "model_input_word", "known_positive_band_decomposition",
                      "length", "exponent_sum", "negative_crossings", "inserted_cancelling_pairs")
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for case, private_case in zip(cases, provenance):
                writer.writerow({
                    "case_id": case["case_id"],
                    "model_input_word": json.dumps(case["word"]),
                    "known_positive_band_decomposition": json.dumps(
                        private_case["hidden_positive_witness"]),
                    "length": case["length"],
                    "exponent_sum": case["exponent_sum"],
                    "negative_crossings": case["negative_crossings"],
                    "inserted_cancelling_pairs": private_case["inserted_cancelling_pairs"],
                })
        _write_json(run / "qp3_input.json", {"cases": [
            {"case_id": case["case_id"], "word": case["word"]} for case in cases]})
        if generate_only:
            update_run_status(run, "corpus_ready")
            return run
        load_start = time.perf_counter()
        model = _load_policy(config["model_path"], config["max_num_bands"])
        model_load_seconds = time.perf_counter() - load_start
        _write_json(run / "runtime_metadata.json", {
            "python": sys.version, "pytorch": torch.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "device": "cpu",
            "model_load_seconds": model_load_seconds,
            "checkpoint_sha256": _checkpoint_hash(config["model_path"]),
        })
        results = []
        with ((run / "model_results.jsonl").open("w", encoding="utf-8") as output,
              (run / "results.csv").open("w", newline="", encoding="utf-8") as csv_output):
            writer = csv.DictWriter(csv_output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for index, case in enumerate(cases, 1):
                result = search_one(case, model, max_num_bands=config["max_num_bands"],
                                    attempts=config["attempts"], max_actions=config["max_actions"],
                                    policy_seed=config["policy_seed"])
                results.append(result)
                output.write(json.dumps(result) + "\n")
                output.flush()
                writer.writerow(result_csv_row(case, provenance[index - 1], result))
                csv_output.flush()
                print(f"{index}/{len(cases)} {case['case_id']}: {result['status']} "
                      f"in {result['elapsed_seconds']:.3f}s", flush=True)
        _write_json(run / "model_summary.json", _summary(results))
        update_run_status(run, "completed")
    except BaseException as error:
        update_run_status(run, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=f"{type(error).__name__}: {error}")
        raise
    return run


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qp3_pilot.json")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    print(run_pilot(args.config, runs_root=args.runs_root,
                    generate_only=args.generate_only))


if __name__ == "__main__":
    main_cli()
