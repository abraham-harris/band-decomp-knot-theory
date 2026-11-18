"""KnotInfo quasipositivity benchmark: prepare, local DP, remote RL, summary.

All experiment output lives in one runs/ directory. The cleaned KnotInfo corpus
is kept separately in data/knotinfo/ and is hashed before every resumed stage.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from urllib.request import urlopen
import zipfile

from dp_braid import braid_upper_bound, normal_form_key
from experiment_utils import initialize_run, update_run_status


SOURCE_URL = "https://raw.githubusercontent.com/knotinfo-org/assets/main/knotinfo_data_complete.xls.zip"
DEFAULT_DATA_DIR = Path("data/knotinfo")
SOURCE_NAME = "knotinfo_data_complete.xls.zip"


class CaseTimeout(Exception):
    pass


def _timeout(_signum, _frame):
    raise CaseTimeout("case time limit exceeded")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=".qp_", delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _download(path):
    with urlopen(SOURCE_URL, timeout=60) as response:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=".knotinfo_", delete=False) as output:
            temporary = Path(output.name)
            try:
                shutil.copyfileobj(response, output)
                output.flush()
                os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()


def _parse_case(row):
    name = str(row["name"]).strip()
    braid_index = int(str(row["braid_index"]).strip())
    word = json.loads(str(row["braid_notation"]).strip())
    if not name or braid_index < 2 or not isinstance(word, list) or not word:
        raise ValueError("missing name, braid index, or nonempty braid word")
    if any(type(x) is not int or x == 0 or abs(x) >= braid_index for x in word):
        raise ValueError("braid word has a generator outside its strand count")
    return {"name": name, "label": "Y", "mirror_witness": str(
        row["quasipositive_braid_notation"]).strip().startswith("M"),
        "braid_index": braid_index, "word": word}


def _clean_rows(rows):
    cases, excluded = [], []
    names = set()
    for row in rows:
        if str(row.get("quasipositive", "")).strip() != "Y":
            continue
        try:
            case = _parse_case(row)
            if case["name"] in names:
                raise ValueError("duplicate knot name")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            excluded.append({"name": str(row.get("name", "")), "reason": str(error)})
            continue
        names.add(case["name"])
        cases.append(case)
    cases.sort(key=lambda case: case["name"])
    return cases, excluded


def _read_xls(source_path):
    try:
        import xlrd
    except ImportError as error:
        raise RuntimeError("KnotInfo preparation requires xlrd==2.0.2; install it in .knotenv") from error
    with zipfile.ZipFile(source_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".xls")
                   and not name.startswith("__MACOSX/")]
        if len(members) != 1:
            raise ValueError("expected one .xls file in the KnotInfo archive")
        workbook = xlrd.open_workbook(file_contents=archive.read(members[0]),
                                      on_demand=True)
    sheet = workbook.sheet_by_index(0)
    headers = [str(value).strip() for value in sheet.row_values(0)]
    required = {"name", "quasipositive", "quasipositive_braid_notation",
                "braid_index", "braid_notation"}
    if not required.issubset(headers):
        raise ValueError(f"KnotInfo columns missing: {sorted(required - set(headers))}")
    indices = {name: headers.index(name) for name in required}
    # Row 1 is the display heading; database values begin at row 2.
    for row_number in range(2, sheet.nrows):
        yield {name: sheet.cell_value(row_number, index)
               for name, index in indices.items()}


def prepare(data_dir=DEFAULT_DATA_DIR, source_zip=None, refresh=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / SOURCE_NAME
    if source_zip is not None:
        supplied = Path(source_zip)
        if not supplied.is_file():
            raise FileNotFoundError(supplied)
        if supplied.resolve() != source.resolve():
            _atomic_bytes(source, supplied.read_bytes())
    elif not source.exists() or refresh:
        _download(source)
    if not zipfile.is_zipfile(source):
        raise ValueError(f"Not a ZIP archive: {source}")
    source_rows = list(_read_xls(source))
    cases, excluded = _clean_rows(source_rows)
    if not cases:
        raise ValueError("No labeled knots with usable braid words were found")
    cases_path = data_dir / "cases.json"
    _write_json(cases_path, cases)
    _write_json(data_dir / "excluded.json", excluded)
    manifest = {"source_url": SOURCE_URL, "source_sha256": _sha256(source),
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "cases_sha256": _sha256(cases_path), "labeled_count": len(cases) + len(excluded),
                "usable_count": len(cases), "excluded_count": len(excluded),
                "database_rows": len(source_rows),
                "database_qp_yes": sum(str(row["quasipositive"]).strip() == "Y"
                                       for row in source_rows),
                "database_qp_no": sum(str(row["quasipositive"]).strip() == "N"
                                      for row in source_rows),
                "database_qp_unlabeled": sum(not str(row["quasipositive"]).strip()
                                             for row in source_rows)}
    _write_json(data_dir / "manifest.json", manifest)
    return manifest


def _cases(data_dir):
    data_dir = Path(data_dir)
    manifest = _load_json(data_dir / "manifest.json")
    path = data_dir / "cases.json"
    if _sha256(path) != manifest["cases_sha256"]:
        raise ValueError("cleaned KnotInfo corpus differs from its manifest")
    cases = _load_json(path)
    if len(cases) != manifest["usable_count"] or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("invalid cleaned KnotInfo corpus")
    return cases, manifest


def _run_config(run_dir, data_dir):
    config = _load_json(run_dir / "config.json")
    cases, manifest = _cases(data_dir)
    if config["cases_sha256"] != manifest["cases_sha256"]:
        raise ValueError("run was created for a different KnotInfo corpus")
    return cases, config


def _results(path):
    return _load_json(path) if path.exists() else {}


def _run_case(word, braid_index, seconds):
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout)
    try:
        if seconds:
            signal.setitimer(signal.ITIMER_REAL, seconds)
        result = braid_upper_bound(word, braid_index)
        exponent = _exponent(word)
        status = ("positive" if exponent > 0 else "negative" if exponent < 0
                  else "identity") if result.upper_bound == abs(exponent) else "unresolved"
        return {"status": status, "upper_bound": result.upper_bound,
                "exponent_sum": exponent, "decomposition": result.decomposition,
                "certificates": result.certificates}
    except CaseTimeout:
        return {"status": "timeout"}
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _timed_dp(word, braid_index, seconds):
    start = time.monotonic()
    result = _run_case(word, braid_index, seconds)
    result["elapsed_seconds"] = round(time.monotonic() - start, 6)
    return result


def run_dp(data_dir=DEFAULT_DATA_DIR, run_dir=None, *, timeout_seconds=120,
           runs_root="runs", retry_failed=False, max_cases=None):
    cases, manifest = _cases(data_dir)
    if run_dir is None:
        run_dir = initialize_run({"name": "knotinfo_qp", "cases_sha256": manifest["cases_sha256"],
                                  "dp_timeout_seconds": timeout_seconds}, runs_root=runs_root)
    else:
        run_dir = Path(run_dir)
        _run_config(run_dir, data_dir)
    rows_path = run_dir / "dp_results.json"
    rows = _results(rows_path)
    for case in cases[:max_cases]:
        name = case["name"]
        if name in rows and not (retry_failed and rows[name]["status"] in ("error", "timeout")):
            continue
        rows[name] = _timed_dp(case["word"], case["braid_index"], timeout_seconds)
        _write_json(rows_path, rows)
        print(f"DP {len(rows)}/{len(cases)} {name}: {rows[name]['status']}", flush=True)
    update_run_status(run_dir, "dp_completed" if len(rows) == len(cases) else "dp_partial")
    return run_dir


def _flat(bands):
    return [crossing for band in bands for crossing in band]


def _exponent(word):
    return sum(1 if crossing > 0 else -1 for crossing in word)


def verify_rl_candidate(case, bands):
    """Use only for decompositions produced by BandEnv's signed-band moves."""
    exponent = _exponent(case["word"])
    if not isinstance(bands, list) or len(bands) < abs(exponent):
        return False
    if any(not isinstance(band, list) or not band or
           any(type(x) is not int or x == 0 or abs(x) >= case["braid_index"]
               for x in band) for band in bands):
        return False
    return normal_form_key(case["braid_index"], _flat(bands)) == normal_form_key(
        case["braid_index"], case["word"])


def _witness_sign(case, bands):
    if not verify_rl_candidate(case, bands) or len(bands) != abs(_exponent(case["word"])):
        return None
    sign = 1 if _exponent(case["word"]) > 0 else -1 if _exponent(case["word"]) < 0 else 0
    if sign and all(_exponent(band) == sign for band in bands):
        return "positive" if sign == 1 else "negative"
    return "identity" if sign == 0 and not bands else None


def run_rl(run_dir, data_dir=DEFAULT_DATA_DIR, *, max_cases=10, epochs=50,
           env_samples=10, max_actions=150, case_timeout_seconds=3600,
           dp_timeout_seconds=120, seed=1729):
    run_dir = Path(run_dir)
    cases, _ = _run_config(run_dir, data_dir)
    by_name = {case["name"]: case for case in cases}
    dp = _results(run_dir / "dp_results.json")
    if len(dp) != len(cases):
        raise ValueError("finish the local DP pass before selecting RL cases")
    selection_path = run_dir / "rl_selection.json"
    if selection_path.exists():
        selected = _load_json(selection_path)
    else:
        eligible = [case for case in cases if dp[case["name"]]["status"] not in
                    ("positive", "negative", "identity") and _exponent(case["word"]) != 0]
        eligible.sort(key=lambda case: (len(case["word"]), case["name"]))
        selected = [case["name"] for case in eligible[:max_cases]]
        _write_json(run_dir / "rl_config.json", {"epochs": epochs,
                    "env_samples": env_samples, "max_actions": max_actions,
                    "case_timeout_seconds": case_timeout_seconds,
                    "dp_timeout_seconds": dp_timeout_seconds, "seed": seed})
        _write_json(selection_path, selected)
    if _load_json(run_dir / "rl_config.json") != {"epochs": epochs,
            "env_samples": env_samples, "max_actions": max_actions,
            "case_timeout_seconds": case_timeout_seconds,
            "dp_timeout_seconds": dp_timeout_seconds, "seed": seed}:
        raise ValueError("RL settings changed; start a new run")
    rows_path = run_dir / "rl_results.json"
    rows = _results(rows_path)
    from main import ppo_single_braid
    import numpy as np
    import random
    import torch
    for position, name in enumerate(selected):
        if name in rows:
            continue
        case = by_name[name]
        case_seed = (seed + position) % 2**32
        random.seed(case_seed)
        np.random.seed(case_seed)
        torch.manual_seed(case_seed)
        start = time.monotonic()
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout)
        try:
            if case_timeout_seconds:
                signal.setitimer(signal.ITIMER_REAL, case_timeout_seconds)
            training = ppo_single_braid(
                band_decomposition=case["word"], braid_index=case["braid_index"],
                epochs=epochs, env_samples=env_samples, max_actions=max_actions,
                max_num_bands=max(16, len(case["word"]) + 8),
                save_path=None, report_path=None, collect_logs=False,
                show_progress=False, observation_type="lk_matrix",
                candidate_validator=lambda bands: verify_rl_candidate(case, bands))
            report = training[4]
            best = report["best_decomposition"]
            witness = _witness_sign(case, best)
            row = {"status": "witness" if witness else "unresolved",
                   "witness_sign": witness, "best_length": len(best),
                   "best_decomposition": best,
                   "best_action_sequence": report["best_action_sequence"],
                   "seed": case_seed, "epochs": epochs,
                   "elapsed_seconds": round(time.monotonic() - start, 6)}
        except CaseTimeout:
            row = {"status": "timeout", "seed": case_seed,
                   "elapsed_seconds": round(time.monotonic() - start, 6)}
        except Exception as error:
            row = {"status": "error", "error": f"{type(error).__name__}: {error}",
                   "seed": case_seed, "elapsed_seconds": round(time.monotonic() - start, 6)}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        if "best_decomposition" in row:
            row["dp_after_rl"] = _timed_dp(_flat(row["best_decomposition"]),
                                           case["braid_index"], dp_timeout_seconds)
        rows[name] = row
        _write_json(rows_path, rows)
        print(f"RL {len(rows)}/{len(selected)} {name}: {row['status']}", flush=True)
    update_run_status(run_dir, "rl_completed" if len(rows) == len(selected) else "rl_partial")
    return run_dir


def summarize(run_dir, data_dir=DEFAULT_DATA_DIR):
    run_dir = Path(run_dir)
    cases, _ = _run_config(run_dir, data_dir)
    manifest = _load_json(Path(data_dir) / "manifest.json")
    dp = _results(run_dir / "dp_results.json")
    rl = _results(run_dir / "rl_results.json")
    fieldnames = ("name", "label", "mirror_witness", "braid_index", "word_length",
                  "exponent_sum", "dp_status", "dp_upper_bound", "rl_status",
                  "rl_best_length", "rl_witness_sign", "dp_after_rl_status",
                  "dp_after_rl_upper_bound")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for case in cases:
        d = dp.get(case["name"], {})
        r = rl.get(case["name"], {})
        h = r.get("dp_after_rl", {})
        writer.writerow({"name": case["name"], "label": case["label"],
                         "mirror_witness": case["mirror_witness"],
                         "braid_index": case["braid_index"],
                         "word_length": len(case["word"]),
                         "exponent_sum": _exponent(case["word"]),
                         "dp_status": d.get("status", "not_run"),
                         "dp_upper_bound": d.get("upper_bound", ""),
                         "rl_status": r.get("status", "not_run"),
                         "rl_best_length": r.get("best_length", ""),
                         "rl_witness_sign": r.get("witness_sign", ""),
                         "dp_after_rl_status": h.get("status", "not_run"),
                         "dp_after_rl_upper_bound": h.get("upper_bound", "")})
    _atomic_bytes(run_dir / "summary.csv", output.getvalue().encode())
    summary = {"labeled_count": manifest["labeled_count"],
               "usable_braid_count": len(cases),
               "excluded_count": manifest["excluded_count"],
               "dp_completed": len(dp), "rl_completed": len(rl)}
    for stage, rows in (("dp", dp), ("rl", rl)):
        statuses = {}
        for row in rows.values():
            status = row["status"]
            statuses[status] = statuses.get(status, 0) + 1
        summary[stage + "_statuses"] = statuses
    witness_statuses = {"positive", "negative", "identity"}
    post_rl_rows = {name: row for name, row in rl.items() if "dp_after_rl" in row}
    raw_post_statuses = {}
    attributed_post_statuses = {}
    dp_confirmed_rl = 0
    for row in post_rl_rows.values():
        dp_status = row["dp_after_rl"]["status"]
        raw_post_statuses[dp_status] = raw_post_statuses.get(dp_status, 0) + 1
        rl_found = row.get("witness_sign") in witness_statuses
        dp_found = dp_status in witness_statuses
        dp_confirmed_rl += rl_found and dp_found
        attributed_status = ("resolved_by_rl" if rl_found else
                             "resolved_by_dp_after_rl" if dp_found else dp_status)
        attributed_post_statuses[attributed_status] = (
            attributed_post_statuses.get(attributed_status, 0) + 1)
    summary["dp_after_rl_statuses"] = attributed_post_statuses
    summary["dp_after_rl_raw_statuses"] = raw_post_statuses
    summary["dp_confirmed_rl_witnesses"] = dp_confirmed_rl
    dp_witnesses = sum(row["status"] in witness_statuses for row in dp.values())
    hybrid_new = sum(
        dp.get(name, {}).get("status") not in witness_statuses and (
            row.get("witness_sign") in witness_statuses or
            row.get("dp_after_rl", {}).get("status") in witness_statuses)
        for name, row in rl.items())
    summary.update(dp_witnesses=dp_witnesses,
                   dp_success_rate_usable=dp_witnesses / len(cases) if cases else None,
                   dp_success_rate_labeled=dp_witnesses / manifest["labeled_count"]
                   if manifest["labeled_count"] else None,
                   hybrid_new_witnesses=hybrid_new)
    _write_json(run_dir / "summary.json", summary)
    return summary


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    prepare_cmd.add_argument("--source-zip")
    prepare_cmd.add_argument("--refresh", action="store_true")
    dp_cmd = sub.add_parser("dp")
    dp_cmd.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    dp_cmd.add_argument("--run-dir")
    dp_cmd.add_argument("--runs-root", default="runs")
    dp_cmd.add_argument("--timeout-seconds", type=float, default=120)
    dp_cmd.add_argument("--retry-failed", action="store_true")
    dp_cmd.add_argument("--max-cases", type=int)
    rl_cmd = sub.add_parser("rl")
    rl_cmd.add_argument("--run-dir", required=True)
    rl_cmd.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    rl_cmd.add_argument("--max-cases", type=int, default=10)
    rl_cmd.add_argument("--epochs", type=int, default=50)
    rl_cmd.add_argument("--env-samples", type=int, default=10)
    rl_cmd.add_argument("--max-actions", type=int, default=150)
    rl_cmd.add_argument("--case-timeout-seconds", type=float, default=3600)
    rl_cmd.add_argument("--dp-timeout-seconds", type=float, default=120)
    rl_cmd.add_argument("--seed", type=int, default=1729)
    sum_cmd = sub.add_parser("summary")
    sum_cmd.add_argument("--run-dir", required=True)
    sum_cmd.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.data_dir, args.source_zip, args.refresh), indent=2))
    elif args.command == "dp":
        if args.timeout_seconds < 0 or args.max_cases is not None and args.max_cases < 1:
            parser.error("timeout must be nonnegative and max-cases positive")
        print(run_dp(args.data_dir, args.run_dir, timeout_seconds=args.timeout_seconds,
                     runs_root=args.runs_root, retry_failed=args.retry_failed,
                     max_cases=args.max_cases))
    elif args.command == "rl":
        if (min(args.max_cases, args.epochs, args.env_samples, args.max_actions) < 1
                or args.case_timeout_seconds < 0 or args.dp_timeout_seconds < 0
                or not 0 <= args.seed < 2**32):
            parser.error("invalid RL budget or seed")
        print(run_rl(args.run_dir, args.data_dir, max_cases=args.max_cases,
                     epochs=args.epochs, env_samples=args.env_samples,
                     max_actions=args.max_actions,
                     case_timeout_seconds=args.case_timeout_seconds,
                     dp_timeout_seconds=args.dp_timeout_seconds, seed=args.seed))
    else:
        print(json.dumps(summarize(args.run_dir, args.data_dir), indent=2))


if __name__ == "__main__":
    main_cli()
