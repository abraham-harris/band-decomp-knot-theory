"""Short corpus, certificate, and resume checks for the KnotInfo benchmark."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from qp_benchmark import (_clean_rows, _sha256, _witness_sign, run_dp,
                          run_rl, summarize, verify_rl_candidate)


def _corpus(root, cases):
    data = root / "data"
    data.mkdir()
    (data / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (data / "manifest.json").write_text(json.dumps({
        "cases_sha256": _sha256(data / "cases.json"),
        "usable_count": len(cases), "labeled_count": len(cases),
        "excluded_count": 0,
    }), encoding="utf-8")
    return data


class QpBenchmarkTests(unittest.TestCase):
    def test_clean_only_explicitly_labeled_knots(self):
        rows = [
            {"name": "a", "quasipositive": "Y", "braid_index": "3",
             "braid_notation": "[2,1,-2]", "quasipositive_braid_notation": "M[[1]]"},
            {"name": "b", "quasipositive": "N"},
            {"name": "c", "quasipositive": ""},
            {"name": "d", "quasipositive": "Y", "braid_index": "2",
             "braid_notation": "[2]", "quasipositive_braid_notation": ""},
        ]
        cases, excluded = _clean_rows(rows)
        self.assertEqual([case["name"] for case in cases], ["a"])
        self.assertTrue(cases[0]["mirror_witness"])
        self.assertEqual(excluded[0]["name"], "d")

    def test_rl_witness_checks_exact_braid_and_sign(self):
        positive = {"braid_index": 3, "word": [2, 1, -2]}
        negative = {"braid_index": 3, "word": [2, -1, -2]}
        self.assertEqual(_witness_sign(positive, [[2, 1, -2]]), "positive")
        self.assertEqual(_witness_sign(negative, [[2, -1, -2]]), "negative")
        self.assertFalse(verify_rl_candidate(positive, [[1]]))
        self.assertIsNone(_witness_sign(positive, [[-2, 1, 2]]))

    def test_dp_partial_resume_and_summary(self):
        with tempfile.TemporaryDirectory(prefix="qp_benchmark_test_") as directory:
            root = Path(directory)
            data = _corpus(root, [
                {"name": "a", "label": "Y", "mirror_witness": False,
                 "braid_index": 2, "word": [1, 1, 1]},
                {"name": "b", "label": "Y", "mirror_witness": True,
                 "braid_index": 2, "word": [-1, -1, -1]},
            ])
            run = run_dp(data, runs_root=root / "runs", max_cases=1)
            first = json.loads((run / "dp_results.json").read_text())
            self.assertEqual(set(first), {"a"})
            run_dp(data, run_dir=run)
            rows = json.loads((run / "dp_results.json").read_text())
            self.assertEqual(rows["a"], first["a"])
            self.assertEqual(rows["b"]["status"], "negative")
            result = summarize(run, data)
            self.assertEqual(result["dp_statuses"], {"positive": 1, "negative": 1})
            (data / "cases.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from its manifest"):
                run_dp(data, run_dir=run)

    def test_rl_saves_decomposition_without_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="qp_benchmark_test_") as directory:
            root = Path(directory)
            case = {"name": "candidate", "label": "Y", "mirror_witness": False,
                    "braid_index": 3, "word": [2, 1, -2]}
            data = _corpus(root, [case])
            run = run_dp(data, runs_root=root / "runs")
            (run / "dp_results.json").write_text(json.dumps({
                "candidate": {"status": "unresolved"}}), encoding="utf-8")

            def fake_trainer(**kwargs):
                self.assertIsNone(kwargs["save_path"])
                self.assertIsNone(kwargs["report_path"])
                self.assertTrue(kwargs["candidate_validator"]([[2, 1, -2]]))
                return (None, None, None, None, {
                    "best_decomposition": [[2, 1, -2]],
                    "best_action_sequence": [],
                })

            trainer = Mock(side_effect=fake_trainer)
            fake_main = types.ModuleType("main")
            fake_main.ppo_single_braid = trainer
            with patch.dict("sys.modules", {"main": fake_main}):
                run_rl(run, data, epochs=1, env_samples=1, max_actions=1)
                run_rl(run, data, epochs=1, env_samples=1, max_actions=1)
            self.assertEqual(trainer.call_count, 1)
            row = json.loads((run / "rl_results.json").read_text())["candidate"]
            self.assertEqual(row["witness_sign"], "positive")
            self.assertEqual(row["dp_after_rl"]["status"], "positive")
            self.assertFalse(any(run.rglob("*.pt")))


if __name__ == "__main__":
    unittest.main()
