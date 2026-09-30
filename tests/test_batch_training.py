"""Small CPU checks for result persistence, resume, and best-length tracking.

Run: ../.knotenv/bin/python -m unittest tests.test_batch_training -v
All outputs are temporary; no research artifacts are touched.
"""

import contextlib
import csv
from functools import wraps
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import batch_training as batch
import main


class BatchTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="band_batch_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "braids.csv"
        with self.dataset.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["Braid index", "Braid ranks", "Braid word"])
            writer.writerow([3, 1, "{1, -1, 2, -2, 1}"])
            writer.writerow([4, 2, "{1, 2, -1, -2}"])
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({
            "data_path": str(self.dataset), "seed": 7,
            "epochs": 1, "env_samples": 2, "max_actions": 2,
            "max_num_bands": 8, "batch_size": 4, "policy_epochs": 1,
        }), encoding="utf-8")
        self.device_patch = patch.object(main, "device", "cpu")
        self.device_patch.start()
        self.addCleanup(self.device_patch.stop)
        save_patch = patch.object(main.torch, "save", side_effect=AssertionError("Unexpected model save"))
        save_patch.start()
        self.addCleanup(save_patch.stop)

    def _read_results(self, run):
        with (run / "results.csv").open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))

    def test_interruption_preserves_completed_braid_and_resume_skips_it(self):
        trainer = main.ppo_single_braid
        calls = []

        @wraps(trainer)
        def interrupt_second(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise KeyboardInterrupt("simulated interruption")
            return trainer(**kwargs)

        with contextlib.redirect_stdout(io.StringIO()), patch.object(
                main, "ppo_single_braid", interrupt_second):
            with self.assertRaises(KeyboardInterrupt):
                batch.run_batch_experiment(self.config, runs_root=self.root / "runs")
        run = next((self.root / "runs").iterdir())
        saved = self._read_results(run)
        self.assertEqual([row["braid_id"] for row in saved], ["0"])
        self.assertEqual(json.loads((run / "run_info.json").read_text())["status"], "interrupted")
        self.assertIsNone(calls[0]["save_path"])
        self.assertIsNone(calls[0]["report_path"])
        self.assertFalse(calls[0]["collect_logs"])
        # Resume is independent of the original dataset path.
        self.dataset.rename(self.root / "original_dataset_moved.csv")
        resumed_calls = []

        @wraps(trainer)
        def resumed(**kwargs):
            resumed_calls.append(kwargs)
            return trainer(**kwargs)

        with contextlib.redirect_stdout(io.StringIO()), patch.object(
                main, "ppo_single_braid", resumed):
            batch.run_batch_experiment(resume=run)
        self.assertEqual(len(resumed_calls), 1)
        self.assertEqual(resumed_calls[0]["braid_index"], 4)
        rows = self._read_results(run)
        self.assertEqual(rows[0], saved[0])
        self.assertEqual([row["seed"] for row in rows], ["7", "8"])
        self.assertEqual({path.name for path in run.iterdir()},
                         {"config.json", "run_info.json", "dataset.csv", "results.csv", "plot.png"})
        status = json.loads((run / "run_info.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertNotIn("error", status)

        @wraps(trainer)
        def finished(**kwargs):
            self.fail("A completed braid was retrained")

        with contextlib.redirect_stdout(io.StringIO()), patch.object(
                main, "ppo_single_braid", finished):
            batch.run_batch_experiment(resume=run)

    def test_best_length_includes_intermediate_state_and_witness_replays(self):
        # At capacity 8 in B_3, action 50 cancels the first pair and 14 creates a pair.
        actions = iter([50, 14])
        get_action = main.get_action_ppo

        def forced_action(network, state):
            _, distribution = get_action(network, state)
            return next(actions), distribution

        with patch.object(main, "get_action_ppo", forced_action):
            training = main.ppo_single_braid(
                [1, -1, 2, -2, 1], 3, epochs=1, env_samples=1,
                max_actions=2, max_num_bands=8, policy_epochs=1,
                save_path=None, report_path=None, collect_logs=False, show_progress=False,
            )
        self.assertEqual(training[3], [])
        report = training[4]
        self.assertEqual(report["initial_length"], 5)
        self.assertEqual(report["best_length"], 3)
        self.assertEqual(report["best_action_sequence"], [50])
        env = main.BandEnv([1, -1, 2, -2, 1], braid_index=3,
                           max_num_bands=8, train_type="deterministic")
        try:
            env.reset()
            for action in report["best_action_sequence"]:
                env.step(action)
            self.assertEqual(env.band_decomposition, report["best_decomposition"])
            env.step(14)
            self.assertEqual(len(env.band_decomposition), 5)
        finally:
            env.close()

    def test_failed_atomic_replace_preserves_previous_csv(self):
        path = self.root / "results.csv"
        batch._save_results([], path)
        previous_bytes = path.read_bytes()
        with patch.object(batch.os, "replace", side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                batch._save_results([dict.fromkeys(batch.RESULT_FIELDS, "test")], path)
        self.assertEqual(path.read_bytes(), previous_bytes)
        self.assertEqual(list(self.root.glob(".results_*.tmp")), [])

    def test_capacity_error_occurs_before_creating_a_run(self):
        config = json.loads(self.config.read_text())
        config["max_num_bands"] = 4
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exceeds max_num_bands"):
            batch.run_batch_experiment(self.config, runs_root=self.root / "runs")
        self.assertFalse((self.root / "runs").exists())

    def test_partial_results_plot_without_training(self):
        @wraps(main.ppo_single_braid)
        def interrupted(**kwargs):
            if kwargs["braid_index"] == 4:
                raise RuntimeError("simulated failure")
            return trainer(**kwargs)

        trainer = main.ppo_single_braid
        with contextlib.redirect_stdout(io.StringIO()), patch.object(main, "ppo_single_braid", interrupted):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                batch.run_batch_experiment(self.config, runs_root=self.root / "runs")
        run = next((self.root / "runs").iterdir())
        self.assertEqual(batch.plot_batch_results(run), run / "plot.png")
        self.assertGreater((run / "plot.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
