"""Witness and restart checks for the braid DP comparison."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from compare_dp_rl import compare
from dp_braid import UpperBoundResult, braid_upper_bound, validate_witness


class BraidDPTests(unittest.TestCase):
    def test_identity_and_conjugate_witnesses(self):
        cases = [
            (3, [], 0),
            (3, [1, -1], 0),
            (3, [1, 2, 1, -2, -1, -2], 0),  # Artin relation.
            (3, [2, 1, -2], 1),
            (4, [3, 2, 1, 2, -1, -2, -3], 1),  # Thesis code example.
            (4, [1, 3, -1], 1),  # Commuting generators.
            (4, [2, 1, 2, -1, -2, 3, 2, -3], 2),  # Thesis code example.
            (7, [4, 6, -2, 6, -4, 6, 2], 3),  # Thesis commuting example.
        ]
        for braid_index, word, expected in cases:
            with self.subTest(word=word):
                result = braid_upper_bound(word, braid_index)
                self.assertEqual(result.upper_bound, expected)
                self.assertEqual(len(result.decomposition), expected)
                self.assertTrue(all(isinstance(band, list)
                                    for band in result.decomposition))
                self.assertTrue(validate_witness(word, braid_index, result))

    def test_invalid_certificate_is_rejected(self):
        bad = UpperBoundResult(1, [[1, 1]], [([], 1)], [1, 1])
        with self.assertRaisesRegex(ValueError, "non-certified band"):
            validate_witness([1, 1], 3, bad)

    def test_comparison_writes_list_of_lists_and_resumes(self):
        with tempfile.TemporaryDirectory(prefix="band_dp_test_") as directory:
            root = Path(directory)
            rl_run = root / "rl_run"
            rl_run.mkdir()
            dataset = rl_run / "dataset.csv"
            with dataset.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(["Braid index", "Braid ranks", "Braid word"])
                writer.writerow([3, 1, "{1, -1, 2, -2, 1}"])
                writer.writerow([3, 0, "{1, 2, 1, -2, -1, -2}"])
            (rl_run / "config.json").write_text(json.dumps({
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            with (rl_run / "results.csv").open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=(
                    "braid_id", "braid_index", "true_rank", "initial_length",
                    "best_length", "best_decomposition"))
                writer.writeheader()
                writer.writerow({"braid_id": 0, "braid_index": 3, "true_rank": 1,
                                 "initial_length": 5, "best_length": 1,
                                 "best_decomposition": "[[1]]"})
                writer.writerow({"braid_id": 1, "braid_index": 3, "true_rank": 0,
                                 "initial_length": 6, "best_length": 0,
                                 "best_decomposition": "[]"})
            output_dir = root / "comparison"
            partial = compare(rl_run, output_dir, braid_ids=[0], timeout_seconds=10)
            self.assertEqual(partial["rows_saved"], 1)
            complete = compare(rl_run, output_dir, timeout_seconds=10)
            self.assertEqual(complete["paired_completed"], 2)
            with (output_dir / "results.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[1]["dp_original_decomposition"]), [])
            self.assertTrue(all(isinstance(band, list) for row in rows
                                for band in json.loads(row["dp_after_rl_decomposition"])))


if __name__ == "__main__":
    unittest.main()
