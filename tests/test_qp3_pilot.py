"""Focused checks for the 3-braid witness benchmark."""

import json
import random
import unittest

import torch

from qp3_pilot import (_one_candidate, flatten, generate_cases, normal_form_key,
                       result_csv_row, search_one, validate_public_cases,
                       verify_witness)


class CancelFirstPair(torch.nn.Module):
    def forward(self, state):
        probabilities = torch.zeros((state.shape[0], 211), dtype=torch.float32)
        probabilities[:, 182] = 1
        return probabilities


class Qp3PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_generated_cases_are_distinct_and_have_hidden_positive_witnesses(self):
        cases, provenance = generate_cases(seed=12345, count=6)
        validate_public_cases(cases, expected_count=6)
        self.assertEqual(len({normal_form_key(case["word"]) for case in cases}), 6)
        self.assertTrue(any(move["move"] in ("left", "right") for private in provenance
                            for move in private["environment_moves"]))
        for case, private in zip(cases, provenance):
            self.assertEqual(case["case_id"], private["case_id"])
            self.assertEqual(normal_form_key(case["word"]),
                             normal_form_key(flatten(private["hidden_positive_witness"])))
            self.assertTrue(verify_witness(case["word"],
                                           private["hidden_positive_witness"]))
            self.assertIn(private["inserted_cancelling_pairs"], (0, 1, 2))
            self.assertEqual(private["inserted_cancelling_pairs"],
                             sum(move["move"] == "create"
                                 for move in private["environment_moves"]))

    def test_candidate_generation_varies_pair_count(self):
        candidates = [_one_candidate(random.Random(seed), 24) for seed in range(60)]
        counts = {candidate[1]["inserted_cancelling_pairs"]
                  for candidate in candidates if candidate is not None}
        self.assertEqual(counts, {0, 1, 2})

    def test_witness_verification_detects_changed_braid(self):
        input_word = [1, -2, 1, 2]
        self.assertFalse(verify_witness(input_word, [[1], [1]]))
        self.assertTrue(verify_witness([1, -1, 2], [[2]]))
        self.assertTrue(verify_witness([2, 1, -2], [[2, 1, -2]]))

    def test_search_stops_at_first_witness_and_records_actions(self):
        case = {"case_id": "qp3_000", "word": [1, -1, 1], "length": 3,
                "exponent_sum": 1, "negative_crossings": 1,
                "negative_bucket": "2-3"}
        result = search_one(case, CancelFirstPair(), attempts=2, max_actions=3)
        self.assertEqual(result["status"], "witness")
        self.assertEqual(result["witness"], [[1]])
        self.assertEqual(result["actions_taken"], 1)
        self.assertEqual(result["attempts_completed"], 1)
        self.assertEqual(result["attempts"][0]["actions"], [182])

    def test_csv_row_shows_input_known_and_found_decompositions(self):
        case = {"case_id": "qp3_000", "word": [2, 1, -2], "length": 3,
                "exponent_sum": 1, "negative_crossings": 1}
        private = {"hidden_positive_witness": [[2, 1, -2]],
                   "inserted_cancelling_pairs": 1}
        result = {"witness": [[2, 1, -2]], "status": "witness",
                  "elapsed_seconds": 0.2, "actions_taken": 2}
        row = result_csv_row(case, private, result)
        self.assertEqual(json.loads(row["model_input_word"]), case["word"])
        self.assertEqual(json.loads(row["known_positive_band_decomposition"]),
                         private["hidden_positive_witness"])
        self.assertEqual(json.loads(row["model_found_decomposition"]), result["witness"])
        self.assertEqual(row["inserted_cancelling_pairs"], 1)
        result.update(witness=None, status="budget_exhausted")
        self.assertEqual(result_csv_row(case, private, result)["model_found_decomposition"], "")


if __name__ == "__main__":
    unittest.main()
