"""Checks for selectable observations and unchanged cancellation behavior."""

import unittest

import numpy as np

from band_env import BandEnv


class ObservationTests(unittest.TestCase):
    def test_default_remains_matrix_encoding(self):
        env = BandEnv(band_decomposition=[1], braid_index=3,
                      max_num_bands=4, train_type="deterministic")
        try:
            observation, _ = env.reset()
            self.assertEqual(env.observation_type, "lk_matrix")
            self.assertEqual(observation.shape, (36,))
            np.testing.assert_array_equal(observation, env.get_state_lk())
        finally:
            env.close()

    def test_one_hot_preserves_crossing_and_band_order(self):
        env = BandEnv(band_decomposition=[[1, -2], [-1]], braid_index=3,
                      max_num_bands=4, train_type="deterministic",
                      observation_type="one_hot")
        try:
            observation, _ = env.reset()
            slots = observation.reshape(4, 13, 4)
            self.assertEqual(observation.shape, (208,))
            self.assertEqual(env.observation_space.shape, (208,))
            np.testing.assert_array_equal(slots[0, 0], [0, 0, 1, 0])
            np.testing.assert_array_equal(slots[0, 1], [1, 0, 0, 0])
            np.testing.assert_array_equal(slots[1, 0], [0, 1, 0, 0])
            self.assertEqual(np.count_nonzero(slots), 3)
        finally:
            env.close()

    def test_cancellation_uses_lk_in_both_observation_modes(self):
        for observation_type, expected_size in (("lk_matrix", 36), ("one_hot", 208)):
            with self.subTest(observation_type=observation_type):
                env = BandEnv(band_decomposition=[[1, 2], [-2, -1], [1]],
                              braid_index=3, max_num_bands=4,
                              train_type="deterministic",
                              observation_type=observation_type)
                try:
                    before, _ = env.reset()
                    self.assertEqual(before.shape, (expected_size,))
                    self.assertEqual(env.get_del_list(), [0])
                    after, _, _, _, _ = env.step(26)
                    self.assertEqual(after.shape, before.shape)
                    self.assertEqual(env.band_decomposition, [[1]])
                finally:
                    env.close()

    def test_one_hot_rejects_band_longer_than_slot(self):
        with self.assertRaisesRegex(ValueError, "max_band_len=13"):
            BandEnv(band_decomposition=[[1] * 14], braid_index=3,
                    max_num_bands=2, train_type="deterministic",
                    observation_type="one_hot")


if __name__ == "__main__":
    unittest.main()
