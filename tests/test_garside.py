"""Normal-form identities and fixed-size Garside observation checks."""

import random
import unittest

import numpy as np

from band_env import BandEnv
from garside import left_normal_form


class GarsideTests(unittest.TestCase):
    def test_known_normal_forms_and_braid_relations(self):
        self.assertEqual(left_normal_form(4, []), (0, []))
        self.assertEqual(left_normal_form(4, [1, 2, 1]),
                         (0, [(2, 1, 0, 3)]))
        self.assertEqual(left_normal_form(4, [1, -2, 3, -1]),
                         (-1, [(1, 3, 2, 0), (0, 2, 3, 1)]))
        for n in range(3, 6):
            for index in range(1, n - 1):
                self.assertEqual(left_normal_form(n, [index, index + 1, index]),
                                 left_normal_form(n, [index + 1, index, index + 1]))
            for index in range(1, n - 2):
                self.assertEqual(left_normal_form(n, [index, index + 2]),
                                 left_normal_form(n, [index + 2, index]))

    def test_inverse_cancellation_and_left_weighting(self):
        rng = random.Random(42)
        for n in range(3, 6):
            for _ in range(100):
                word = [rng.choice((-1, 1))*rng.randrange(1, n)
                        for _ in range(rng.randrange(1, 12))]
                inverse = [-crossing for crossing in reversed(word)]
                self.assertEqual(left_normal_form(n, word + inverse), (0, []))
                _, factors = left_normal_form(n, word)
                identity = tuple(range(n))
                delta = tuple(range(n - 1, -1, -1))
                self.assertTrue(all(factor not in (identity, delta) for factor in factors))
                for left, right in zip(factors, factors[1:]):
                    positions = {value: position for position, value in enumerate(right)}
                    self.assertFalse(any(positions[index + 1] < positions[index]
                                         and left[index] < left[index + 1]
                                         for index in range(n - 1)))

    def test_band_slots_and_word_length(self):
        env = BandEnv(band_decomposition=[[1, 2, 1], [], [-2]], braid_index=4,
                      max_num_bands=4, train_type="deterministic",
                      observation_type="garside")
        try:
            observation, _ = env.reset()
            slots = observation.reshape(4, 3 + 13*4)
            self.assertEqual(observation.shape, (220,))
            self.assertEqual(env.observation_space.shape, (220,))
            self.assertEqual(observation.dtype, np.float32)
            np.testing.assert_allclose(slots[0, :7],
                                       [1, 3/13, 0, 2/3, 1/3, 0, 1])
            np.testing.assert_array_equal(slots[0, 7:], 0)
            self.assertEqual(slots[1, 0], 1)  # Existing empty band.
            np.testing.assert_array_equal(slots[1, 1:], 0)
            np.testing.assert_allclose(slots[2, :7],
                                       [1, 1/13, -1/13, 1, 1/3, 2/3, 0])
            np.testing.assert_array_equal(slots[3], 0)  # Unused band slot.
        finally:
            env.close()

    def test_equivalent_braids_keep_stored_word_length(self):
        env = BandEnv(band_decomposition=[[1], [1, -1, 1]], braid_index=3,
                      max_num_bands=2, train_type="deterministic",
                      observation_type="garside")
        try:
            slots = env.get_state().reshape(2, 3 + 13*3)
            np.testing.assert_array_equal(slots[0, 2:], slots[1, 2:])
            self.assertEqual(slots[0, 1], np.float32(1/13))
            self.assertEqual(slots[1, 1], np.float32(3/13))
        finally:
            env.close()

    def test_step_shape_and_lk_cancellation(self):
        env = BandEnv(band_decomposition=[[1, 2], [-2, -1], [1]],
                      braid_index=3, max_num_bands=4,
                      train_type="deterministic", observation_type="garside")
        try:
            before, _ = env.reset()
            self.assertEqual(env.get_del_list(), [0])
            after, _, _, _, _ = env.step(26)
            self.assertEqual(after.shape, before.shape)
            self.assertEqual(env.band_decomposition, [[1]])
        finally:
            env.close()

    def test_rejects_band_longer_than_slot(self):
        with self.assertRaisesRegex(ValueError, "max_band_len=13"):
            BandEnv(band_decomposition=[[1]*14], braid_index=3,
                    max_num_bands=2, train_type="deterministic",
                    observation_type="garside")


if __name__ == "__main__":
    unittest.main()
