"""Braid-word environment whose costly move is deleting one crossing.

The state contains no bands.  A sequence ending at the empty word after D
deletions certifies a band factorization of length D: deleting u*sigma*v
changes the braid by the conjugate u*sigma*u^-1.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from generator import RandomBraid


class CrossingRemovalEnv(gym.Env):
    def __init__(self, *, braid_index, max_word_length, max_actions,
                 word=None, random_length_stdev=24, active_braid_index=None):
        super().__init__()
        if type(braid_index) is not int or braid_index < 2:
            raise ValueError("braid_index must be at least 2")
        if type(max_word_length) is not int or max_word_length < 1:
            raise ValueError("max_word_length must be positive")
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be positive")
        if word is None and braid_index < 3:
            raise ValueError("RandomBraid requires braid_index at least 3")
        if random_length_stdev <= 0:
            raise ValueError("random_length_stdev must be positive")
        self.braid_index = braid_index
        self.active_braid_index = braid_index if active_braid_index is None else active_braid_index
        if (type(self.active_braid_index) is not int or
                not 2 <= self.active_braid_index <= braid_index):
            raise ValueError("active_braid_index must be between 2 and braid_index")
        self.max_word_length = max_word_length
        self.max_actions = max_actions
        self.random_length_stdev = random_length_stdev
        self.initial_word = None if word is None else self._validate_word(word)
        # Keep the checkpoint's full generator vocabulary, but restrict insertions
        # to the strand count of the braid currently being evaluated.
        self.crossings = tuple(range(1, braid_index)) + tuple(range(-1, -braid_index, -1))
        self.insert_allowed = np.array(
            [abs(crossing) < self.active_braid_index for crossing in self.crossings], dtype=bool
        )
        self.channel = {crossing: index for index, crossing in enumerate(self.crossings)}
        width = max_word_length
        # Each move type owns a fixed range of action IDs. An ID's offset within
        # that range gives its word position; insertion also chooses a generator.
        self.offsets = {
            "delete": 0,
            "cancel": width,
            "commute": width + width - 1,
            "artin": width + 2 * (width - 1),
            "insert": width + 2 * (width - 1) + max(0, width - 2),
        }
        self.action_count = self.offsets["insert"] + (width + 1) * len(self.crossings)
        self.action_space = spaces.Discrete(self.action_count)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf,
            shape=(width * len(self.crossings) + 4,), dtype=np.float32,
        )
        self.word = []
        self.deleted = 0
        self.steps = 0
        self.moves = []
        self.best_cost = 0
        self.best_word = []
        self.best_moves = []

    def _validate_word(self, word):
        if len(word) > self.max_word_length:
            raise ValueError("Starting word exceeds max_word_length")
        values = list(word)
        if any(type(value) not in (int, np.int64, np.int32)
               or value == 0 or abs(value) >= self.active_braid_index for value in values):
            raise ValueError("Word contains an invalid generator")
        return [int(value) for value in values]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # RandomBraid currently uses NumPy's module-level generator.
            np.random.seed(seed)
        if self.initial_word is None:
            for _ in range(20):
                self.word = self._validate_word(RandomBraid(
                    max_braid_index=self.braid_index,
                    max_braid_length=self.max_word_length,
                    braid_length_stdev=self.random_length_stdev,
                ).word)
                if self.word:
                    break
            else:
                raise RuntimeError("RandomBraid repeatedly generated an empty word")
        else:
            self.word = self.initial_word.copy()
        self.start_word = self.word.copy()
        self.original_length = len(self.word)
        self.deleted = 0
        self.steps = 0
        self.moves = []
        self.best_cost = len(self.word)
        self.best_word = self.word.copy()
        self.best_moves = []
        return self._observation(), {"action_mask": self.valid_action_mask()}

    def _observation(self):
        """Flatten padded one-hot word slots, then append four scaled counters."""
        state = np.zeros(self.observation_space.shape, dtype=np.float32)
        channels = len(self.crossings)
        for position, crossing in enumerate(self.word):
            state[position * channels + self.channel[crossing]] = 1.0
        scale = max(1, self.max_word_length)
        state[-4:] = (
            self.original_length / scale,
            self.deleted / max(1, self.max_word_length + self.max_actions),
            self.best_cost / scale,
            self.steps / self.max_actions,
        )
        return state

    @staticmethod
    def _artin_replacement(triple):
        a, b, c = triple
        if a != c or abs(abs(a) - abs(b)) != 1 or (a > 0) != (b > 0):
            return None
        return [b, a, b]

    def valid_action_mask(self):
        """Mark the action IDs legal for the current word and strand count."""
        mask = np.zeros(self.action_count, dtype=bool)
        length = len(self.word)
        mask[:length] = True
        for position in range(length - 1):
            left, right = self.word[position:position + 2]
            mask[self.offsets["cancel"] + position] = left == -right
            mask[self.offsets["commute"] + position] = abs(abs(left) - abs(right)) >= 2
        for position in range(length - 2):
            mask[self.offsets["artin"] + position] = (
                self._artin_replacement(self.word[position:position + 3]) is not None
            )
        if length + 2 <= self.max_word_length:
            start = self.offsets["insert"]
            for position in range(length + 1):
                slot = start + position * len(self.crossings)
                mask[slot:slot + len(self.crossings)] = self.insert_allowed
        return mask

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space")
        if not self.valid_action_mask()[action]:
            raise ValueError(f"Action {action} is invalid for the current word")
        action = int(action)
        if action < self.offsets["cancel"]:
            position = action
            crossing = self.word.pop(position)
            self.deleted += 1
            move = {"type": "delete", "position": position, "crossing": crossing}
        elif action < self.offsets["commute"]:
            position = action - self.offsets["cancel"]
            pair = self.word[position:position + 2]
            del self.word[position:position + 2]
            move = {"type": "cancel", "position": position, "pair": pair}
        elif action < self.offsets["artin"]:
            position = action - self.offsets["commute"]
            self.word[position:position + 2] = self.word[position:position + 2][::-1]
            move = {"type": "commute", "position": position}
        elif action < self.offsets["insert"]:
            position = action - self.offsets["artin"]
            self.word[position:position + 3] = self._artin_replacement(
                self.word[position:position + 3]
            )
            move = {"type": "artin", "position": position}
        else:
            slot = action - self.offsets["insert"]
            position, crossing_index = divmod(slot, len(self.crossings))
            crossing = self.crossings[crossing_index]
            self.word[position:position] = [crossing, -crossing]
            move = {"type": "insert", "position": position, "crossing": crossing}
        self.moves.append(move)
        self.steps += 1
        cost = self.deleted + len(self.word)
        gain = max(0, self.best_cost - cost)
        if gain:
            self.best_cost = cost
            self.best_word = self.word.copy()
            self.best_moves = self.moves.copy()
        terminated = len(self.word) == 0
        truncated = self.steps >= self.max_actions and not terminated
        return self._observation(), gain / max(1, self.original_length), terminated, truncated, {
            "action_mask": self.valid_action_mask(), "best_length": self.best_cost,
        }

    def completed_result(self):
        """Finish the best prefix with deletions; return a word-only certificate."""
        automatic = [
            {"type": "delete", "position": 0, "crossing": crossing, "automatic": True}
            for crossing in self.best_word
        ]
        return {
            "initial_word": self.start_word.copy(),
            "initial_length": self.original_length,
            "best_length": self.best_cost,
            "actions_taken": len(self.best_moves),
            "moves": self.best_moves + automatic,
        }
