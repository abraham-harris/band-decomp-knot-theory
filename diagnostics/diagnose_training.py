"""Small, read-only diagnostics; run with the project's .knotenv Python.

Run: ../.knotenv/bin/python -m diagnostics.diagnose_training

Does not save checkpoints, plots, or training logs. Failures are reported and
the remaining checks continue. This is a diagnostic, not a full training run.
"""

import copy
import platform
import traceback

import gymnasium
import numpy as np

from band_env import BandEnv


def check_observations():
    env = BandEnv(band_decomposition=[1, 2], train_type="deterministic",
                  braid_index=8, max_num_bands=80)
    result = env.reset(seed=0)
    state = result[0] if isinstance(result, tuple) else result
    original = copy.deepcopy(env.band_decomposition)
    next_state = env.step(78)[0]  # A slide beyond the current braid: no change.
    assert env.band_decomposition == original
    assert np.array_equal(state, next_state), (
        "reset and step encode the same unchanged braid differently"
    )


def check_reset_api():
    env = BandEnv(band_decomposition=[1, 2], train_type="deterministic")
    result = env.reset(seed=0)
    assert isinstance(result, tuple) and len(result) == 2, (
        "Gymnasium reset must return (observation, info)"
    )


def check_three_strand_step():
    env = BandEnv(band_decomposition=[1, 2, 1], train_type="deterministic")
    env.reset(seed=0)
    env.step(14)


def check_curriculum_capacity():
    for difficulty in range(4):
        for seed in range(4):
            np.random.seed(seed)
            env = BandEnv(braid_index=8, max_num_bands=80,
                          train_type="curriculum", difficulty=difficulty)
            assert len(env.band_decomposition) <= env.max_num_bands
            env.close()


def check_ppo_update():
    import random
    import torch
    from itertools import chain
    from torch.utils.data import DataLoader
    import main

    # Exercise the real PPO code on CPU, independent of cluster GPU availability.
    main.device = "cpu"
    torch.set_num_threads(1)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    print(f"PyTorch={torch.__version__}; smoke-test device=cpu")
    env = gymnasium.make("BandEnv-v0", braid_index=8, max_num_bands=80,
                         train_type="curriculum", difficulty=0)
    policy = main.PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    value = main.ValueNetwork(env.observation_space.shape[0])
    optimizer = torch.optim.Adam(chain(policy.parameters(), value.parameters()),
                                 lr=0.0008432777999828978)
    memory = []
    for episode in range(2):
        result = env.reset(seed=episode)
        state = result[0] if isinstance(result, tuple) else result
        rollout = []
        for _ in range(5):
            action, distribution = main.get_action_ppo(policy, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rollout.append((state, action, distribution, reward))
            state = next_state
            if terminated or truncated:
                break
        main.calculate_return(memory, rollout, 0.9082237929205784)
    loader = DataLoader(main.RLDataset(memory), batch_size=4, shuffle=True)
    losses = main.learn_ppo(optimizer, policy, value, loader,
                            0.16273153856100495, 2, env.action_space.n)
    assert np.isfinite(losses).all(), f"Non-finite PPO losses: {losses}"
    print(f"Transitions={len(memory)}; policy/value losses={losses}")
    env.close()


def check_training_loops():
    import torch
    import main

    main.device = "cpu"
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    for name, run, kwargs in (
        ("regular", main.ppo_main, dict(max_actions=3)),
        ("curriculum", main.ppo_main_curriculum,
         dict(difficulties=(0, 1, 2, 3), max_actions_per_stage=3)),
    ):
        rewards, policy_losses, value_losses, _ = run(
            epochs=1, env_samples=2, save_path=None, **kwargs
        )
        assert len(rewards) == (8 if name == "curriculum" else 2)
        assert np.isfinite(policy_losses).all() and np.isfinite(value_losses).all()
        print(f"{name}: episodes={len(rewards)}, updates={len(policy_losses)}")


if __name__ == "__main__":
    print(f"Python={platform.python_version()}; NumPy={np.__version__}; "
          f"Gymnasium={gymnasium.__version__}", flush=True)
    failures = 0
    for check in (check_observations, check_reset_api, check_three_strand_step,
                  check_curriculum_capacity, check_ppo_update,
                  check_training_loops):
        print(f"\n{check.__name__}", flush=True)
        try:
            check()
            print("PASS", flush=True)
        except Exception:
            failures += 1
            traceback.print_exc()
    print(f"\n{failures} diagnostic checks failed.", flush=True)
    raise SystemExit(bool(failures))
