# Project working notes

- Goal: train an RL policy to find short band decompositions of braids. Read `prospectus.pdf` for the research goals.
- Training runs on a remotely accessed supercomputer with a separate copy of this folder. The user manually copies code and results between machines. Keep changes and diagnostic commands easy to transfer; do not assume local and remote files or dependencies match.
- Use the existing `../.knotenv` environment. Locally it was created under WSL Ubuntu with Python 3.12.3, not native Windows Python. Its interpreter is `/mnt/c/Users/Abe/Desktop/Programming/KnotTheory/.knotenv/bin/python`. A saved remote log references Python 3.9; verify versions when reproducing errors.
- An unused `.venv` was created during initial diagnostics; dependency installation did not complete. Do not use it in place of `.knotenv`.
- `main.py` implements custom PyTorch PPO, curriculum training, and inference. Its current script entry point has training commented out and runs inference. Importing it does not start the full run.
- `band_env.py` implements the Gymnasium environment, band moves, matrix and one-hot observations, and curriculum generation. `generator.py` uses `slice.py` to generate random braids. `tuning.py` contains a separate PPO implementation and Optuna tuning.
- Preserve existing logs, plots, datasets, and checkpoints. Use separate paths for diagnostics rather than overwriting research results.
- The original supercomputer training traceback is unavailable. Reproduce failures in small tests before attributing a cause. `diagnostics/diagnose_training.py` checks observations, reset API, padding capacity, and a short CPU PPO update without saving research outputs.
- Confirmed locally: reset uses matrix observations while step uses one-hot observations; reset returns only an array; inference ignores the supplied braid because it selects random mode. The mixed representation already exists in the initial Git commit, so history does not establish which encoding was intended. The user does not recall and wants investigation before choosing.
- Follow-up decision: compare matrix observations (`get_state`) with one-hot observations (`get_state_ohe`) before settling on a representation for training. The current working change makes reset and step both use matrix observations for consistency; this does not establish that matrices are better for learning. Ask the user before making further code changes.
- Confirmed crash: with NumPy seed 0, `BandEnv(braid_index=8, max_num_bands=80, train_type="curriculum", difficulty=1)` exceeds the observation capacity and raises `ValueError: index can't contain negative values`. A deterministic 3-strand environment with `[1, 2, 1]` also raises that error in one-hot padding on a no-op step. These are distinct capacity bugs.
- Local diagnostic versions: Python 3.12.3, NumPy 2.3.3, Gymnasium 1.2.0, PyTorch 2.8.0+cu128. `pypdf` 6.19.0 was added to the existing `.knotenv` to read the prospectus. The CPU PPO smoke test passed with 10 transitions and two optimization epochs; this does not establish long-run training stability. Four environment diagnostic checks failed as described above. Production code has not yet been changed.
- Prospectus context: seek short band decompositions as a route to constructing low-genus surfaces / upper bounds; use known-rank braids for evaluation. Planned avenues include fixed small braid index, alternative representations, curriculum and reward improvements, classical simplification, and supermoves/dynamic action spaces.

# Safety

- Never commit API keys, tokens, passwords, or `.env` files. Never print credential-file contents. Flag hardcoded secrets if encountered.
- Ask for confirmation before destructive or irreversible actions, including force-pushing, rewriting Git history, pushing directly to `main`, and bulk deletes.
