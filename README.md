# Deep Reinforcement Learning for Simplifying Braid Band Decompositions

## Brief Topological Background
In knot theory, a braid is a set of $n$ strings that are attached to a horizontal bar at the top and travel downward, crossing over and under each other however they like (without going upward) before attaching to a horizontal bar at the bottom.

<div align="center">
  <img src="/README_images/braid.png" alt="Braid" width="200">
  <br>
  <sub><em>Figure 1: an example of a braid.</em></sub>
  <br>
  <sub>Source: Dylan Skinner Blog (https://dylanskinner.dev/blog/braids)</sub>
  <br><br>
</div>

One reason braids are useful in knot theory is that they can be easily converted to knots or links by connecting the bottom strands to the top.

<div align="center">
  <img src="/README_images/braid_to_knot.png" alt="Braid" width="300">
  <br>
  <sub><em>Figure 2: turning a braid into a knot.</em></sub>
  <br>
  <sub>Source: Dylan Skinner Blog (https://dylanskinner.dev/blog/braids)</sub>
  <br><br>
</div>

To more easily talk about braids, we can label their crossings. If the crossing occurs between strands $j$ and $k$ going from left to right, we label it $\sigma_j$. If the same crossing goes right to left, we label it $\sigma_{j-1}$.

<div align="center">
  <img src="/README_images/crossings.png" alt="Braid" width="300">
  <br>
  <sub><em>Figure 3: crossings in a braid.</em></sub>
  <br>
  <sub>Source: Adams, Colin C. The Knot Book (page 133)</sub>
  <br><br>
</div>

This allows us to represent braids as “braid words.” For example, the braid in Figure 4 can be written as $\sigma_2 \sigma_1 \sigma_1 \sigma_2^{-1} \sigma_1 \sigma_1$.

<div align="center">
  <img src="/README_images/braid_word_example.png" alt="Braid" width="150">
  <br>
  <sub><em>Figure 4: braid example.</em></sub>
  <br>
  <sub>Source: Adams, Colin C. The Knot Book (page 133)</sub>
  <br><br>
</div>

Braids can also be decomposed into bands. A band is an element of the form $\omega \sigma_i \omega^{-1}$ where $\sigma_i$ is a crossing and $\omega$ is another braid in the braid group with the same number of strands. An upper bound on a braid’s minimal length band decomposition is the number of crossings in the braid, because each crossing is a trivial band.

## Research Goal
Finding a braid’s minimal length band decomposition can be challenging, since in some cases adding new crossings and twists to our braid can allow us to decompose it into shorter bands, even though the number of crossings has increased. I explore the possibility of training a deep reinforcement learning model to take as input a braid and output the shortest band decomposition it can find. This involves creating a custom RL environment, curriculum learning, and other techniques. 

Being able to find shortest band decompositions would be useful in studying other problems in knot theory, such as slice genus and quasipositive braid detection.

## Running experiments

Run these commands from the repository root using the existing `../.knotenv` environment:

```bash
../.knotenv/bin/python main.py --config configs/regular.json
../.knotenv/bin/python main.py --config configs/curriculum.json
../.knotenv/bin/python main.py --config configs/specific_braid.json
../.knotenv/bin/python main.py --config configs/inference.json
```

Each command starts the experiment specified by its config. Training configs expose
the PPO hyperparameters, braid index, band capacity, epoch and episode counts, and
action limits. Curriculum `epochs` is the number of epochs **per difficulty**;
`max_actions_per_stage` is multiplied by `difficulty + 1` to set that stage's action
limit. Specific-braid training uses the config's `band_decomposition`.

Use `--name` to override the descriptive name for one run:

```bash
../.knotenv/bin/python main.py --config configs/specific_braid.json --name my_experiment
```

The name follows the timestamp; it does not replace it. Without `--name`, the name
comes from the config. Spaces and other filename-unfriendly characters are
replaced with underscores in the directory name. Use `--help` to see the command
options. Running without `--config` retains the existing hardcoded entry-point
behavior and output paths; use the config commands for organized experiments.

### Run directories and outputs

Each configured experiment creates a unique directory under `runs/`, using a UTC
timestamp with microseconds and the descriptive name. For example, specific-braid
training produces:

```text
runs/20260930_143012_123456Z_my_experiment/
├── config.json
├── run_info.json
├── results.json
├── metrics.csv
├── plot.png
└── Braid_Simplificationator_specific
```

`config.json` records the resolved parameters, including omitted parameters filled
from the function defaults and any name override. `run_info.json` records start
and finish times, Git revision and dirty status, and whether the run is `running`,
`completed`, or `failed`. Failed runs also record the error and may have only
partial outputs.

All training modes save one row per epoch to `metrics.csv`, with `epoch`,
`mean_return`, `policy_loss`, and `value_loss`. Curriculum adds `difficulty`;
specific-braid training adds the best-so-far `best_length` at the end of each epoch.
Epoch numbers start at zero and continue across curriculum stages. `plot.png`
uses these same epoch means and, for specific-braid training, best lengths.
`results.json` contains final summaries and, for specific-braid training, the best
decomposition and action/move sequences. Configured runs do not save detailed
environment logs or repeat epoch histories in the results JSON.

With `save_model: true`, the model filename is the configured `model_name`. Model
saves contain `policy_network.state_dict()`: intermediate saves use the existing
`epoch > 0 and epoch % 1000 == 0` condition, and a final save follows training.
All saves overwrite the same model file inside that run's directory. Set
`save_model: false` to skip model saves. Existing artifacts are left in place;
generated `runs/` directories are ignored by Git.

### Seeds

The supplied configs use `"seed": null`, preserving the current unseeded behavior.
An integer from `0` through `4294967295` seeds Python's `random`, NumPy, and PyTorch
before the experiment starts. The resolved seed is saved in `config.json`.

### Training separately on every dataset braid

The batch runner calls the existing specific-braid PPO trainer for **every CSV
row**, creating a fresh policy and optimizer for each braid. It uses each row's
`Braid index` as its strand count and tracks the shortest decomposition encountered
at **any step** across all that braid's training episodes, including its starting
decomposition. This is the best length found within the training budget, not a
proof of minimality. Known ranks are used for comparison, not as a stopping rule.

```bash
../.knotenv/bin/python batch_training.py --config configs/batch_braids.json
```

Edit `configs/batch_braids.json` to set the training budget and PPO parameters.
The supplied configuration uses 500 epochs per braid, 10 episodes per epoch,
at most 150 actions per episode, and a capacity of 80 bands. The current dataset
contains 100 braids, with starting word lengths up to 59. The full run therefore
has a budget of 500,000 episodes. Reduce these settings for an initial trial.
`--name` and `--runs-root` can customize the run name and parent directory.

The runner prints its unique run directory. It contains `config.json`,
`run_info.json`, a `dataset.csv` snapshot, one `results.csv`, and a single aggregate
`plot.png` after completion. No per-braid models, images, training histories, or
environment logs are saved. The single-braid trainer also skips accumulating its
epoch environment logs in memory for this mode.

Each result has a zero-based `braid_id` identifying the original CSV row,
`braid_index`, `true_rank`, `initial_length`, `best_length`, `rank_gap`
(`best_length - true_rank`), seed, completed epoch/episode counts, elapsed seconds,
and JSON-encoded best decomposition, action sequence, and discovery location.
For example, pandas can read it with `pd.read_csv("runs/.../results.csv")`;
use `json.loads` to decode the decomposition or action sequence if needed.

After **each completed braid**, the results CSV is flushed to disk and atomically
replaced before training starts on the next braid. Completed rows survive Ctrl+C,
job termination, or a later braid's failure. The braid currently training is
restarted from scratch when resuming; its partial progress is not saved.

```bash
../.knotenv/bin/python batch_training.py --resume runs/<run_directory>
../.knotenv/bin/python batch_training.py --plot-only runs/<run_directory>
```

Resume uses the saved configuration and dataset snapshot, skips completed IDs,
and writes into that same run directory. Run only one process per run directory.
For a configured seed, braid `i` uses `(seed + i) % 2**32`, so restarting does not
depend on how many preceding braids have already finished; `null` leaves training
unseeded. Exact reproducibility also depends on hardware and library versions.
Ctrl+C and SIGTERM mark the run as `interrupted`; an uncatchable kill may leave
`run_info.json` marked `running`, but the saved CSV can still be resumed.

The plot command works on any nonempty set of completed results, even before the
experiment finishes. It compares known rank, initial length, and best achieved
length in the same descending order as the inference plot. Partial plots include
completed braids only. To transfer or resume on the supercomputer, copy the code
and config, and copy the entire run directory when resuming. Use that machine's
Python environment; the local `../.knotenv/bin/python` path is for WSL.

### Selecting a model for inference

Set `model_path` in an inference config to the saved model you want to evaluate,
for example:

```json
"model_path": "runs/20260930_143012_123456Z_my_experiment/Braid_Simplificationator_specific"
```

Set `braid_index` and `max_num_bands` to the values used to train that model so the
network dimensions match. `data_path` selects the evaluation CSV, and
`max_actions` limits the actions taken for each braid. Relative paths are resolved
from the working directory where the command is run.

Inference creates its own run directory containing `config.json`, `run_info.json`,
`results.json`, `plot.png`, `inference_logfile.txt`, and
`inference_final_forms.txt`. It reads the selected model without copying or
overwriting it. Inference results include the ranks, starting/final/best lengths,
best decompositions, and returns for the evaluated braids.
