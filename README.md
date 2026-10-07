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

Run commands from the repository root with the existing `../.knotenv` environment:

```bash
../.knotenv/bin/python main.py --config configs/regular.json         # regular PPO
../.knotenv/bin/python main.py --config configs/curriculum.json      # curriculum PPO
../.knotenv/bin/python main.py --config configs/specific_braid.json  # one braid
../.knotenv/bin/python main.py --config configs/inference.json       # evaluate a model
```

Edit the configs to change PPO settings and training budgets. Curriculum `epochs`
applies to each difficulty; its action limit is `max_actions_per_stage` times
`difficulty + 1`. Specific-braid training uses `band_decomposition` from its
config. Add `--name my_experiment` to override a run name. Running `main.py`
without `--config` uses its older hardcoded entry point and output paths.

### Run directories and outputs

Each configured run gets a timestamped directory under `runs/`. For example,
specific-braid training produces:

```text
runs/20260930_143012_123456Z_my_experiment/
├── config.json
├── run_info.json
├── results.json
├── metrics.csv
├── plot.png
└── Braid_Simplificationator_specific
```

`config.json` records the settings used; `run_info.json` records status, times,
and Git state.
Training runs also write `results.json` (final summary), `metrics.csv` (one row
per epoch), and `plot.png`. Specific-braid results include the best decomposition
and the actions that found it. Inference runs write `results.json`, `plot.png`,
`inference_logfile.txt`, and `inference_final_forms.txt`.

For `main.py` training configs, `save_model: true` saves policy weights under
`model_name` in the run directory. Periodic saves and the final save overwrite
that file. Use `save_model: false` to skip it. Batch runs use the rules below.

### Seeds

`"seed": null` leaves a run unseeded. An integer from `0` through `4294967295`
seeds Python, NumPy, and PyTorch. Batch runs use `(seed + braid_id) % 2**32`
for each braid. Exact reproducibility also depends on hardware and library
versions.

## Batch training

Both batch modes record the shortest decomposition seen at any step, including
the starting one. Known rank is used for comparison, not as a stopping rule or
proof of minimality. The supplied configs train 50 epochs per braid.

### Independent policies

```bash
../.knotenv/bin/python batch_training.py --config configs/batch_braids.json
```

This trains a fresh policy for every braid in the dataset. It saves each
completed braid's result to `results.csv` but does not save policy models.
The run directory also contains a dataset snapshot and an aggregate `plot.png`
when training completes.

`braid_id` is the original CSV row number, starting at zero. Each result records
known rank, starting and best lengths, the gap between best length and rank,
training counts, time, and the best decomposition and action sequence. The last
two are JSON strings inside the CSV.

Completed rows survive interruption. To continue an independent run, use its
saved run directory; the braid in progress restarts from scratch:

```bash
../.knotenv/bin/python batch_training.py --resume runs/<run_directory>
```

### Iterative policies

Set `"iterative": true` and `"braid_index"` to train all dataset braids with
up to that many strands. The sample config includes all 100 braids up to index
8:

```bash
../.knotenv/bin/python batch_training.py --config configs/batch_braids_iterative.json
```

Braids train from smallest known rank to largest, breaking ties by starting
length and then CSV row. Each strand count keeps its own policy and optimizer
in memory; training continues when another braid with that strand count appears.
Only after the full run ends does it save one final policy per strand count,
for example `models/braid_index_4.pt`. It writes no training checkpoints, so
an interrupted iterative run cannot resume, though its completed CSV rows remain.

### Plots and inference

Plot completed batch results at any time:

```bash
../.knotenv/bin/python batch_training.py --plot-only runs/<run_directory>
```

Batch and inference plots show smallest known rank first, then shortest
starting decomposition. To evaluate a saved policy, set `model_path` in
`configs/inference.json`, with the `braid_index` and `max_num_bands` used to
train it. `data_path` selects the evaluation dataset. Inference reads the model
and writes its own run directory.

On the supercomputer, use its Python environment. Copy the entire run directory
when transferring results or resuming an independent batch run.
