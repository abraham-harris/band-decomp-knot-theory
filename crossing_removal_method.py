"""Train and evaluate PPO policies that simplify braid words by crossing removal.

Run with --config configs/crossing_removal_*.json.  Every run writes a new
directory under runs/; train_each can resume from its saved run directory.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from crossing_removal_env import CrossingRemovalEnv
from experiment_utils import initialize_run, load_config, save_metrics, update_run_status


RESULT_FIELDS = (
    "braid_id", "braid_index", "true_rank", "initial_length", "best_length",
    "rank_gap", "seed", "actions_taken", "moves",
)


class ActorCritic(nn.Module):
    """Small PPO network with logits so invalid actions can be masked safely."""

    def __init__(self, state_size, action_size):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_size, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU())
        self.policy = nn.Linear(64, action_size)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        features = self.body(state)
        return self.policy(features), self.value(features).squeeze(-1)


def _distribution(logits, mask):
    return Categorical(logits=logits.masked_fill(~mask, -1e9))


def _seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def _json(path, data):
    with Path(path).open("w", encoding="utf-8") as output:
        json.dump(data, output, indent=2)
        output.write("\n")


def _dataset(data):
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    required = {"Braid index", "Braid ranks", "Braid word"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Dataset needs Braid index, Braid ranks, and Braid word columns")
    rows = []
    for braid_id, row in enumerate(reader):
        try:
            braid_index = int(row["Braid index"])
            true_rank = int(row["Braid ranks"])
            raw = row["Braid word"].strip()
            if not (raw.startswith("{") and raw.endswith("}")):
                raise ValueError("Word must be enclosed in braces")
            body = raw[1:-1].strip()
            word = [int(value.strip()) for value in body.split(",")] if body else []
            if braid_index < 2 or true_rank < 0 or any(
                value == 0 or abs(value) >= braid_index for value in word
            ):
                raise ValueError("Invalid braid index, rank, or generator")
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError(f"Invalid dataset row {braid_id + 2}: {error}") from error
        rows.append({"braid_id": braid_id, "braid_index": braid_index,
                     "true_rank": true_rank, "word": word})
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def _validate_config(raw):
    mode = raw.get("mode")
    if mode not in {"train_fixed", "train_random", "train_each", "inference"}:
        raise ValueError("mode must be train_fixed, train_random, train_each, or inference")
    allowed = {
        "name", "mode", "seed", "braid_index", "max_word_length", "max_actions",
        "word", "random_length_stdev", "epochs", "env_samples", "batch_size",
        "policy_epochs", "learning_rate", "gamma", "clip_epsilon", "entropy_coefficient",
        "data_path", "model_path", "attempts", "max_braid_index", "save_individual_models",
        "device",
    }
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(extra))}")
    config = {
        "name": raw.get("name", f"crossing_removal_{mode}"),
        "mode": mode,
        "seed": raw.get("seed", 0),
        "braid_index": raw.get("braid_index", 3),
        "max_word_length": raw.get("max_word_length", 80),
        "max_actions": raw.get("max_actions", 150),
        "random_length_stdev": raw.get("random_length_stdev", 24),
        "epochs": raw.get("epochs", 50),
        "env_samples": raw.get("env_samples", 10),
        "batch_size": raw.get("batch_size", 256),
        "policy_epochs": raw.get("policy_epochs", 5),
        "learning_rate": raw.get("learning_rate", 3e-4),
        "gamma": raw.get("gamma", 0.99),
        "clip_epsilon": raw.get("clip_epsilon", 0.2),
        "entropy_coefficient": raw.get("entropy_coefficient", 0.01),
        "data_path": raw.get("data_path", "data/braids_with_ranks.csv"),
        "attempts": raw.get("attempts", 10),
        "max_braid_index": raw.get("max_braid_index", 8),
        "save_individual_models": raw.get("save_individual_models", False),
        "device": raw.get("device", "auto"),
    }
    if "word" in raw:
        config["word"] = raw["word"]
    if "model_path" in raw:
        config["model_path"] = raw["model_path"]
    if config["seed"] is not None and (type(config["seed"]) is not int or
                                       not 0 <= config["seed"] < 2**32):
        raise ValueError("seed must be null or an integer in [0, 2**32)")
    for key in ("braid_index", "max_word_length", "max_actions", "epochs",
                "env_samples", "batch_size", "policy_epochs", "attempts",
                "max_braid_index"):
        if type(config[key]) is not int or config[key] < (2 if key in ("braid_index", "max_braid_index") else 1):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("random_length_stdev", "learning_rate", "gamma", "clip_epsilon"):
        value = config[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["gamma"] > 1 or config["clip_epsilon"] >= 1:
        raise ValueError("gamma must be <= 1 and clip_epsilon < 1")
    entropy = config["entropy_coefficient"]
    if type(entropy) not in (int, float) or not np.isfinite(entropy) or entropy < 0:
        raise ValueError("entropy_coefficient must be nonnegative and finite")
    if type(config["save_individual_models"]) is not bool:
        raise ValueError("save_individual_models must be a boolean")
    if config["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if mode == "train_fixed" and ("word" not in config or not config["word"]):
        raise ValueError("train_fixed requires a nonempty word")
    if mode == "inference" and not config.get("model_path"):
        raise ValueError("inference requires model_path")
    if mode == "train_random" and config["braid_index"] < 3:
        raise ValueError("RandomBraid requires braid_index at least 3")
    return config


def _environment(config, *, word=None, braid_index=None, random_mode=False):
    return CrossingRemovalEnv(
        braid_index=config["braid_index"] if braid_index is None else braid_index,
        max_word_length=config["max_word_length"], max_actions=config["max_actions"],
        word=None if random_mode else word,
        random_length_stdev=config["random_length_stdev"],
    )


def _choose(model, observation, mask, device):
    with torch.no_grad():
        state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        valid = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
        logits, value = model(state)
        distribution = _distribution(logits, valid)
        action = distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())


def _episode(env, model, device, *, training):
    observation, info = env.reset()
    transitions = []
    done = len(env.word) == 0
    while not done:
        mask = info["action_mask"]
        action, log_probability, value = _choose(model, observation, mask, device)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if training:
            transitions.append((observation, mask, action, log_probability, value, reward))
        observation = next_observation
        done = terminated or truncated
    return env.completed_result(), transitions, sum(item[-1] for item in transitions)


def _update(model, optimizer, transitions, config, device):
    states = torch.as_tensor(np.stack([item[0] for item in transitions]),
                             dtype=torch.float32, device=device)
    masks = torch.as_tensor(np.stack([item[1] for item in transitions]),
                            dtype=torch.bool, device=device)
    actions = torch.as_tensor([item[2] for item in transitions], device=device)
    old_log_probs = torch.as_tensor([item[3] for item in transitions],
                                    dtype=torch.float32, device=device)
    old_values = torch.as_tensor([item[4] for item in transitions],
                                 dtype=torch.float32, device=device)
    returns = torch.as_tensor([item[6] for item in transitions],
                              dtype=torch.float32, device=device)
    advantages = returns - old_values
    if len(transitions) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    losses = []
    for _ in range(config["policy_epochs"]):
        for indices in torch.randperm(len(transitions), device=device).split(config["batch_size"]):
            logits, values = model(states[indices])
            distribution = _distribution(logits, masks[indices])
            ratio = torch.exp(distribution.log_prob(actions[indices]) - old_log_probs[indices])
            surrogate = torch.minimum(
                ratio * advantages[indices],
                torch.clamp(ratio, 1 - config["clip_epsilon"],
                            1 + config["clip_epsilon"]) * advantages[indices],
            )
            policy_loss = -surrogate.mean()
            value_loss = nn.functional.mse_loss(values, returns[indices])
            loss = policy_loss + 0.5 * value_loss - config["entropy_coefficient"] * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.4)
            optimizer.step()
            losses.append((float(policy_loss.item()), float(value_loss.item())))
    return float(np.mean([pair[0] for pair in losses])), float(np.mean([pair[1] for pair in losses]))


def _score(result, random_mode):
    if random_mode:
        return (result["initial_length"] - result["best_length"]) / max(1, result["initial_length"])
    return -result["best_length"]


def train_policy(config, *, word=None, braid_index=None, random_mode=False):
    env = _environment(config, word=word, braid_index=braid_index, random_mode=random_mode)
    requested_device = config["device"]
    device = torch.device(
        "cuda" if requested_device == "cuda" or
        (requested_device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    model = ActorCritic(env.observation_space.shape[0], env.action_count).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best = None
    metrics = []
    try:
        for epoch in range(config["epochs"]):
            memory = []
            lengths = []
            improvements = []
            for _ in range(config["env_samples"]):
                result, transitions, _ = _episode(env, model, device, training=True)
                running_return = 0.0
                episode_returns = []
                for transition in reversed(transitions):
                    running_return = transition[5] + config["gamma"] * running_return
                    episode_returns.append(running_return)
                memory.extend(transition + (value,) for transition, value in
                              zip(transitions, reversed(episode_returns)))
                lengths.append(result["best_length"])
                improvements.append(result["initial_length"] - result["best_length"])
                if best is None or _score(result, random_mode) > _score(best, random_mode):
                    best = result
            if not memory:
                raise RuntimeError("Training collected no transitions")
            policy_loss, value_loss = _update(model, optimizer, memory, config, device)
            metrics.append({
                "epoch": epoch, "mean_best_length": float(np.mean(lengths)),
                "mean_improvement": float(np.mean(improvements)),
                "policy_loss": policy_loss, "value_loss": value_loss,
            })
            if (epoch + 1) % max(1, config["epochs"] // 10) == 0 or epoch + 1 == config["epochs"]:
                print(f"Epoch {epoch + 1}/{config['epochs']}: mean completed length "
                      f"{metrics[-1]['mean_best_length']:.2f}, mean improvement "
                      f"{metrics[-1]['mean_improvement']:.2f}", flush=True)
    finally:
        env.close()
    return model.cpu(), best, metrics


def _checkpoint(model, config, path, braid_index=None):
    torch.save({
        "state_dict": model.state_dict(),
        "braid_index": config["braid_index"] if braid_index is None else braid_index,
        "max_word_length": config["max_word_length"],
        "max_actions": config["max_actions"],
    }, path)


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    braid_index = checkpoint["braid_index"]
    max_word_length = checkpoint["max_word_length"]
    probe = CrossingRemovalEnv(braid_index=braid_index,
                               max_word_length=max_word_length,
                               max_actions=checkpoint["max_actions"], word=[])
    model = ActorCritic(probe.observation_space.shape[0], probe.action_count)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    probe.close()
    return model, checkpoint


def _save_rows(rows, path):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=".crossing_results_", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _result_row(braid, result, seed):
    return {
        "braid_id": braid["braid_id"], "braid_index": braid["braid_index"],
        "true_rank": braid["true_rank"], "initial_length": result["initial_length"],
        "best_length": result["best_length"],
        "rank_gap": result["best_length"] - braid["true_rank"],
        "seed": seed, "actions_taken": result["actions_taken"],
        "moves": json.dumps(result["moves"], separators=(",", ":")),
    }


def _plot(rows, path):
    if not rows:
        return
    import matplotlib.pyplot as plt
    ordered = sorted(rows, key=lambda row: (int(row["true_rank"]), int(row["initial_length"])))
    x = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(11, 4))
    try:
        for key, label in (("true_rank", "Known rank"),
                           ("initial_length", "Initial length"),
                           ("best_length", "Crossings removed")):
            axis.scatter(x, [int(row[key]) for row in ordered], label=label, s=16)
        axis.set_xlabel("Test braid, sorted by known rank")
        axis.set_ylabel("Band decomposition length")
        axis.legend()
        axis.grid()
        figure.tight_layout()
        figure.savefig(path)
    finally:
        plt.close(figure)


def _run_single(config, run_directory):
    random_mode = config["mode"] == "train_random"
    model, best, metrics = train_policy(
        config, word=None if random_mode else config["word"], random_mode=random_mode,
    )
    _checkpoint(model, config, run_directory / "model.pt")
    _json(run_directory / "results.json", {
        "mode": config["mode"], "episodes": config["epochs"] * config["env_samples"],
        "best_example": best, "final_epoch": metrics[-1], "model_file": "model.pt",
    })
    save_metrics(metrics, run_directory / "metrics.csv")


def _run_each(config, run_directory, *, resume=False):
    if resume:
        data = (run_directory / "dataset.csv").read_bytes()
        with (run_directory / "results.csv").open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != list(RESULT_FIELDS):
                raise ValueError("Saved results.csv has unexpected columns")
            rows = list(reader)
    else:
        data = Path(config["data_path"]).read_bytes()
        (run_directory / "dataset.csv").write_bytes(data)
        rows = []
        _save_rows(rows, run_directory / "results.csv")
    if hashlib.sha256(data).hexdigest() != config.get("dataset_sha256"):
        raise ValueError("Dataset snapshot does not match the run config")
    braids = _dataset(data)
    selected = [braid for braid in braids if braid["braid_index"] <= config["max_braid_index"]]
    selected = [braid for braid in selected if len(braid["word"]) <= config["max_word_length"]]
    if not selected:
        raise ValueError("No dataset braids fit max_braid_index and max_word_length")
    completed = {int(row["braid_id"]) for row in rows}
    for braid in selected:
        if braid["braid_id"] in completed:
            continue
        seed = None if config["seed"] is None else (config["seed"] + braid["braid_id"]) % 2**32
        _seed(seed)
        model, best, _ = train_policy(config, word=braid["word"],
                                      braid_index=braid["braid_index"])
        if config["save_individual_models"]:
            model_dir = run_directory / "models"
            model_dir.mkdir(exist_ok=True)
            _checkpoint(model, config, model_dir / f"braid_{braid['braid_id']:03d}.pt",
                        braid_index=braid["braid_index"])
        rows.append(_result_row(braid, best, seed))
        _save_rows(rows, run_directory / "results.csv")
        print(f"Braid {braid['braid_id']}: {best['best_length']} crossings removed "
              f"(known rank {braid['true_rank']})", flush=True)
    _plot(rows, run_directory / "plot.png")
    _json(run_directory / "summary.json", {
        "braids_completed": len(rows), "braids_selected": len(selected),
        "known_rank_reached": sum(int(row["best_length"]) == int(row["true_rank"]) for row in rows),
    })


def _run_inference(config, run_directory):
    model, checkpoint = _load_checkpoint(config["model_path"])
    data = Path(config["data_path"]).read_bytes()
    (run_directory / "dataset.csv").write_bytes(data)
    braids = _dataset(data)
    selected = [braid for braid in braids if braid["braid_index"] <= checkpoint["braid_index"]
                and len(braid["word"]) <= checkpoint["max_word_length"]]
    if not selected:
        raise ValueError("No dataset braids fit the saved model's strand and word capacity")
    rows = []
    for braid in selected:
        best = None
        for attempt in range(config["attempts"]):
            seed = None if config["seed"] is None else (
                config["seed"] + braid["braid_id"] * config["attempts"] + attempt
            ) % 2**32
            _seed(seed)
            env = CrossingRemovalEnv(
                braid_index=checkpoint["braid_index"],
                active_braid_index=braid["braid_index"],
                max_word_length=checkpoint["max_word_length"],
                max_actions=config["max_actions"], word=braid["word"],
            )
            try:
                result, _, _ = _episode(env, model, torch.device("cpu"), training=False)
            finally:
                env.close()
            if best is None or result["best_length"] < best["best_length"]:
                best = result
        rows.append(_result_row(braid, best, config["seed"]))
    _save_rows(rows, run_directory / "results.csv")
    _plot(rows, run_directory / "plot.png")
    _json(run_directory / "summary.json", {
        "braids_evaluated": len(rows), "dataset_braids": len(braids),
        "known_rank_reached": sum(int(row["best_length"]) == int(row["true_rank"]) for row in rows),
        "model_path": config["model_path"], "attempts_per_braid": config["attempts"],
    })
    print(f"Evaluated {len(rows)}/{len(braids)} dataset braids; "
          f"known rank reached on {sum(int(row['best_length']) == int(row['true_rank']) for row in rows)}.",
          flush=True)


def run_experiment(config_path=None, *, resume=None, name=None, runs_root="runs"):
    if (config_path is None) == (resume is None):
        raise ValueError("Provide exactly one of config_path or resume")
    if resume is not None:
        run_directory = Path(resume)
        config = load_config(run_directory / "config.json")
        if config["mode"] != "train_each":
            raise ValueError("Only train_each runs can resume")
    else:
        config = _validate_config(load_config(config_path))
        if name is not None:
            config["name"] = name
        if config["mode"] == "train_each":
            config["dataset_sha256"] = hashlib.sha256(Path(config["data_path"]).read_bytes()).hexdigest()
        run_directory = initialize_run(config, runs_root=runs_root)
    print(f"Run directory: {run_directory}", flush=True)
    try:
        _seed(config["seed"])
        if config["mode"] == "train_each":
            _run_each(config, run_directory, resume=resume is not None)
        elif config["mode"] == "inference":
            _run_inference(config, run_directory)
        else:
            _run_single(config, run_directory)
        update_run_status(run_directory, "completed")
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        update_run_status(run_directory, status, error=f"{type(error).__name__}: {error}")
        raise
    return run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="New experiment config")
    source.add_argument("--resume", help="Resume a train_each run directory")
    parser.add_argument("--name", help="Override the run name")
    parser.add_argument("--runs-root", default="runs", help="Parent directory for run outputs")
    args = parser.parse_args()
    if args.name is not None and args.config is None:
        parser.error("--name requires --config")
    run_experiment(args.config, resume=args.resume, name=args.name, runs_root=args.runs_root)


if __name__ == "__main__":
    main()
