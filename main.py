import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from itertools import chain
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import copy
import argparse
import inspect
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
from generator import RandomBraid
from band_env import BandEnv
from experiment_utils import load_config, initialize_run, update_run_status, save_metrics



InteractiveShell.ast_node_interactivity = 'all' # Use before asynchronous code

# Change runtime type
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

torch.autograd.set_detect_anomaly(True)

# Make environment compatible with gym API
register(
    id='BandEnv-v0',
    entry_point='band_env:BandEnv',
    max_episode_steps=200,
)



### PPO ALGORITHM ###
def calculate_return(memory, rollout, gamma):
  """Return memory with calculated return in experience tuple

    Args:
        memory (list): (state, action, action_dist, return) tuples
        rollout (list): (state, action, action_dist, reward) tuples from last rollout
        gamma (float): discount factor

    Returns:
        list: memory updated with (state, action, action_dist, return) tuples from rollout
  """
  running_return = 0
  for i, transition in enumerate(reversed(rollout)): # calculate rollout in reverse
    state, action, action_distribution, reward = transition
    running_return = reward + gamma*running_return # add discounted return to new reward for new return
    rollout[len(rollout) - i - 1] = (state, action, action_distribution, running_return)
  memory.extend(rollout) # add rollout to end of memory
  return memory


def get_action_ppo(network, state):
    """Sample action from the distribution obtained from the policy network

        Args:
            network (PolicyNetwork): Policy Network
            state (np-array): current state, size (state_size)

        Returns:
            int: action sampled from output distribution of policy network
            array: output distribution of policy network
    """
    # Since we are gathering data, we don't want the gradient info for the action_distribution
    # If you don't use torch.no_grad() you will have to detach action_distribution in the loss function
    with torch.no_grad():
        # PPO acts according to the network's policy; no purely random or greedy actions like DQN
        # Get action distribution
        action_distribution = network(torch.from_numpy(state).float().unsqueeze(0).to(device)).squeeze(0)
        # Sample from action distribution
        try:
            selected_action = torch.multinomial(action_distribution, 1).item()
        except RuntimeError:
            print("ERROR CAUGHT")
            print(action_distribution)
            changed_dist = nn.functional.normalize(torch.ones_like(action_distribution), dim=0, p=1)
            selected_action = torch.multinomial(changed_dist, 1).item()
            # TODO: RE-INITIALIZE NETWORK WEIGHTS??? After normalizing, this issue has stopped coming up.
    # Return both values: we use action_distribution in the loss function as old policy (pi_old)
    return selected_action, action_distribution


def learn_ppo(optim, policy, value, memory_dataloader, epsilon, policy_epochs, action_size):
    """Implement PPO policy and value network updates. Iterate over your entire
        memory the number of times indicated by policy_epochs.

        Args:
            optim (Adam): value and policy optimizer
            policy (PolicyNetwork): Policy Network
            value (ValueNetwork): Value Network
            memory_dataloader (DataLoader): dataloader with (state, action, action_dist, return, discounted_sum_rew) tensors
            epsilon (float): trust region
            policy_epochs (int): number of times to iterate over all memory
            action_size (int): number of possible actions
    """
    policy_losses = []
    value_losses = []
    for epoch in range(policy_epochs):
        for state, action, action_distribution, returns in memory_dataloader:
            optim.zero_grad()

            state, action, action_distribution, returns = state.float().to(device), action.to(device), \
                                                            action_distribution.to(device), returns.float().to(device)

            # Value loss: simple regression MSE loss - try to get state value to match the actual return
            state_value = value(state).squeeze()
            value_loss = nn.functional.mse_loss(returns, state_value)

            # Policy loss: simple policy gradient w/ policy ratio w/ clipping
            # Advantage: how much more (or less) return did we get than what we expected at this state?
            advantage = returns - state_value
            advantage = advantage.detach()

            # Turn actions into one-hot encoding, since our loss only uses the actions we took
            action_ohe = nn.functional.one_hot(action, num_classes=action_size).bool()

            # Get action distribution of current policy
            current_policy = policy(state)[action_ohe]

            # Use the action distribution used to gather the data (while experiencing the environment) as the "old" policy
            old_policy = action_distribution[action_ohe]

            # Policy ratio: how much has our policy changed?
            policy_ratio = current_policy / old_policy

            # Policy gradient loss term using ratio (vanilla is just current_policy*A)
            policy_grad_loss = policy_ratio * advantage

            # Clipping: prevents incentivizing large-scale changes from the current policy
            clipped_policy_grad_loss = torch.clamp(policy_ratio, 1-epsilon, 1+epsilon) * advantage

            # PPO Loss: minimum between policy grad loss w/ and w/o ratio clipping
            policy_loss = -torch.mean(torch.min(policy_grad_loss, clipped_policy_grad_loss))

            loss = value_loss + policy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.4)
            torch.nn.utils.clip_grad_norm_(value.parameters(), 0.4)
            optim.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    return np.mean(policy_losses), np.mean(value_losses)


# Dataset that wraps memory for a dataloader
class RLDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = []
        for d in data:
            self.data.append(d)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        hidden_size = 32 

        self.action_size = action_size

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=1)
        )

        def init_weights(m) :
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.net.apply(init_weights)

    def forward(self, x):
        """Get policy from state

        Args:
            state (tensor): current state, size (batch x state_size)

        Returns:
            action_dist (tensor): probability distribution over actions (batch x action_size)
        """
        return self.net(x)


# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        hidden_size = 32

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        def init_weights(m) :
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.net.apply(init_weights)

    def forward(self, x):
        """Estimate value given state

        Args:
            state (tensor): current state, size (batch x state_size)

        Returns:
            value (tensor): estimated value, size (batch)
        """
        return self.net(x)
    

def ppo_main(epochs=5000, env_samples=10, max_actions=150,
             save_path="./models/Braid_Simplificationator_2000", *,
             braid_index=8, max_num_bands=80,
             learning_rate=0.0008432777999828978,
             gamma=0.9082237929205784, batch_size=256,
             epsilon=0.16273153856100495, policy_epochs=5):
    """Train PPO on random braids with configurable experiment parameters."""

    # Init environment
    # env = gym.make('BandEnv-v0', band_decomposition=[1,2,-1], train_type="deterministic") # Learn to simplify a specific braid
    env = gym.make('BandEnv-v0', braid_index=braid_index, max_num_bands=max_num_bands, train_type="random") # Learn to simplify random braids
    action_size = env.unwrapped.max_num_actions
    state_size = env.unwrapped.get_state().size

    # Init networks
    policy_network = PolicyNetwork(state_size, action_size).to(device)
    value_network = ValueNetwork(state_size).to(device)

    # Init optimizer
    optim = torch.optim.Adam(chain(policy_network.parameters(), value_network.parameters()), lr=learning_rate)

    # Start main loop
    results_ppo = []
    policy_loss_ppo = []
    value_loss_ppo = []
    logs = []
    loop = tqdm(total=epochs, position=0, leave=False)
    for epoch in range(epochs):

        memory = []  # Reset memory every epoch
        rewards = []  # Calculate average episodic reward per epoch

        # Begin experience loop
        for episode in range(env_samples):
            # Reset environment
            state, _ = env.reset()
            done = False
            rollout = []
            cum_reward = 0  # Track cumulative reward
            num_actions_taken = 0

            # Begin episode
            while not done and num_actions_taken < max_actions:  # End after a given number of steps
                # Get action
                action, action_dist = get_action_ppo(policy_network, state)

                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store step
                rollout.append((state, action, action_dist, reward))

                cum_reward += reward
                state = next_state  # Set current state

                # increase num_actions_taken
                num_actions_taken += 1

            # Calculate returns and add episode to memory
            memory = calculate_return(memory, rollout, gamma)

            rewards.append(cum_reward)

        # Train
        dataset = RLDataset(memory)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        mean_policy_loss_item, mean_value_loss_item = learn_ppo(optim, policy_network, value_network, loader, epsilon, policy_epochs, action_size)
        policy_loss_ppo.append(mean_policy_loss_item)
        value_loss_ppo.append(mean_value_loss_item)

        # Print results
        num_bands = len(env.unwrapped.band_decomposition)
        results_ppo.extend(rewards)  # Store rewards for this epoch
        logs.extend([env.unwrapped.log])
        loop.update(1)
        loop.set_description("Epochs: {}   Reward: {}   Num Bands: {}  ".format(epoch, results_ppo[-1], num_bands))

        if save_path is not None and epoch > 0 and epoch % 1000 == 0:
            torch.save(policy_network.state_dict(), save_path)

    if save_path is not None:
        torch.save(policy_network.state_dict(), save_path)
    env.close()
    return results_ppo, policy_loss_ppo, value_loss_ppo, logs


def ppo_main_curriculum(epochs=500, env_samples=10, difficulties=(0, 1, 2, 3),
                        max_actions_per_stage=15,
                        save_path="./models/Braid_Simplificationator_2000", *,
                        braid_index=8, max_num_bands=80,
                        learning_rate=0.0008432777999828978,
                        gamma=0.9082237929205784, batch_size=256,
                        epsilon=0.16273153856100495, policy_epochs=5):
    """Same as PPO main but adjusted for curriculum learning."""

    # Curriculum stages

    # Initialize environment to get sizes
    env = gym.make('BandEnv-v0', braid_index=braid_index, max_num_bands=max_num_bands, train_type="curriculum", difficulty=difficulties[0])
    action_size = env.unwrapped.max_num_actions
    state_size = env.unwrapped.get_state().size

    # Initialize networks and optimizer once (shared across difficulties)
    policy_network = PolicyNetwork(state_size, action_size).to(device)
    value_network = ValueNetwork(state_size).to(device)
    optim = torch.optim.Adam(chain(policy_network.parameters(), value_network.parameters()), lr=learning_rate)

    # Logging 
    results_ppo = []
    policy_loss_ppo = []
    value_loss_ppo = []
    logs = []

    loop = tqdm(total=epochs * len(difficulties), position=0, leave=False)

    # Curriculum loop
    for difficulty in difficulties:
        print(f"Starting training on difficulty {difficulty}...")

        # Reinitialize environment with current difficulty
        env.close()
        env = gym.make('BandEnv-v0', braid_index=braid_index, max_num_bands=max_num_bands, train_type="curriculum", difficulty=difficulty)
        max_actions = max_actions_per_stage * (difficulty + 1)

        # Start main loop for this difficulty
        for epoch in range(epochs):

            memory = []  # Reset memory every epoch
            rewards = []  # Calculate average episodic reward per epoch

            # Begin experience loop
            for episode in range(env_samples):
                # Reset environment
                state, _ = env.reset()
                done = False
                rollout = []
                cum_reward = 0  # Track cumulative reward
                num_actions_taken = 0

                # Begin episode
                while not done and num_actions_taken < max_actions:  # End after a given number of steps
                    # Get action
                    action, action_dist = get_action_ppo(policy_network, state)

                    # Take step
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Store step
                    rollout.append((state, action, action_dist, reward))

                    cum_reward += reward
                    state = next_state  # Set current state

                    # increase num_actions_taken
                    num_actions_taken += 1

                # Calculate returns and add episode to memory
                memory = calculate_return(memory, rollout, gamma)
                rewards.append(cum_reward)

            # Train
            dataset = RLDataset(memory)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            mean_policy_loss_item, mean_value_loss_item = learn_ppo(
                optim, policy_network, value_network, loader, epsilon, policy_epochs, action_size
            )
            policy_loss_ppo.append(mean_policy_loss_item)
            value_loss_ppo.append(mean_value_loss_item)

            # Print results
            num_bands = len(env.unwrapped.band_decomposition)
            results_ppo.extend(rewards)  # Store rewards for this epoch
            logs.extend([env.unwrapped.log])
            loop.update(1)
            loop.set_description(
                f"Difficulty: {difficulty} | Epochs: {epoch} | Reward: {results_ppo[-1]} | Num Bands: {num_bands}"
            )

            # Periodic save
            if save_path is not None and epoch > 0 and epoch % 1000 == 0:
                torch.save(policy_network.state_dict(), save_path)

    # Final save
    if save_path is not None:
        torch.save(policy_network.state_dict(), save_path)
    env.close()

    return results_ppo, policy_loss_ppo, value_loss_ppo, logs


def ppo_single_braid(band_decomposition, braid_index, epochs=5000,
                     env_samples=10, max_actions=150, max_num_bands=80,
                     save_path="./models/Braid_Simplificationator_specific",
                     report_path="./logs/specific_training_report.json", *,
                     learning_rate=0.0008432777999828978,
                     gamma=0.9082237929205784, batch_size=256,
                     epsilon=0.16273153856100495, policy_epochs=5,
                     collect_logs=True, show_progress=True,
                     training_state=None, return_training_state=False):
    """Train PPO on one fixed braid and retain the best path found.

    Set collect_logs=False to skip retaining epoch environment logs, and
    show_progress=False for quiet batch training. Neither changes the best-path
    tracking or the returned report.
    """
    if band_decomposition is None:
        raise ValueError("band_decomposition must be set for specific-braid training")
    if len(band_decomposition) > max_num_bands:
        raise ValueError("max_num_bands cannot be smaller than the starting decomposition")

    env = gym.make(
        'BandEnv-v0',
        band_decomposition=band_decomposition,
        braid_index=braid_index,
        max_num_bands=max_num_bands,
        train_type="deterministic",
    )
    action_size = env.unwrapped.max_num_actions
    state_size = env.unwrapped.get_state().size

    policy_network = PolicyNetwork(state_size, action_size).to(device)
    value_network = ValueNetwork(state_size).to(device)
    optim = torch.optim.Adam(
        chain(policy_network.parameters(), value_network.parameters()), lr=learning_rate
    )
    if training_state is not None:
        policy_network.load_state_dict(training_state["policy"])
        value_network.load_state_dict(training_state["value"])
        optim.load_state_dict(training_state["optimizer"])

    env.reset()
    original_decomposition = copy.deepcopy(env.unwrapped.band_decomposition)
    best_decomposition = copy.deepcopy(original_decomposition)
    best_decomp_len = len(best_decomposition)
    best_action_sequence = []
    best_move_sequence = []
    best_location = None

    results_ppo = []
    policy_loss_ppo = []
    value_loss_ppo = []
    best_length_history = []
    logs = []

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    loop = tqdm(total=epochs, position=0, leave=False, disable=not show_progress)
    for epoch in range(epochs):
        memory = []
        rewards = []

        for episode in range(env_samples):
            state, _ = env.reset()
            done = False
            rollout = []
            episode_actions = []
            cum_reward = 0
            num_actions_taken = 0

            while not done and num_actions_taken < max_actions:
                action, action_dist = get_action_ppo(policy_network, state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                rollout.append((state, action, action_dist, reward))
                episode_actions.append(action)
                cum_reward += reward
                state = next_state
                num_actions_taken += 1

                current_decomp_len = len(env.unwrapped.band_decomposition)
                if current_decomp_len < best_decomp_len:
                    best_decomp_len = current_decomp_len
                    best_decomposition = copy.deepcopy(env.unwrapped.band_decomposition)
                    best_action_sequence = episode_actions.copy()
                    best_move_sequence = env.unwrapped.log["Moves"].copy()
                    best_location = {
                        "epoch": epoch,
                        "episode": episode,
                        "step": num_actions_taken,
                    }

            memory = calculate_return(memory, rollout, gamma)
            rewards.append(cum_reward)
            best_length_history.append(best_decomp_len)

        dataset = RLDataset(memory)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        mean_policy_loss_item, mean_value_loss_item = learn_ppo(
            optim, policy_network, value_network, loader, epsilon,
            policy_epochs, action_size
        )
        policy_loss_ppo.append(mean_policy_loss_item)
        value_loss_ppo.append(mean_value_loss_item)

        results_ppo.extend(rewards)
        if collect_logs:
            logs.append(copy.deepcopy(env.unwrapped.log))
        loop.update(1)
        loop.set_description(
            f"Epoch: {epoch} | Reward: {results_ppo[-1]} | Best length: {best_decomp_len}"
        )

        if save_path is not None and epoch > 0 and epoch % 1000 == 0:
            torch.save(policy_network.state_dict(), save_path)

    loop.close()
    if save_path is not None:
        torch.save(policy_network.state_dict(), save_path)

    report = {
        "initial_length": len(original_decomposition),
        "initial_decomposition": original_decomposition,
        "best_length": best_decomp_len,
        "best_decomposition": best_decomposition,
        "best_action_sequence": best_action_sequence,
        "best_move_sequence": best_move_sequence,
        "best_found_at": best_location,
        "best_length_history": best_length_history,
    }
    if report_path is not None:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    env.close()
    if show_progress:
        print(f"Initial decomposition length: {report['initial_length']}")
        print(f"Best decomposition length: {report['best_length']}")
        print(f"Best decomposition: {report['best_decomposition']}")

    result = (results_ppo, policy_loss_ppo, value_loss_ppo, logs, report)
    if return_training_state:
        return result + ({
            "policy": policy_network.state_dict(),
            "value": value_network.state_dict(),
            "optimizer": optim.state_dict(),
        },)
    return result


def run_inference(model_path="./models/Braid_Simplificationator_2000",
                  data_path="./data/braids_with_ranks.csv",
                  log_path="./logs/inference_logfile.txt",
                  forms_path="./logs/inference_final_forms.txt",
                  plot_path="./results/inference.png", *,
                  braid_index=8, max_num_bands=80, max_actions=150):
    """Evaluate a saved policy on all braids in the inference dataset."""
    braid_df = pd.read_csv(data_path)
    true_ranks = braid_df["Braid ranks"].values
    braid_words = braid_df["Braid word"]

    size_env = gym.make(
        'BandEnv-v0', braid_index=braid_index, max_num_bands=max_num_bands, train_type="random"
    )
    action_size = size_env.unwrapped.max_num_actions
    state_size = size_env.unwrapped.get_state().size
    size_env.close()

    model = PolicyNetwork(state_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(forms_path).parent.mkdir(parents=True, exist_ok=True)
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

    print("Inference...")
    final_decomp_lens = []
    best_decomp_lens = []
    simplest_forms = []
    initial_decomp_lens = []
    rewards = []

    for i in range(len(true_ranks)):
        initial_word = [int(sigma) for sigma in braid_words.iloc[i][1:-1].split(", ")]
        initial_decomp_lens.append(len(initial_word))

        env = gym.make(
            'BandEnv-v0', band_decomposition=initial_word, braid_index=braid_index,
            max_num_bands=max_num_bands, train_type="deterministic"
        )
        state, _ = env.reset()
        best_decomp_len = len(env.unwrapped.band_decomposition)
        simplest_form = copy.deepcopy(env.unwrapped.band_decomposition)

        file_mode = 'w' if i == 0 else 'a'
        with open(log_path, file_mode) as f:
            f.write(f"\nBraid {i}\n")

        with torch.no_grad():
            num_actions_taken = 0
            done = False
            cum_reward = 0
            while not done and num_actions_taken < max_actions:
                with open(log_path, 'a') as f:
                    f.write(str(env.unwrapped.band_decomposition) + "\n")

                action, action_dist = get_action_ppo(model, state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                cum_reward += reward
                state = next_state
                num_actions_taken += 1

                current_decomp_len = len(env.unwrapped.band_decomposition)
                if current_decomp_len <= best_decomp_len:
                    best_decomp_len = current_decomp_len
                    simplest_form = copy.deepcopy(env.unwrapped.band_decomposition)

        rewards.append(cum_reward)
        best_decomp_lens.append(best_decomp_len)
        simplest_forms.append(simplest_form)
        final_decomp_lens.append(len(env.unwrapped.band_decomposition))
        env.close()

    correct = sum(
        true_rank == best_length
        for true_rank, best_length in zip(true_ranks, best_decomp_lens)
    )
    print("Number completely simplified:", correct)

    with open(forms_path, 'w') as f:
        for item in simplest_forms:
            f.write(f"{item}\n")

    comparison = sorted(zip(true_ranks, initial_decomp_lens, best_decomp_lens))
    sorted_optimal, sorted_initial, sorted_best = zip(*comparison)

    plt.figure(figsize=(13, 4))
    x_vals = np.arange(1, len(comparison) + 1)
    plt.scatter(x_vals, sorted_optimal, label="True Rank")
    plt.scatter(x_vals, sorted_initial, color="green", label="Initial Band Decomp Length")
    plt.scatter(x_vals, sorted_best, marker="+", label="Best Band Decomp Length Achieved")
    plt.ylabel("Band Decomposition Length")
    plt.xlabel("Test Braid Identifier")
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    plt.close()

    return {
        "true_ranks": list(true_ranks),
        "initial_lengths": initial_decomp_lens,
        "best_lengths": best_decomp_lens,
        "final_lengths": final_decomp_lens,
        "simplest_forms": simplest_forms,
        "rewards": rewards,
    }



def _json_default(value):
    """Convert NumPy results to ordinary JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _save_run_json(path, value):
    with Path(path).open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, default=_json_default)
        output.write("\n")


def _resolve_experiment_config(config, name=None):
    """Fill in function defaults without allowing legacy output-path overrides."""
    mode = config.get("mode", "training")
    method = config.get("training_method", "regular")
    if mode == "training":
        if method not in {"regular", "curriculum"}:
            raise ValueError("training_method must be 'regular' or 'curriculum'")
        function = ppo_main if method == "regular" else ppo_main_curriculum
    elif mode == "specific_training":
        function = ppo_single_braid
    elif mode == "inference":
        function = run_inference
    else:
        raise ValueError("mode must be 'training', 'specific_training', or 'inference'")

    output_parameters = {"save_path", "report_path", "log_path", "forms_path", "plot_path"}
    parameters = {
        key: parameter for key, parameter in inspect.signature(function).parameters.items()
        if key not in output_parameters
    }
    metadata = {"name", "mode", "seed"}
    if mode == "training":
        metadata.add("training_method")
    if mode != "inference":
        metadata.update({"save_model", "model_name"})
    unknown = set(config) - set(parameters) - metadata
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")

    resolved = {
        "name": name if name is not None else config.get("name", mode),
        "mode": mode,
        "seed": config.get("seed"),
    }
    seed = resolved["seed"]
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("seed must be null or an integer from 0 through 2**32 - 1")
    if mode == "training":
        resolved["training_method"] = method
    for key, parameter in parameters.items():
        if key in config:
            resolved[key] = config[key]
        elif parameter.default is not inspect.Parameter.empty:
            resolved[key] = parameter.default
        else:
            raise ValueError(f"Missing required config field: {key}")
    if mode != "inference":
        resolved["save_model"] = config.get("save_model", True)
        if type(resolved["save_model"]) is not bool:
            raise ValueError("save_model must be true or false")
        default_model = Path(inspect.signature(function).parameters["save_path"].default).name
        model_name = config.get("model_name", default_model)
        reserved = {"config.json", "run_info.json", "results.json", "metrics.csv", "plot.png", "logs.json"}
        if (not isinstance(model_name, str) or not model_name
                or model_name in {".", ".."} or "/" in model_name or "\\" in model_name
                or model_name in reserved):
            raise ValueError("model_name must be a filename distinct from the run's other outputs")
        resolved["model_name"] = model_name
    return resolved, function, parameters


def _training_metrics(config, returns, policy_losses, value_losses, report=None):
    """Aggregate episode returns in training order, using a global zero-based epoch."""
    env_samples = config["env_samples"]
    curriculum = config.get("training_method") == "curriculum"
    num_epochs = config["epochs"] * (len(config["difficulties"]) if curriculum else 1)
    if env_samples <= 0:
        raise ValueError("env_samples must be positive to aggregate epoch metrics")
    if (len(policy_losses) != num_epochs or len(value_losses) != num_epochs
            or len(returns) != num_epochs * env_samples):
        raise ValueError("Training results do not match the configured epoch and episode counts")
    if report is not None and len(report["best_length_history"]) != len(returns):
        raise ValueError("Best-length history does not match the episode count")

    rows = []
    for epoch in range(num_epochs):
        start = epoch * env_samples
        end = start + env_samples
        row = {"epoch": epoch}
        if curriculum:
            row["difficulty"] = config["difficulties"][epoch // config["epochs"]]
        row.update({
            "mean_return": float(np.mean(returns[start:end])),
            "policy_loss": float(policy_losses[epoch]),
            "value_loss": float(value_losses[epoch]),
        })
        if report is not None:
            # This history tracks the best length so far after each episode.
            row["best_length"] = report["best_length_history"][end - 1]
        rows.append(row)
    return rows


def _training_summary(config, metrics, num_episodes, report=None):
    """Keep final results and the best path, leaving epoch histories in the CSV."""
    final = metrics[-1] if metrics else {}
    result = {
        "epochs_completed": len(metrics),
        "episodes_completed": num_episodes,
        "mean_return": float(np.mean([row["mean_return"] for row in metrics])) if metrics else None,
        "final_mean_return": final.get("mean_return"),
        "final_policy_loss": final.get("policy_loss"),
        "final_value_loss": final.get("value_loss"),
        "model_file": config["model_name"] if config["save_model"] else None,
    }
    if "difficulty" in final:
        result["final_difficulty"] = final["difficulty"]
    if report is not None:
        result.update({key: value for key, value in report.items() if key != "best_length_history"})
    return result


def _plot_training_metrics(config, metrics, plot_path):
    """Plot the same epoch values that are saved to metrics.csv."""
    specific = config["mode"] == "specific_training"
    epochs = [row["epoch"] for row in metrics]
    figure, axes = plt.subplots(2 if specific else 1, 1, figsize=(10, 7) if specific else (10, 4))
    axes = np.atleast_1d(axes)
    try:
        axes[0].plot(epochs, [row["mean_return"] for row in metrics])
        axes[0].set_ylabel("Mean return")
        axes[0].set_title(config["name"])
        if specific:
            axes[1].plot(epochs, [row["best_length"] for row in metrics])
            axes[1].set_ylabel("Best length found")
        for axis in axes:
            axis.set_xlabel("Epoch (zero-based, across all stages)")
            axis.grid()
        figure.tight_layout()
        figure.savefig(plot_path)
    finally:
        plt.close(figure)


def run_configured_experiment(config_path, name=None, runs_root="runs"):
    """Run a configured experiment, keeping all generated files in its run folder."""
    config, function, parameters = _resolve_experiment_config(load_config(config_path), name)
    run_directory = initialize_run(config, runs_root=runs_root)
    print(f"Run directory: {run_directory}")
    try:
        if config["seed"] is not None:
            random.seed(config["seed"])
            np.random.seed(config["seed"])
            torch.manual_seed(config["seed"])

        kwargs = {key: config[key] for key in parameters}
        if config["mode"] == "inference":
            result = function(
                **kwargs,
                log_path=run_directory / "inference_logfile.txt",
                forms_path=run_directory / "inference_final_forms.txt",
                plot_path=run_directory / "plot.png",
            )
            _save_run_json(run_directory / "results.json", result)
        else:
            kwargs["save_path"] = (
                run_directory / config["model_name"] if config["save_model"] else None
            )
            if config["mode"] == "specific_training":
                # The returned report is included in results.json instead of a separate file.
                kwargs["report_path"] = None
            training_result = function(**kwargs)
            returns, policy_losses, value_losses, _logs = training_result[:4]
            report = training_result[4] if config["mode"] == "specific_training" else None
            metrics = _training_metrics(config, returns, policy_losses, value_losses, report)
            result = _training_summary(config, metrics, len(returns), report)
            _save_run_json(run_directory / "results.json", result)
            save_metrics(metrics, run_directory / "metrics.csv")
            _plot_training_metrics(config, metrics, run_directory / "plot.png")
        update_run_status(run_directory, "completed")
    except BaseException as error:
        update_run_status(run_directory, "failed", error=f"{type(error).__name__}: {error}")
        raise
    return run_directory


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run a braid experiment from a JSON config.")
    parser.add_argument("--config", help="Path to an experiment JSON config")
    parser.add_argument("--name", help="Override the descriptive run name")
    args = parser.parse_args()
    if args.name is not None and args.config is None:
        parser.error("--name requires --config")
    if args.config is not None:
        run_configured_experiment(args.config, name=args.name)
        raise SystemExit(0)

    RUN_MODE = "specific_training"  # "training", "inference", or "specific_training"
    TRAINING_METHOD = "curriculum"  # "regular" or "curriculum" (when RUN_MODE=="training")

    # Settings used only when RUN_MODE == "specific_training".
    SPECIFIC_BRAID = [3, -3, 2, -3, 2, 1, 1, -2, 1, -2] 
    SPECIFIC_BRAID_INDEX = 5
    SPECIFIC_MAX_NUM_BANDS = 40
    SPECIFIC_EPOCHS = 500
    SPECIFIC_ENV_SAMPLES = 10
    SPECIFIC_MAX_ACTIONS = 150
    SPECIFIC_MODEL_PATH = "./models/Braid_Simplificationator_specific"
    SPECIFIC_REPORT_PATH = "./logs/specific_training_report.json"
    SPECIFIC_PLOT_PATH = "./results/specific_training.png"

    if RUN_MODE == "training":
        print(f"Training model with the {TRAINING_METHOD} method...")
        if TRAINING_METHOD == "regular":
            results_ppo, policy_loss_ppo, value_loss_ppo, logs = ppo_main()
        elif TRAINING_METHOD == "curriculum":
            results_ppo, policy_loss_ppo, value_loss_ppo, logs = ppo_main_curriculum()
        else:
            raise ValueError("TRAINING_METHOD must be 'regular' or 'curriculum'")

        with open('./logs/logs.json', 'w') as f:
            json.dump(logs, f)

        plt.figure()
        plt.plot(results_ppo)
        plt.title("Number of Bands Removed at each Training Episode")
        plt.ylabel("Return")
        plt.xlabel("Episode")
        plt.savefig("./results/training.png")
        plt.close()
        print("Training complete.")

    elif RUN_MODE == "inference":
        run_inference()

    elif RUN_MODE == "specific_training":
        if SPECIFIC_BRAID is None:
            raise ValueError(
                "Set SPECIFIC_BRAID before running specific-braid training"
            )

        results_ppo, policy_loss_ppo, value_loss_ppo, logs, report = ppo_single_braid(
            band_decomposition=SPECIFIC_BRAID,
            braid_index=SPECIFIC_BRAID_INDEX,
            epochs=SPECIFIC_EPOCHS,
            env_samples=SPECIFIC_ENV_SAMPLES,
            max_actions=SPECIFIC_MAX_ACTIONS,
            max_num_bands=SPECIFIC_MAX_NUM_BANDS,
            save_path=SPECIFIC_MODEL_PATH,
            report_path=SPECIFIC_REPORT_PATH,
        )

        Path(SPECIFIC_PLOT_PATH).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 1, 1)
        plt.plot(results_ppo)
        plt.ylabel("Return")
        plt.title("Specific-braid training")
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(report["best_length_history"])
        plt.ylabel("Best length found")
        plt.xlabel("Episode")
        plt.grid()
        plt.tight_layout()
        plt.savefig(SPECIFIC_PLOT_PATH)
        plt.close()

    else:
        raise ValueError(
            "RUN_MODE must be 'training', 'inference', or 'specific_training'"
        )
