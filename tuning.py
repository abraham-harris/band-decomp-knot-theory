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
from IPython.core.interactiveshell import InteractiveShell
from generator import RandomBraid
from band_env import BandEnv
import optuna
import optuna.visualization as vis

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



### PPO ALGORITHM HELPER FUNCTIONS ###
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


### HYPERPARAMETER OPTIMIZATION ###
def optimize_ppo(trial):
    """
    Hyperparameter optimization for braid simplifying PPO algorithm.
    
    Returns a score for an Optuna study to maximize.
    """
    ### Define search space (suggest hyperparameters) ###
    lr = trial.suggest_float('lr', 1e-5, 3e-3, log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [8, 16, 32])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.88, 0.99)
    epsilon = trial.suggest_float('epsilon', 0.1, 0.3)

    ### Functions for PPO ###
    # Policy Network
    class PolicyNetwork(nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()

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

    ### Create and train PPO model ###
    def ppo_main():
        # Hyper parameters that we aren't tuning
        epochs = 100
        env_samples = 5  # episodes per epoch
        policy_epochs = 5
        max_actions = 100

        # Init environment
        # env = gym.make('BandEnv-v0', band_decomposition=[1,2,-1]) # Learn to simplify a specific band
        env = gym.make('BandEnv-v0', braid_index=8, max_num_bands=80, random=True) # Learn to simplify random bands
        action_size = env.unwrapped.max_num_actions
        state_size = env.unwrapped.get_state().size

        # Init networks
        policy_network = PolicyNetwork(state_size, action_size).to(device)
        value_network = ValueNetwork(state_size).to(device)

        # Init optimizer
        optim = torch.optim.Adam(chain(policy_network.parameters(), value_network.parameters()), lr=lr)

        # Start main loop
        results_ppo = []
        policy_loss_ppo = []
        value_loss_ppo = []
        logs = []
        # loop = tqdm(total=epochs, position=0, leave=False)
        for epoch in range(epochs):

            memory = []  # Reset memory every epoch
            rewards = []  # Calculate average episodic reward per epoch

            # Begin experience loop
            for episode in range(env_samples):
                # Reset environment
                state = env.reset()
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
            # loop.update(1)
            # loop.set_description("Epochs: {}   Reward: {}   Num Bands: {}  ".format(epoch, results_ppo[-1], num_bands))

        return results_ppo, policy_loss_ppo, value_loss_ppo, logs

    results_ppo, policy_loss_ppo, value_loss_ppo, logs = ppo_main()
    # Get average score from last 5 epochs
    av_score = np.mean(results_ppo[-25:])
    return av_score



if __name__ == "__main__":
    # Run Optuna study
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(optimize_ppo, n_trials=100, n_jobs=1)
    print("Best parameters:", study.best_params)

    # Visualizations
    fig1 = vis.plot_optimization_history(study)
    fig2 = vis.plot_param_importances(study)
    fig1.write_html("results/optimization_history.html")
    fig2.write_html("results/param_importances.html")