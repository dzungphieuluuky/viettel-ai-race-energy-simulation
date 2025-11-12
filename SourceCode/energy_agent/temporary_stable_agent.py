import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import logging
import os
import zipfile
import pickle

# ===================================================================
# This script is now self-contained but requires the trained_model folder.
# The original helper files (models.py, transition.py) are no longer used.
# ===================================================================


# --- 1. Define a Network Architecture that EXACTLY Matches the SB3 Model ---
# SB3's MlpPolicy consists of a features_extractor and an mlp_extractor.
# We combine them here into a single, shared network body.
class ActorCriticPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, features_dim=256):
        super(ActorCriticPolicy, self).__init__()

        # This network mimics the SB3 'CustomNetwork' features extractor.
        # It's the shared body of the network.
        self.features_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, features_dim),
            nn.Tanh()
        )

        # In SB3, the mlp_extractor is usually empty if a custom features_extractor is used.
        # The features are fed directly to the action and value heads.

        # This mimics the SB3 'action_net' (the final layer for the actor).
        self.action_net = nn.Linear(features_dim, action_dim)
        
        # This mimics the SB3 'value_net' (the final layer for the critic).
        self.value_net = nn.Linear(features_dim, 1)

        # PPO's policy is a Gaussian distribution, which needs a standard deviation.
        # This is a learnable parameter, just like in SB3.
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """
        Performs a forward pass through the network.
        Returns the action distribution and the predicted state value.
        """
        features = self.features_extractor(state)
        
        # Actor head
        action_mean = self.action_net(features)
        action_std = torch.exp(self.log_std)
        action_dist = Normal(action_mean, action_std)
        
        # Critic head
        value_pred = self.value_net(features)
        
        return action_dist, value_pred


class RLAgent:
    def __init__(self, n_cells, n_ues, max_time, log_file='rl_agent.log', use_gpu=False):
        """
        Initializes a PPO agent by loading pre-trained weights from an SB3 model.
        It also re-implements the PPO training logic for potential fine-tuning.
        """
        print("Initializing RL Agent from pre-trained SB3 weights.")
        self.n_cells = n_cells
        self.setup_logging(log_file)
        
        # --- Device Selection ---
        self.device = self._get_best_device()
        
        # --- State and Action Dimensions ---
        self.state_dim = 17 + 14 + (n_cells * 12)
        self.action_dim = n_cells
        
        # --- Load Normalization Statistics ---
        self._load_normalization_stats()

        # --- Instantiate and Load the Policy Network ---
        self.policy = ActorCriticPolicy(self.state_dim, self.action_dim).to(self.device)
        self._load_sb3_weights()

        # --- Re-implement SB3 Hyperparameters for Fine-Tuning ---
        self.learning_rate = 1e-4
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.clip_range = 0.2
        self.ppo_epochs = 10
        self.batch_size = 128
        self.ent_coef = 0.01
        self.vf_coef = 0.5

        # --- Experience Buffer (Simplified) ---
        self.buffer = []
        self.training_mode = True # Set to False if you ONLY want inference
        self.logger.info(f"RL Agent initialized on device: {self.device}")

    def _get_best_device(self):
        """Helper to detect the best available hardware."""
        if hasattr(torch, 'has_dml') and torch.has_dml and torch.dml.is_available():
            return torch.device("dml")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_normalization_stats(self):
        """Loads the mean and variance from the vec_normalize.pkl file."""
        stats_path = os.path.join(os.path.dirname(__file__), "trained_model", "vec_normalize.pkl")
        try:
            with open(stats_path, "rb") as f:
                vec_normalize_stats = pickle.load(f)
            self.running_mean = vec_normalize_stats.obs_rms.mean
            self.running_var = vec_normalize_stats.obs_rms.var
            self.epsilon = 1e-8
            self.logger.info("Successfully loaded observation normalization stats.")
        except FileNotFoundError:
            self.logger.error(f"FATAL: Normalization file not found at {stats_path}.")
            raise

    def _load_sb3_weights(self):
        """Extracts and loads the policy weights from the SB3 .zip file."""
        model_path = os.path.join(os.path.dirname(__file__), "trained_model", "ppo_5g_model.zip")
        try:
            with zipfile.ZipFile(model_path, 'r') as archive:
                # The weights are stored in 'policy.pth' inside the zip
                with archive.open('policy.pth', 'r') as f:
                    sb3_state_dict = torch.load(f, map_location=self.device)

            # Manually map the SB3 state_dict keys to our new network structure
            # This requires knowing the internal names SB3 uses.
            agent_state_dict = self.policy.state_dict()
            
            # Map features extractor weights
            agent_state_dict['features_extractor.0.weight'].copy_(sb3_state_dict['features_extractor.shared_net.0.weight'])
            agent_state_dict['features_extractor.0.bias'].copy_(sb3_state_dict['features_extractor.shared_net.0.bias'])
            agent_state_dict['features_extractor.2.weight'].copy_(sb3_state_dict['features_extractor.shared_net.2.weight'])
            agent_state_dict['features_extractor.2.bias'].copy_(sb3_state_dict['features_extractor.shared_net.2.bias'])

            # Map action and value head weights
            agent_state_dict['action_net.weight'].copy_(sb3_state_dict['action_net.weight'])
            agent_state_dict['action_net.bias'].copy_(sb3_state_dict['action_net.bias'])
            agent_state_dict['value_net.weight'].copy_(sb3_state_dict['value_net.weight'])
            agent_state_dict['value_net.bias'].copy_(sb3_state_dict['value_net.bias'])

            # Map the learned log_std
            agent_state_dict['log_std'].copy_(sb3_state_dict['log_std'])

            self.policy.load_state_dict(agent_state_dict)
            self.logger.info("Successfully loaded pre-trained weights into the policy network.")

        except FileNotFoundError:
            self.logger.error(f"FATAL: Trained model file not found at {model_path}.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model weights: {e}")
            raise

    def normalize_state(self, state):
        """Normalizes the state using the loaded running mean and variance."""
        state = np.array(state).flatten()
        return (state - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

    def setup_logging(self, log_file):
        """Setup logging configuration."""
        self.logger = logging.getLogger('PPOAgent_FineTune')
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())

    def start_scenario(self):
        """Called at the start of a scenario."""
        self.buffer.clear() # Clear buffer for the new scenario
        self.logger.info("New scenario started, buffer cleared.")
    
    def end_scenario(self):
        """Called at the end of a scenario. Triggers training."""
        self.logger.info("Scenario ended.")
        if self.training_mode and len(self.buffer) > 0:
            self.train()

    def get_action(self, state):
        """Gets an action from the policy network for a given state."""
        normalized_state = self.normalize_state(state)
        state_tensor = torch.tensor(normalized_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_dist, value = self.policy(state_tensor)
            
            if self.training_mode:
                action = action_dist.sample()
            else:
                action = action_dist.mean

        # Store information needed for the 'update' step
        self.last_state = normalized_state
        self.last_action = action.cpu().numpy().flatten()
        self.last_log_prob = action_dist.log_prob(action).sum().cpu().item()
        self.last_value = value.cpu().item()

        return self.last_action

    def calculate_reward(self, prev_state, action, current_state):
        """Calculates reward. This should mirror the logic from your training environment."""
        # This is a placeholder. You should copy your final, refined reward function here.
        # For now, let's use a very simple one.
        raw_prev_state = np.array(prev_state).flatten()
        raw_current_state = np.array(current_state).flatten()

        prev_energy = raw_prev_state[17] # Assuming totalEnergy is the 18th feature
        current_energy = raw_current_state[17]
        
        # Simple energy saving reward + penalty for doing nothing
        reward = (prev_energy - current_energy) * 10 - 0.1
        return reward

    def update(self, state, action, next_state, done):
        """Stores a transition in the experience buffer."""
        if not self.training_mode:
            return

        reward = self.calculate_reward(state, action, next_state)
        normalized_next_state = self.normalize_state(next_state)

        # Store all necessary information for GAE and PPO loss calculation.
        self.buffer.append((
            self.last_state, self.last_action, reward,
            normalized_next_state, done, self.last_log_prob, self.last_value
        ))

    def compute_gae(self, rewards, values, next_values, dones):
        """Computes Generalized Advantage Estimation, mirroring SB3's logic."""
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        return advantages, returns

    def train(self):
        """Performs a PPO update, mirroring the logic of Stable Baselines3."""
        self.logger.info(f"Starting training with {len(self.buffer)} samples.")
        self.policy.train()

        # 1. Unpack buffer data
        states, actions, rewards, next_states, dones, old_log_probs, values = map(np.array, zip(*self.buffer))
        
        # 2. Compute next values and GAE
        with torch.no_grad():
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            _, next_values_tensor = self.policy(next_states_tensor)
            next_values = next_values_tensor.cpu().numpy().flatten()
        
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Convert to Tensors for training
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # 4. PPO training loop
        for _ in range(self.ppo_epochs):
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)
            for start in range(0, len(self.buffer), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get new policy outputs for the minibatch
                action_dist, value_preds = self.policy(states_tensor[batch_indices])
                entropy = action_dist.entropy().mean()
                new_log_probs = action_dist.log_prob(actions_tensor[batch_indices]).sum(axis=-1)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - old_log_probs_tensor[batch_indices])
                policy_loss_1 = advantages_tensor[batch_indices] * ratio
                policy_loss_2 = advantages_tensor[batch_indices] * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(returns_tensor[batch_indices], value_preds.flatten())

                # Total loss
                loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.buffer.clear()
        self.policy.eval()
        self.logger.info("Training complete.")