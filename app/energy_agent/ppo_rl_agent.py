import numpy as np
import torch
import torch.nn as nn
import logging
import os
import pickle
import glob
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from collections import deque

# --- Stable Baselines3 Imports ---
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from .state_normalizer import StateNormalizer 

# ==============================================================================
# === STATE FEATURE INDICES ===
# ==============================================================================
TOTAL_CELLS           = 0
TOTAL_UES             = 1
SIM_TIME              = 2
TIME_STEP             = 3
TIME_PROGRESS         = 4
CARRIER_FREQUENCY     = 5
ISD                   = 6
MIN_TX_POWER          = 7
MAX_TX_POWER          = 8
BASE_POWER            = 9
IDLE_POWER            = 10
DROP_CALL_THRESHOLD   = 11
LATENCY_THRESHOLD     = 12
CPU_THRESHOLD         = 13
PRB_THRESHOLD         = 14
TRAFFIC_LAMBDA        = 15
PEAK_HOUR_MULTIPLIER  = 16

TOTAL_ENERGY          = 17
ACTIVE_CELLS          = 18
AVG_DROP_RATE         = 19
AVG_LATENCY           = 20
TOTAL_TRAFFIC         = 21
CONNECTED_UES         = 22
CONNECTION_RATE       = 23
CPU_VIOLATIONS        = 24
PRB_VIOLATIONS        = 25
MAX_CPU_USAGE         = 26
MAX_PRB_USAGE         = 27
KPI_VIOLATIONS        = 28
TOTAL_TX_POWER        = 29
AVG_POWER_RATIO       = 30

TOTAL_FEATURES        = 31

# ==============================================================================
# === CUSTOM NETWORK ===
# ==============================================================================
class LightweightAttentionNetwork(BaseFeaturesExtractor):
    """Lightweight attention-based feature extractor for cell features."""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, 
                 max_cells: int = 57, n_cell_features: int = 12):
        super().__init__(observation_space, features_dim)
        self.max_cells, self.n_cell_features = max_cells, n_cell_features
        self.global_features_dim = 17 + 14
        
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_features_dim, 128), nn.ReLU(), 
            nn.Linear(128, 64), nn.ReLU()
        )
        self.cell_embedding = nn.Linear(self.n_cell_features, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.final_mlp = nn.Sequential(
            nn.Linear(64 + 128, features_dim), nn.ReLU(), 
            nn.Linear(features_dim, features_dim), nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        global_features = observations[:, :self.global_features_dim]
        cell_features_flat = observations[:, self.global_features_dim:]
        batch_size = observations.shape[0]
        
        cell_features_interleaved = cell_features_flat.view(batch_size, self.n_cell_features, self.max_cells)
        cell_features_structured = cell_features_interleaved.permute(0, 2, 1)

        processed_global = self.global_mlp(global_features)
        embedded_cells = self.cell_embedding(cell_features_structured)
        attn_output, _ = self.attention(embedded_cells, embedded_cells, embedded_cells)
        pooled_cells = attn_output.mean(dim=1)
        
        combined_features = torch.cat([processed_global, pooled_cells], dim=1)
        return self.final_mlp(combined_features)


# ==============================================================================
# === CURRICULUM LEARNING CALLBACK ===
# ==============================================================================
class CurriculumCallback(BaseCallback):
    """
    Implements curriculum learning with phases:
    1. Constraint satisfaction focus
    2. Balanced optimization
    3. Energy optimization focus
    """
    def __init__(self, agent, phase_episodes=[500, 1000, 2000], verbose=1):
        super().__init__(verbose)
        self.agent = agent
        self.phase_episodes = phase_episodes
        self.current_phase = 0
        
    def _on_step(self) -> bool:
        episode = self.agent.total_episodes
        
        # Phase transitions
        if self.current_phase < len(self.phase_episodes) - 1:
            if episode >= self.phase_episodes[self.current_phase]:
                self.current_phase += 1
                self.agent.update_curriculum_phase(self.current_phase)
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"CURRICULUM PHASE {self.current_phase} STARTED at episode {episode}")
                    print(f"{'='*60}\n")
        
        return True


# ==============================================================================
# === ENHANCED RL AGENT WITH CURRICULUM LEARNING ===
# ==============================================================================

class RLAgent:
    """
    Enhanced PPO Agent with:
    - Curriculum learning (constraint satisfaction -> energy optimization)
    - Adaptive reward shaping
    - Constraint violation tracking
    - Progressive training strategy
    """
    
    def __init__(self, 
                 n_cells: int,
                 max_time: int,
                 n_ues: int,
                 use_gpu: bool = False,
                 model_path: Optional[str] = None, 
                 training_mode: bool = True,
                 log_file: str = 'rl_agent.log'):
        
        self.max_cells = 57
        self.n_cells = n_cells
        self.max_time = max_time
        self.n_ues = n_ues
        self.training_mode = training_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
        # --- NEW: Define persistent storage paths ---
        self.model_dir = "online_trained_models"
        self.buffer_path = os.path.join(self.model_dir, "rollout_buffer.pkl")
        os.makedirs(self.model_dir, exist_ok=True) # Ensure directory exists


        # === CURRICULUM LEARNING PHASES ===
        self.curriculum_phase = 0  # 0: Constraints, 1: Balanced, 2: Energy
        self.phase_names = ["CONSTRAINT_FOCUS", "BALANCED", "ENERGY_OPTIMIZATION"]
        
        # === HYPERPARAMETERS (Optimized for constraint satisfaction) ===
        self.n_steps = 2048  # Collect 2048 steps before update
        self.batch_size = 128  # Larger batch for stable gradients
        self.n_epochs = 20  # More epochs for better learning
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.learning_rate = 3e-4  # Standard PPO learning rate
        self.clip_range = 0.2  # PPO clip range
        self.ent_coef = 0.01  # Entropy coefficient (exploration)
        self.vf_coef = 0.5  # Value function coefficient
        
        # === ADAPTIVE REWARD WEIGHTS (Phase-dependent) ===
        self.reward_weights = {
            'drop_call': 50.0,      # Highest priority
            'latency': 30.0,        # High priority
            'cpu_violation': 20.0,  # High priority
            'prb_violation': 20.0,  # High priority
            'connection': 15.0,     # Medium priority
            'energy': 1.0,          # Low initially, increases in later phases
            'stability': 5.0        # Encourage stable actions
        }
        
        log_mode = "TRAINING" if self.training_mode else "INFERENCE"
        print(f"\n{'='*70}")
        print(f"Initializing RL Agent in {log_mode} MODE")
        print(f"Max Cells: {self.max_cells} | Active Cells: {n_cells} | UEs: {n_ues}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        # === STATE & ACTION SPACES ===
        self.state_dim = 17 + 14 + (self.max_cells * 12)
        self.action_dim = self.max_cells
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)
        action_space = spaces.Box(low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.state_normalizer = StateNormalizer(state_dim=self.state_dim, n_cells=self.max_cells)

        # === MODIFICATION: Auto-load latest model and buffer ===
        latest_model_path, found_buffer_path = self._find_latest_model_and_buffer()

        # === LOAD OR CREATE MODEL ===
        if latest_model_path:
            print(f"Found latest model: {latest_model_path}")
            custom_objects = {"policy": {"features_extractor_class": LightweightAttentionNetwork}}
            self.model = PPO.load(latest_model_path, device=self.device, custom_objects=custom_objects)
            print("Model loaded successfully.")
        else:
            print("Creating a new PPO model from scratch...")
            policy_kwargs = {
                'features_extractor_class': LightweightAttentionNetwork,
                'features_extractor_kwargs': {
                    'features_dim': 256, 
                    'max_cells': self.max_cells, 
                    'n_cell_features': 12
                },
                'net_arch': dict(pi=[256, 128], vf=[256, 128])
            }
            
            self.model = PPO(
                policy=ActorCriticPolicy, 
                env=None, 
                policy_kwargs=policy_kwargs, 
                device=self.device,
                n_steps=self.n_steps, 
                batch_size=self.batch_size, 
                n_epochs=self.n_epochs, 
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                learning_rate=self.learning_rate,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                verbose=1, 
                _init_setup_model=False
            )

            self.model.observation_space = obs_space
            self.model.action_space = action_space
            self.model.n_envs = 1
            self.model._setup_model()
            print("PPO model created and set up correctly.")

        # === FIX: Manually configure and set the logger for the model ===
        # This will create log files (stdout, csv, tensorboard) in your model directory.
        sb3_logger = configure(self.model_dir, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(sb3_logger)

        # === ROLLOUT BUFFER FOR TRAINING ===
        # === MODIFICATION: Load persistent buffer or create new one ===
        self.buffer = None
        if self.training_mode:
            if found_buffer_path:
                try:
                    with open(found_buffer_path, 'rb') as f:
                        self.buffer = pickle.load(f)
                    print(f"Successfully loaded persistent buffer. Size: {self.buffer.pos}/{self.buffer.buffer_size}")
                except (pickle.UnpicklingError, EOFError) as e:
                    print(f"Warning: Could not load buffer file ({e}). Creating a new one.")
                    self.buffer = RolloutBuffer(self.n_steps, obs_space, action_space, self.device,
                                                gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=1)
            else:
                print("No persistent buffer found. Creating a new one.")
                self.buffer = RolloutBuffer(self.n_steps, obs_space, action_space, self.device,
                                            gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=1)
            
            # CRITICAL: Always associate the buffer with the loaded/created model
            self.model.rollout_buffer = self.buffer

        # === LOGGING ===
        self.setup_logging(log_file)
        
        # === EPISODE TRACKING ===
        self.total_episodes = 0
        self.episode_reward = 0.0
        self.last_obs = None
        self.last_done = True
        self.last_action_details = None
        self.prev_state = None
        
        # === CONSTRAINT VIOLATION TRACKING ===
        self.episode_violations = {
            'drop_calls': 0,
            'latency': 0,
            'cpu': 0,
            'prb': 0,
            'total': 0
        }
        self.violation_history = deque(maxlen=100)  # Last 100 episodes
        
        # === PERFORMANCE METRICS ===
        self.best_constraint_score = float('-inf')
        self.best_energy_efficiency = float('inf')
        self.episodes_without_violations = 0
        
        print(f"\nAgent initialization complete - Phase: {self.phase_names[self.curriculum_phase]}\n")

    def _prepare_observation(self, state: np.ndarray) -> np.ndarray:
        """Helper to consistently flatten, pad, and normalize a state vector."""
        flat_state = np.array(state).flatten()
        
        current_state_len = len(flat_state)
        if current_state_len < self.state_dim:
            padded_state = np.zeros(self.state_dim, dtype=np.float32)
            padded_state[:current_state_len] = flat_state
            state_to_use = padded_state
        else:
            state_to_use = flat_state
            
        return self.state_normalizer.normalize(state_to_use)

    def setup_logging(self, log_file: str):
        """Setup logging configuration."""
        self.logger = logging.getLogger('RLAgent')
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): 
            self.logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def update_curriculum_phase(self, phase: int):
        """Update curriculum phase and adjust reward weights."""
        self.curriculum_phase = phase
        
        if phase == 0:  # Constraint focus
            self.reward_weights['energy'] = 0.5
            self.reward_weights['drop_call'] = 50.0
            self.reward_weights['latency'] = 30.0
            self.ent_coef = 0.02  # Higher exploration
            
        elif phase == 1:  # Balanced
            self.reward_weights['energy'] = 5.0
            self.reward_weights['drop_call'] = 40.0
            self.reward_weights['latency'] = 25.0
            self.ent_coef = 0.01
            
        elif phase == 2:  # Energy optimization
            self.reward_weights['energy'] = 15.0
            self.reward_weights['drop_call'] = 30.0
            self.reward_weights['latency'] = 20.0
            self.ent_coef = 0.005  # Lower exploration
        
        # Update model's entropy coefficient
        self.model.ent_coef = self.ent_coef
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"PHASE TRANSITION: {self.phase_names[phase]}")
        self.logger.info(f"Reward Weights: {self.reward_weights}")
        self.logger.info(f"Entropy Coefficient: {self.ent_coef}")
        self.logger.info(f"{'='*60}\n")

    def _train(self, last_next_state: np.ndarray, last_done: bool):
        """Encapsulated training logic, called when the buffer is full."""
        self.logger.info(f"Rollout buffer is full ({self.buffer.pos}/{self.buffer.buffer_size}). Starting PPO training...")
        
        with torch.no_grad():
            # The "last state" for GAE calculation is the next_state of the transition that filled the buffer.
            prepared_last_obs = self._prepare_observation(last_next_state)
            obs_tensor = torch.as_tensor(prepared_last_obs, device=self.device).unsqueeze(0)
            last_value = self.model.policy.predict_values(obs_tensor)

        self.buffer.compute_returns_and_advantage(last_values=last_value, dones=np.array([last_done]))
        self.logger.info("Computed returns and advantages for the rollout buffer.")
        self.model.train()
        self.logger.info("Training complete.")
        
        self.buffer.reset() # Reset buffer immediately after training
        self.logger.info("Rollout buffer reset for next collection phase.")
        self.save_model(suffix=f"trained_step{self.buffer.pos}") # Save the newly trained model


    def start_scenario(self):
        """Reset episode tracking at scenario start."""
        self.total_episodes += 1
        self.episode_reward = 0.0
        self.prev_state = None
        self.episode_violations = {
            'drop_calls': 0,
            'latency': 0,
            'cpu': 0,
            'prb': 0,
            'total': 0
        }
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Episode {self.total_episodes} - Phase: {self.phase_names[self.curriculum_phase]}")
        self.logger.info(f"{'='*50}")

    def end_scenario(self):
        """Handle scenario end, training, and checkpointing."""
        # Track violations
        total_violations = self.episode_violations['total']
        self.violation_history.append(total_violations)
        
        if total_violations == 0:
            self.episodes_without_violations += 1
        else:
            self.episodes_without_violations = 0
        
        # Calculate metrics
        avg_violations = np.mean(self.violation_history) if self.violation_history else 0
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Episode {self.total_episodes} COMPLETED")
        self.logger.info(f"Total Reward: {self.episode_reward:.2f}")
        self.logger.info(f"Violations: {self.episode_violations}")
        self.logger.info(f"Avg Violations (last 100): {avg_violations:.2f}")
        self.logger.info(f"Consecutive Violation-Free: {self.episodes_without_violations}")
        self.logger.info(f"{'='*50}\n")
        
        self.last_done = True

        if self.training_mode and self.buffer:
            try:
                with open(self.buffer_path, 'wb') as f:
                    pickle.dump(self.buffer, f)
                self.logger.info(f"Persistent buffer saved. Size: {self.buffer.pos}/{self.buffer.buffer_size}")
            except Exception as e:
                self.logger.error(f"Failed to save persistent buffer: {e}")
        
        # === CHECKPOINTING ===
        if self.training_mode:
            self.save_model(suffix="scenario_end")
            # Save best models
            if total_violations == 0 and self.episode_reward > self.best_constraint_score:
                self.best_constraint_score = self.episode_reward
                self.save_model(suffix="best_constraints")
                self.logger.info("New best constraint-satisfying model saved!")            

    def calculate_reward(self, prev_state: np.ndarray, current_state: np.ndarray) -> float:
        """
        Calculate reward with adaptive weights based on curriculum phase.
        Prioritizes constraint satisfaction before energy optimization.
        """
        if prev_state is None:
            return 0.0
        
        reward = 0.0
        
        # === CONSTRAINT PENALTIES ===
        
        # 1. Drop Call Rate (CRITICAL)
        drop_rate = current_state[AVG_DROP_RATE]
        drop_threshold = current_state[DROP_CALL_THRESHOLD]
        if drop_rate > drop_threshold:
            drop_penalty = (drop_rate - drop_threshold) * self.reward_weights['drop_call']
            reward -= drop_penalty
            self.episode_violations['drop_calls'] += 1
        else:
            # Reward for staying below threshold
            reward += (drop_threshold - drop_rate) * self.reward_weights['drop_call'] * 0.1
        
        # 2. Latency (CRITICAL)
        latency = current_state[AVG_LATENCY]
        latency_threshold = current_state[LATENCY_THRESHOLD]
        if latency > latency_threshold:
            latency_penalty = (latency - latency_threshold) * self.reward_weights['latency']
            reward -= latency_penalty
            self.episode_violations['latency'] += 1
        else:
            reward += (latency_threshold - latency) * self.reward_weights['latency'] * 0.1
        
        # 3. CPU Violations
        cpu_violations = current_state[CPU_VIOLATIONS]
        if cpu_violations > 0:
            reward -= cpu_violations * self.reward_weights['cpu_violation']
            self.episode_violations['cpu'] += int(cpu_violations)
        
        # 4. PRB Violations
        prb_violations = current_state[PRB_VIOLATIONS]
        if prb_violations > 0:
            reward -= prb_violations * self.reward_weights['prb_violation']
            self.episode_violations['prb'] += int(prb_violations)
        
        # 5. Connection Rate
        connection_rate = current_state[CONNECTION_RATE]
        if connection_rate < 0.95:  # Expect >95% connection rate
            reward -= (0.95 - connection_rate) * 100 * self.reward_weights['connection']
        
        # === ENERGY OPTIMIZATION (Secondary) ===
        prev_energy = prev_state[TOTAL_ENERGY]
        current_energy = current_state[TOTAL_ENERGY]
        energy_saved = prev_energy - current_energy
        
        if energy_saved > 0:
            reward += energy_saved * self.reward_weights['energy']
        
        # === STABILITY BONUS ===
        # Penalize large power changes (encourage smooth control)
        prev_power = prev_state[TOTAL_TX_POWER]
        current_power = current_state[TOTAL_TX_POWER]
        power_change = abs(current_power - prev_power)
        if power_change > 0.1:  # Threshold for "large" change
            reward -= power_change * self.reward_weights['stability']
        
        # === TOTAL VIOLATIONS TRACKING ===
        self.episode_violations['total'] = (
            self.episode_violations['drop_calls'] +
            self.episode_violations['latency'] +
            self.episode_violations['cpu'] +
            self.episode_violations['prb']
        )
        
        # === CONSTRAINT SATISFACTION BONUS ===
        # Big bonus for maintaining all constraints
        if self.episode_violations['total'] == 0:
            reward += 10.0 * (1.0 + self.curriculum_phase)  # Increases with phase
        
        return float(np.clip(reward, -1000, 1000))

    def get_action(self, state: np.ndarray) -> np.ndarray:
        # Use the helper function for clean observation preparation
        normalized_obs = self._prepare_observation(state)
        self.last_obs = normalized_obs

        if self.training_mode:
            obs_tensor = torch.as_tensor(normalized_obs, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, value, log_prob = self.model.policy(obs_tensor)
            self.last_action_details = (action, value, log_prob)
            return np.clip(action.cpu().numpy().flatten()[:self.n_cells], 0.0, 1.0)
        else:
            action, _ = self.model.predict(normalized_obs, deterministic=True)
            return np.clip(action[:self.n_cells], 0.0, 1.0)

    def update(self, state: np.ndarray, action: np.ndarray, 
               next_state: np.ndarray, done: bool):
        """Update buffer and trigger training immediately if the buffer is full."""
        if not self.training_mode:
            return

        reward = self.calculate_reward(self.prev_state, next_state)
        self.episode_reward += reward
        
        if self.last_action_details is not None:
            _action, value, log_prob = self.last_action_details
            
            if self.buffer.full:
                self._train(next_state, done)

            # The buffer's `add` method returns True if it's now full.
            # However, checking `self.buffer.full` after adding is more explicit.
            self.buffer.add(self.last_obs, _action.reshape(1, -1), np.array([reward]),
                            np.array([self.last_done]), value, log_prob)
            
            # === CORE CHANGE: IMMEDIATE TRAINING TRIGGER ===
            if self.buffer.full:
                self._train(next_state, done)
        
        self.prev_state = state # Update prev_state to the state *before* this transition
        self.last_done = done

    def save_model(self, suffix: str = ""):
        """Save model checkpoint."""
        if not self.training_mode:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_model = self.model.__class__.__name__.lower()
        filename = f"{name_model}_{suffix}_{timestamp}.zip" if suffix else f"{name_model}_{timestamp}.zip"
        filepath = os.path.join(self.model_dir, filename)
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
        
        # Save training metadata
        metadata = {
            'episode': self.total_episodes,
            'phase': self.curriculum_phase,
            'best_constraint_score': self.best_constraint_score,
            'violation_history': list(self.violation_history),
            'reward_weights': self.reward_weights
        }
        
        metadata_path = filepath.replace('.zip', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'episode': self.total_episodes,
            'phase': self.phase_names[self.curriculum_phase],
            'avg_violations': np.mean(self.violation_history) if self.violation_history else 0,
            'consecutive_clean_episodes': self.episodes_without_violations,
            'best_constraint_score': self.best_constraint_score,
            'current_reward_weights': self.reward_weights.copy()
        }
    
    def _find_latest_model_and_buffer(self) -> Tuple[Optional[str], Optional[str]]:
        """Scans a directory to find the latest model .zip and the buffer file."""
        buffer_path = os.path.join(self.model_dir, "rollout_buffer.pkl")
        if not os.path.exists(buffer_path):
            buffer_path = None
            
        # Find all model files
        model_files = glob.glob(os.path.join(self.model_dir, "ppo_*.zip"))
        if not model_files:
            return None, buffer_path
            
        # Find the latest one by modification time
        latest_model_file = max(model_files, key=os.path.getmtime)
        return latest_model_file, buffer_path