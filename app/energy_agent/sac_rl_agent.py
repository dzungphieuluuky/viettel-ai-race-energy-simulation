# --- START OF FILE rl_agent.py ---

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import logging
import os
import pickle
import glob
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from collections import deque

# --- Stable Baselines3 Imports (Minimized) ---
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer

from .state_normalizer import StateNormalizer 

# ==============================================================================
# === STATE FEATURE INDICES & CUSTOM NETWORK (Unchanged) =======================
# ... (These sections are identical) ...
# ==============================================================================
TOTAL_CELLS, TOTAL_UES, SIM_TIME, TIME_STEP, TIME_PROGRESS = 0, 1, 2, 3, 4
CARRIER_FREQUENCY, ISD, MIN_TX_POWER, MAX_TX_POWER = 5, 6, 7, 8
BASE_POWER, IDLE_POWER = 9, 10
DROP_CALL_THRESHOLD, LATENCY_THRESHOLD = 11, 12
CPU_THRESHOLD, PRB_THRESHOLD = 13, 14
TRAFFIC_LAMBDA, PEAK_HOUR_MULTIPLIER = 15, 16

TOTAL_ENERGY, ACTIVE_CELLS = 17, 18
AVG_DROP_RATE, AVG_LATENCY = 19, 20
TOTAL_TRAFFIC, CONNECTED_UES, CONNECTION_RATE = 21, 22, 23
CPU_VIOLATIONS, PRB_VIOLATIONS = 24, 25
MAX_CPU_USAGE, MAX_PRB_USAGE = 26, 27
KPI_VIOLATIONS, TOTAL_TX_POWER, AVG_POWER_RATIO = 28, 29, 30

class LightweightAttentionNetwork(nn.Module):
    # ... (Unchanged) ...
    def __init__(self, features_dim: int = 256, max_cells: int = 57, n_cell_features: int = 12):
        super().__init__()
        self.features_dim, self.max_cells, self.n_cell_features = features_dim, max_cells, n_cell_features
        self.global_features_dim = 17 + 14
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_features_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU())
        self.cell_embedding = nn.Linear(self.n_cell_features, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.final_mlp = nn.Sequential(
            nn.Linear(64 + 128, features_dim), 
            nn.ReLU(), 
            nn.Linear(features_dim, features_dim), 
            nn.ReLU())
        
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
# === MANUAL SAC NETWORK DEFINITIONS (Unchanged) ===============================
# ... (Actor and Critic classes are the same) ...
# ==============================================================================
LOG_STD_MAX, LOG_STD_MIN = 2, -20
class Actor(nn.Module):
    # ... (Unchanged) ...
    def __init__(self, feature_extractor: nn.Module, action_dim: int):
        super().__init__()
        self.feature_extractor, self.fc_mean, self.fc_log_std = feature_extractor, nn.Linear(feature_extractor.features_dim, action_dim), nn.Linear(feature_extractor.features_dim, action_dim)
    def forward(self, state):
        features = self.feature_extractor(state)
        mean, log_std = self.fc_mean(features), torch.clamp(self.fc_log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    def sample(self, state):
        mean, log_std = self.forward(state)
        
        # FIX: First, calculate and assign 'std'.
        std = log_std.exp()
        
        # Then, use the newly created 'std' variable.
        normal = Normal(mean, std)
        
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        
        # This part for enforcing action bounds remains correct.
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
class Critic(nn.Module):
    # ... (Unchanged) ...
    def __init__(self, feature_extractor: nn.Module, action_dim: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.l1, self.l2, self.l3 = nn.Linear(feature_extractor.features_dim + action_dim, 256), nn.Linear(256, 256), nn.Linear(256, 1)
        self.l4, self.l5, self.l6 = nn.Linear(feature_extractor.features_dim + action_dim, 256), nn.Linear(256, 256), nn.Linear(256, 1)
    def forward(self, state, action):
        features = self.feature_extractor(state)
        sa = torch.cat([features, action], 1)
        q1 = F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(sa))))))
        q2 = F.relu(self.l6(F.relu(self.l5(F.relu(self.l4(sa))))))
        return q1, q2

# ==============================================================================
# === REWARD CALCULATOR (CORRECTED) ============================================
# ==============================================================================
class StabilizedRewardCalculator:
    """
    Stabilized reward calculator with normalized outputs.
    
    Key Changes:
    1. Rewards normalized to [-1, 1] range
    2. Gradual penalty ramp-up (prevents early explosion)
    3. Running statistics for adaptive normalization
    """
    
    def __init__(self, curriculum_phase: int = 0, total_steps: int = 0):
        self.curriculum_phase = curriculum_phase
        self.total_steps = total_steps
        
        # Violation tracking
        self.violation_history = deque(maxlen=100)
        self.recent_violations = deque(maxlen=20)
        self.episode_violations = 0
        self.episode_steps = 0
        self.consecutive_clean_steps = 0
        
        # Energy baseline
        self.baseline_energy = None
        self.episode_energy_samples = []
        
        # Running statistics for reward normalization
        self.reward_history = deque(maxlen=1000)
        self.raw_reward_mean = 0.0
        self.raw_reward_std = 1.0
        
        # === STABILIZED WEIGHTS (Smaller initial values) ===
        # Gradually increase over time
        self.base_weights = self._get_base_weights(curriculum_phase)
        self.weights = self.base_weights.copy()
        self._apply_rampup()
        
        # Minimum consecutive clean steps
        self.min_clean_steps_for_energy = 50
    
    def _get_base_weights(self, phase: int) -> Dict[str, float]:
        """Get base weights for curriculum phase."""
        if phase == 0:  # Constraint focus
            return {
                'drop_call': 100.0,      # Reduced from 500
                'latency': 80.0,         # Reduced from 400
                'cpu_violation': 60.0,   # Reduced from 300
                'prb_violation': 60.0,   # Reduced from 300
                'connection': 40.0,      # Reduced from 200
                'energy': 0.0,
                'perfect_bonus': 10.0    # Reduced from 50
            }
        elif phase == 1:  # Balanced
            return {
                'drop_call': 80.0,
                'latency': 64.0,
                'cpu_violation': 48.0,
                'prb_violation': 48.0,
                'connection': 32.0,
                'energy': 2.0,
                'perfect_bonus': 8.0
            }
        else:  # Energy optimization
            return {
                'drop_call': 60.0,
                'latency': 48.0,
                'cpu_violation': 40.0,
                'prb_violation': 40.0,
                'connection': 24.0,
                'energy': 10.0,
                'perfect_bonus': 6.0
            }
    
    def _apply_rampup(self):
        """
        Gradually increase penalty weights over training.
        Prevents early training instability.
        """
        # Ramp up over first 10,000 steps
        rampup_steps = 10000
        if self.total_steps < rampup_steps:
            # Start at 20% of full strength, linearly increase to 100%
            rampup_factor = 0.2 + 0.8 * (self.total_steps / rampup_steps)
        else:
            rampup_factor = 1.0
        
        # Apply rampup to penalty weights (not bonus)
        for key in ['drop_call', 'latency', 'cpu_violation', 'prb_violation', 'connection']:
            self.weights[key] = self.base_weights[key] * rampup_factor
        
        # Bonus and energy weights don't need rampup
        self.weights['energy'] = self.base_weights['energy']
        self.weights['perfect_bonus'] = self.base_weights['perfect_bonus']
    
    def _validate_state(self, state: np.ndarray) -> bool:
        """Validate state values."""
        if state is None or len(state) < 31:
            return False
        
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return False
        
        if state[AVG_DROP_RATE] < 0:
            return False
        if state[AVG_LATENCY] < 0:
            return False
        if state[TOTAL_ENERGY] < 0:
            return False
        if state[CONNECTION_RATE] < 0 or state[CONNECTION_RATE] > 1.0:
            return False
        if state[CPU_VIOLATIONS] < 0 or state[CPU_VIOLATIONS] > 1000:
            return False
        if state[PRB_VIOLATIONS] < 0 or state[PRB_VIOLATIONS] > 1000:
            return False
        
        return True
    
    def _calculate_constraint_penalty(self, state: np.ndarray) -> tuple[float, bool, dict]:
        """Calculate constraint penalties."""
        penalty = 0.0
        has_violation = False
        details = {
            'drop_call': False,
            'latency': False,
            'cpu': False,
            'prb': False,
            'connection': False
        }
        
        # 1. DROP CALL
        drop_rate = state[AVG_DROP_RATE]
        drop_threshold = state[DROP_CALL_THRESHOLD]
        
        if drop_threshold > 0 and drop_rate > drop_threshold:
            violation_ratio = (drop_rate - drop_threshold) / drop_threshold
            
            # Smoothed escalation (not as extreme)
            if violation_ratio > 0.5:
                escalation = 5.0      # Reduced from 20
            elif violation_ratio > 0.25:
                escalation = 3.0      # Reduced from 10
            elif violation_ratio > 0.1:
                escalation = 2.0      # Reduced from 5
            else:
                escalation = 1.5
            
            penalty += self.weights['drop_call'] * (violation_ratio ** 2) * escalation
            has_violation = True
            details['drop_call'] = True
        
        # 2. LATENCY
        latency = state[AVG_LATENCY]
        latency_threshold = state[LATENCY_THRESHOLD]
        
        if latency_threshold > 0 and latency > latency_threshold:
            violation_ratio = (latency - latency_threshold) / latency_threshold
            
            if violation_ratio > 0.5:
                escalation = 4.0
            elif violation_ratio > 0.25:
                escalation = 2.5
            elif violation_ratio > 0.1:
                escalation = 1.8
            else:
                escalation = 1.3
            
            penalty += self.weights['latency'] * (violation_ratio ** 2) * escalation
            has_violation = True
            details['latency'] = True
        
        # 3. CPU VIOLATIONS
        cpu_violations = int(state[CPU_VIOLATIONS])
        if cpu_violations > 0:
            penalty += self.weights['cpu_violation'] * (cpu_violations ** 1.3)  # Reduced from 1.5
            has_violation = True
            details['cpu'] = True
        
        # 4. PRB VIOLATIONS
        prb_violations = int(state[PRB_VIOLATIONS])
        if prb_violations > 0:
            penalty += self.weights['prb_violation'] * (prb_violations ** 1.3)
            has_violation = True
            details['prb'] = True
        
        # 5. CONNECTION RATE
        connection_rate = state[CONNECTION_RATE]
        target_rate = 0.98
        
        if connection_rate < target_rate:
            shortage = target_rate - connection_rate
            penalty += self.weights['connection'] * (shortage ** 2) * 100  # Reduced from 500
            has_violation = True
            details['connection'] = True
        
        return penalty, has_violation, details
    
    def _calculate_energy_reward(self, prev_state: np.ndarray, 
                                 current_state: np.ndarray,
                                 sustained_compliance: bool) -> float:
        """Calculate energy reward with sustained compliance."""
        if self.curriculum_phase == 0:
            return 0.0
        
        if not sustained_compliance:
            return 0.0
        
        if self.consecutive_clean_steps < self.min_clean_steps_for_energy:
            return 0.0
        
        prev_energy = prev_state[TOTAL_ENERGY]
        current_energy = current_state[TOTAL_ENERGY]
        
        if prev_energy <= 0 or current_energy <= 0:
            return 0.0
        
        energy_saved = prev_energy - current_energy
        
        if self.baseline_energy is None:
            self.baseline_energy = prev_energy
        
        if current_energy >= self.baseline_energy * 0.95:
            return 0.0
        
        if energy_saved > 0:
            max_reward = self.weights['energy'] * 3.0  # Reduced cap
            reward = min(self.weights['energy'] * np.log1p(energy_saved), max_reward)
            return reward
        
        return 0.0
    
    def _calculate_perfect_operation_bonus(self, sustained_compliance: bool) -> float:
        """Calculate perfect operation bonus."""
        if not sustained_compliance:
            return 0.0
        
        if self.consecutive_clean_steps < 20:
            return 0.0
        
        bonus = self.weights['perfect_bonus']
        
        if self.consecutive_clean_steps >= 100:
            bonus *= 2.0
        elif self.consecutive_clean_steps >= 50:
            bonus *= 1.5
        
        return bonus
    
    def _normalize_reward(self, raw_reward: float) -> float:
        """
        Normalize reward to roughly [-1, 1] range using running statistics.
        This is KEY for SAC stability.
        """
        # Update running statistics
        self.reward_history.append(raw_reward)
        if len(self.reward_history) >= 100:
            # Update mean and std
            rewards_array = np.array(self.reward_history)
            self.raw_reward_mean = np.mean(rewards_array)
            self.raw_reward_std = np.std(rewards_array) + 1e-6  # Avoid division by zero
            normalized = (raw_reward - self.raw_reward_mean) / (self.raw_reward_std + 1e-8)
            return float(np.clip(normalized, -3.0, 3.0))        
        else:
            # Before statistics: simple scaling
            return float(np.clip(raw_reward / 50.0, -3.0, 3.0))
    
    def calculate_reward(self, prev_state: Optional[np.ndarray], 
                        current_state: np.ndarray) -> float:
        """
        Calculate reward with stabilization.
        
        Key changes:
        1. Smaller base penalties
        2. Gradual ramp-up
        3. Reward normalization
        """
        # Update total steps for rampup
        self.total_steps += 1
        self._apply_rampup()
        
        # Flatten states
        flat_current_state = np.array(current_state).flatten()
        flat_prev_state = np.array(prev_state).flatten() if prev_state is not None else None
        
        # Validate
        if not self._validate_state(flat_current_state):
            raw_reward = -100.0  # Reduced from -1000
        elif len(flat_current_state) < 31:
            raw_reward = 0.0
        else:
            # Calculate penalties
            penalty, has_violation, violation_details = self._calculate_constraint_penalty(flat_current_state)
            
            # Start with negative penalty
            raw_reward = -penalty
            
            # Update violation tracking
            if has_violation:
                self.consecutive_clean_steps = 0
                self.recent_violations.append(1)
            else:
                self.consecutive_clean_steps += 1
                self.recent_violations.append(0)
            
            # Sustained compliance check
            sustained_compliance = (
                self.consecutive_clean_steps >= 20 and
                len(self.recent_violations) >= 20 and
                sum(self.recent_violations) <= 2
            )
            
            # Energy reward
            if flat_prev_state is not None and sustained_compliance:
                energy_reward = self._calculate_energy_reward(
                    flat_prev_state, 
                    flat_current_state,
                    sustained_compliance
                )
                raw_reward += energy_reward
            
            # Perfect bonus
            perfect_bonus = self._calculate_perfect_operation_bonus(sustained_compliance)
            raw_reward += perfect_bonus
            
            # Tracking
            self.violation_history.append(1 if has_violation else 0)
            self.episode_violations += (1 if has_violation else 0)
            self.episode_steps += 1
            self.episode_energy_samples.append(flat_current_state[TOTAL_ENERGY])
            
            # Clip raw reward (less extreme than before)
            raw_reward = float(np.clip(raw_reward, -1000, 100))
        
        # === NORMALIZE REWARD (KEY FOR STABILITY) ===
        normalized_reward = self._normalize_reward(raw_reward)
        
        return normalized_reward
    
    def reset_episode(self):
        """Reset episode tracking."""
        self.episode_violations = 0
        self.episode_steps = 0
        self.consecutive_clean_steps = 0
        
        # Update baseline energy
        if len(self.episode_energy_samples) > 0:
            avg_episode_energy = np.mean(self.episode_energy_samples)
            if self.baseline_energy is None:
                self.baseline_energy = avg_episode_energy
            else:
                self.baseline_energy = 0.95 * self.baseline_energy + 0.05 * avg_episode_energy
        
        self.episode_energy_samples = []
    
    def get_stats(self) -> Dict:
        """Get episode statistics."""
        return {
            'violations': self.episode_violations,
            'steps': self.episode_steps,
            'violation_rate': self.episode_violations / max(1, self.episode_steps),
            'recent_violation_rate': np.mean(self.recent_violations) if self.recent_violations else 0.0,
            'consecutive_clean_steps': self.consecutive_clean_steps,
            'baseline_energy': self.baseline_energy,
            'reward_mean': self.raw_reward_mean,
            'reward_std': self.raw_reward_std,
            'current_weights': self.weights.copy()
        }

class StabilizedSACTraining:
    """
    Additional stabilization techniques for SAC training.
    Add these to your RLAgent._train() method.
    """
    
    @staticmethod
    def compute_critic_loss_with_huber(current_q1, current_q2, target_q, delta=1.0):
        """
        Use Huber loss instead of MSE for critic.
        More robust to outliers and large errors.
        """
        import torch.nn.functional as F
        
        # Huber loss (smooth L1)
        loss1 = F.smooth_l1_loss(current_q1, target_q, reduction='mean', beta=delta)
        loss2 = F.smooth_l1_loss(current_q2, target_q, reduction='mean', beta=delta)
        
        return loss1 + loss2
    
    @staticmethod
    def clip_gradients(model, max_norm=1.0):
        """Clip gradients to prevent explosion."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def get_adaptive_learning_rate(initial_lr, total_steps, warmup_steps=2000, decay_steps=50000):
        """
        Adaptive learning rate schedule.
        Warmup -> Constant -> Decay
        """
        if total_steps < warmup_steps:
            # Linear warmup
            return initial_lr * (total_steps / warmup_steps)
        elif total_steps < decay_steps:
            # Constant
            return initial_lr
        else:
            # Exponential decay
            decay_factor = 0.95 ** ((total_steps - decay_steps) / 5000)
            return initial_lr * decay_factor

# ==============================================================================
# === FULLY MANUAL RL AGENT (FIXED) ============================================
# ==============================================================================
class RLAgent:
    def __init__(self, 
                 n_cells: int,
                 max_time: int,
                 n_ues: int,
                 use_gpu: bool = False,
                 training_mode: bool = False,
                 curriculum_phase: int = 0,
                 log_file: str = 'rl_agent.log'):
        
        self.max_cells, self.n_cells = 57, n_cells
        self.max_time, self.n_ues = max_time, n_ues
        self.training_mode = training_mode
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.curriculum_phase = curriculum_phase

        self.model_dir = "online_trained_models_manual_sac"
        self.buffer_path = os.path.join(self.model_dir, "replay_buffer.pkl")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # HYPERPARAMETERS
        self.buffer_size = 100_000
        self.batch_size = 512
        self.learning_starts = 2000
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

        self.policy_lr = 3e-4
        self.critic_lr = 3e-4
        self.alpha_lr = 3e-4

        self.state_dim, self.action_dim = 17 + 14 + (self.max_cells * 12), self.max_cells
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)
        action_space = spaces.Box(low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.state_normalizer = StateNormalizer(state_dim=self.state_dim, n_cells=self.max_cells)

        self.setup_logging(log_file)

        feature_extractor = LightweightAttentionNetwork(max_cells=self.max_cells).to(self.device)
        self.actor = Actor(feature_extractor, self.action_dim).to(self.device)
        self.critic = Critic(feature_extractor, self.action_dim).to(self.device)
        self.critic_target = Critic(feature_extractor, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), 
                                          lr=self.policy_lr,
                                          eps=1e-8)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 
                                           lr=self.critic_lr,
                                           eps=1e-8)
        
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], 
                                          lr=self.alpha_lr,
                                          eps=1e-8)
        self.alpha = self.log_alpha.exp()
        

        if self.training_mode:
            if os.path.exists(self.buffer_path):
                try:
                    with open(self.buffer_path, 'rb') as f: self.replay_buffer = pickle.load(f)
                    self.logger.info(f"Loaded ReplayBuffer. Size: {len(self.replay_buffer.observations)}/{self.buffer_size}")
                except Exception: self.replay_buffer = ReplayBuffer(self.buffer_size, obs_space, action_space, self.device, n_envs=1)
            else: self.replay_buffer = ReplayBuffer(self.buffer_size, obs_space, action_space, self.device, n_envs=1)
        
        self.total_steps, self.total_episodes, self.episode_reward = 0, 0, 0.0
        self.prev_state, self.last_obs = None, None
        self.reward_calculator = StabilizedRewardCalculator(
                                curriculum_phase=curriculum_phase,
                                total_steps=self.total_steps)
        self._load_latest_model()
    def _prepare_observation(self, state: np.ndarray) -> np.ndarray:
        flat_state = np.array(state).flatten()
        padded_state = np.zeros(self.state_dim, dtype=np.float32)
        padded_state[:len(flat_state)] = flat_state
        return self.state_normalizer.normalize(padded_state)

    def setup_logging(self, log_file: str):
        self.logger = logging.getLogger('RLAgent')
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def start_scenario(self):
        self.total_episodes += 1
        self.episode_reward, self.prev_state = 0.0, None
        self.logger.info(f"--- Episode {self.total_episodes} Starting ---")
        self.reward_calculator.reset_episode()

    def end_scenario(self):
        self.logger.info(f"--- Episode {self.total_episodes} Ended | Reward: {self.episode_reward:.2f} ---")
        stats = self.reward_calculator.get_stats()
        self.logger.info(f"Episode Stats: {stats}")
        if self.training_mode:
            try:
                with open(self.buffer_path, 'wb') as f: pickle.dump(self.replay_buffer, f)
                self.logger.info(f"ReplayBuffer saved. Size: {len(self.replay_buffer.observations)}/{self.buffer_size}")
            except Exception as e: self.logger.error(f"Failed to save buffer: {e}")
            self.save_model(suffix="scenario_end")

    # In RLAgent class

    def get_action(self, state: np.ndarray) -> np.ndarray:
        normalized_obs = self._prepare_observation(state)
        self.last_obs = normalized_obs
        state_tensor = torch.FloatTensor(normalized_obs).to(self.device).unsqueeze(0)

        if self.training_mode and self.total_steps < self.learning_starts:
            action = np.random.uniform(0.0, 0.8, size=self.n_cells)
            self.logger.info(f"Random action (exploration): {action}")
            return action
        else:
            if not self.training_mode:
                mean, _ = self.actor(state_tensor)
                # The action is the mean of the distribution for deterministic evaluation
                action_tanh = torch.tanh(mean)
            else:
                # Sample from the policy
                action_tanh, _ = self.actor.sample(state_tensor)
            
            action_numpy = action_tanh.detach().cpu().numpy().flatten()
            
            # --- FIX: Rescale action from [-1, 1] to [0, 1] instead of clipping ---
            action_rescaled = (action_numpy + 1.0) / 2.0
            
            return action_rescaled[:self.n_cells]
            
    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: bool):
        if not self.training_mode: return
        self.total_steps += 1
        reward = self.calculate_reward(state, next_state)
        self.episode_reward += reward
        next_obs_prepared = self._prepare_observation(next_state)
        flat_action = np.array(action).flatten()
        action_padded = np.zeros(self.action_dim)
        action_padded[:len(flat_action)] = flat_action
        
        # --- FIX: Wrap scalar reward and done in NumPy arrays ---
        reward_arr = np.array([reward])
        done_arr = np.array([done])
        # The ReplayBuffer expects infos to be a list of dictionaries
        infos_arr = [{}]
        # --------------------------------------------------------
        
        stats = self.reward_calculator.get_stats()
        self.logger.info(f"Reward mean: {stats['reward_mean']:.10f}") 
        self.logger.info(f"Reward std: {stats['reward_std']:.10f}")
        self.logger.info(f"Weights: {json.dumps(stats['current_weights'])}")

        self.replay_buffer.add(self.last_obs, next_obs_prepared, action_padded, reward_arr, done_arr, infos_arr)
        self.logger.info(f"Step {self.total_steps} | Reward: {reward:.2f} | Done: {done} | Buffer: {self.replay_buffer.pos}/{self.buffer_size}")
        if self.total_steps > self.learning_starts:
            self.logger.info(f"Training step at total steps: {self.total_steps}")
            self._train()
            self.logger.info(f"Training completed for step: {self.total_steps}")
        if self.total_steps == 20_000:
            self.reward_calculator.curriculum_phase = 1
        if self.total_steps == 40_000:
            self.reward_calculator.curriculum_phase = 2
        self.prev_state = state

    def _train(self):
        """
        Stabilized training method for SAC.
        Replace your current _train() method with this.
        """
        # Sample from buffer
        data = self.replay_buffer.sample(self.batch_size)
        states = data.observations
        actions = data.actions
        rewards = data.rewards
        next_states = data.next_observations
        dones = data.dones
        
        # === CRITIC UPDATE WITH HUBER LOSS ===
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q_backup = rewards + (1 - dones) * self.gamma * target_q
            target_q_backup = torch.clamp(target_q_backup, -50.0, 50.0)  # Prevent extreme targets
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)

        # Compute critic loss using Huber loss for stability
        critic_loss = StabilizedSACTraining.compute_critic_loss_with_huber(current_q1, current_q2, target_q_backup, delta=1.0)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        StabilizedSACTraining.clip_gradients(self.critic, max_norm=1.0)
        self.critic_optimizer.step()

        # 3. === ACTOR AND ALPHA UPDATE (Combined) ===
        # Freeze the critic network during the actor update to save computation
        for p in self.critic.parameters():
            p.requires_grad = False

        # Calculate actor and alpha loss from a single forward pass
        pi, log_pi = self.actor.sample(states)
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        # Actor loss
        actor_loss = ((self.alpha.detach() * log_pi) - min_qf_pi).mean()
        
        # Alpha (temperature) loss
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()

        # --- FIX: Perform actor and alpha updates together ---
        # Zero gradients for both optimizers first
        self.actor_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        
        # Backward pass on the combined loss graph (implicitly)
        # This resolves the "backward a second time" error without needing retain_graph=True
        actor_loss.backward(retain_graph=True) # Retain for alpha backward pass
        alpha_loss.backward()

        # Clip gradients
        StabilizedSACTraining.clip_gradients(self.actor, max_norm=1.0)
        
        # Step both optimizers
        self.actor_optimizer.step()
        self.alpha_optimizer.step()
        
        with torch.no_grad():
            self.log_alpha.data = torch.clamp(self.log_alpha.data, -3.0, 0.0)
            self.alpha = self.log_alpha.exp().detach() # Update alpha value

        # Unfreeze the critic network for the next training step
        for p in self.critic.parameters():
            p.requires_grad = True
        # ----------------------------------------------------

        # 4. === SOFT TARGET UPDATE ===
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # === LOGGING (Every 100 steps) ===
        if self.total_steps % 100 == 0:
            self.logger.info(
                f"Step {self.total_steps} | \n"
                f"Critic Loss: {critic_loss.item():.4f} | \n"
                f"Critic Loss < 10.0: {critic_loss.item() < 10.0} | \n"
                f"Actor Loss: {actor_loss.item():.4f} | \n"
                f"Actor Loss < 5.0: {actor_loss.item() < 5.0} | \n"
                f"Alpha: {self.alpha.item():.4f} | \n"
                f"Alpha in range [0.05, 0.5]: {0.05 <= self.alpha.item() <= 0.5} | \n"
                f"Q Mean: {current_q1.mean().item():.2f} | \n"
                f"Q Mean in range [-10, 10]: {-10 <= current_q1.mean().item() <= 10}"
            )

            # Check training healthy
            if critic_loss.item() > 100.0:
                self.logger.warning("High critic loss detected! Consider reviewing training stability.")
            if actor_loss.item() > 50.0:
                self.logger.warning("High actor loss detected! Consider reviewing training stability.")
            if not (0.01 <= self.alpha.item() <= 1.0):
                self.logger.warning("Alpha out of expected range! Check temperature tuning.")
            if not (-20 <= current_q1.mean().item() <= 20):
                self.logger.warning("Q-value mean out of expected range! Check critic performance.")

    def calculate_reward(self, prev_state: np.ndarray, current_state: np.ndarray) -> float:
        if prev_state is None: return 0.0
        return self.reward_calculator.calculate_reward(prev_state, current_state)

    def save_model(self, suffix: str = ""):
        if not self.training_mode: return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_sac_{suffix}_{timestamp}.pth"
        filepath = os.path.join(self.model_dir, filename)
        torch.save({'actor_state_dict': self.actor.state_dict(), 
                    'critic_state_dict': self.critic.state_dict(), 
                    'log_alpha': self.log_alpha, 
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(), 
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(), 
                    'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                    'total_steps': self.total_steps,
                    'curriculum_phase': self.curriculum_phase,
                    'reward_calculator': self.reward_calculator}, filepath)
        self.logger.info(f"Model saved: {filepath}")

    def _load_latest_model(self):
        model_files = glob.glob(os.path.join(self.model_dir, "manual_sac_*.pth"))
        if not model_files: 
            self.logger.info("No saved model found. Starting from scratch.")
            return
        latest_model_file = max(model_files, key=os.path.getmtime)
        self.logger.info(f"Loading latest model: {latest_model_file}")
        try:
            checkpoint = torch.load(latest_model_file, map_location=self.device, weights_only=False)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha = self.log_alpha.exp()
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.total_steps = checkpoint['total_steps']
            self.curriculum_phase = checkpoint.get('curriculum_phase', 0)
            self.reward_calculator = checkpoint.get('reward_calculator', 
                                                    StabilizedRewardCalculator(
                                                        curriculum_phase=self.curriculum_phase,
                                                        total_steps=self.total_steps))
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load model checkpoint: {e}. Starting fresh.")