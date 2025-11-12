# custom_models_sb3.py
import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# A placeholder MLP network (can be replaced with your actual implementation)
class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 n_cells: int = 30, n_cell_features: int = 12): # Added parameters
        super().__init__(observation_space, features_dim)
        
        # --- Store these crucial dimensions ---
        self.n_cells = n_cells
        self.n_cell_features = n_cell_features
        
        # Calculate the size of the global features (sim + network)
        self.n_global_features = 17 + 14
        
        # Define the network architecture
        # IMPORTANT: The input to the first layer is the full observation space size
        input_size = observation_space.shape[0]
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # This simple network processes the whole flat vector.
        # It does not need to reshape.
        return self.network(observations)

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class MultiHeadCellAttention(nn.Module):
    """
    Multi-head attention mechanism specifically designed for cell features.
    Allows the network to attend to important cells dynamically.
    """
    def __init__(self, cell_feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert cell_feature_dim % num_heads == 0, "cell_feature_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = cell_feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(cell_feature_dim, cell_feature_dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(cell_feature_dim, cell_feature_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, n_cells, cell_feature_dim)
            mask: (batch_size, n_cells) - boolean mask where True = valid cell
        Returns:
            attended_features: (batch_size, n_cells, cell_feature_dim)
        """
        batch_size, n_cells, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch, n_cells, 3*dim)
        qkv = qkv.reshape(batch_size, n_cells, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, n_cells, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch, heads, n_cells, n_cells)
        
        # Apply mask if provided (mask out padded cells)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_cells)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (batch, heads, n_cells, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, n_cells, -1)
        
        return self.out_proj(out)


class EnhancedAttentionNetwork(BaseFeaturesExtractor):
    """
    Enhanced attention-based feature extractor with:
    1. Multi-head attention for cells
    2. Cell-to-global and global-to-cell interactions
    3. Residual connections
    4. Layer normalization
    5. Proper handling of padded cells
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 max_cells: int = 57, n_cell_features: int = 12, 
                 num_attention_heads: int = 4, use_layer_norm: bool = True):
        super().__init__(observation_space, features_dim)
        
        self.max_cells = max_cells
        self.n_cell_features = n_cell_features
        self.n_global_features = 17 + 14  # sim + network features
        self.use_layer_norm = use_layer_norm
        
        # --- Global Feature Processing ---
        self.global_extractor = nn.Sequential(
            nn.Linear(self.n_global_features, 128),
            nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            nn.ReLU()
        )
        
        # --- Cell Feature Processing ---
        # Initial embedding of cell features
        self.cell_embedding = nn.Sequential(
            nn.Linear(self.n_cell_features, 128),
            nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-head attention for cell interactions
        self.cell_attention = MultiHeadCellAttention(
            cell_feature_dim=128,
            num_heads=num_attention_heads,
            dropout=0.1
        )
        
        # Post-attention processing
        self.cell_post_attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            nn.ReLU()
        )
        
        # --- Cross-Attention: Global influences Cell ---
        self.global_to_cell_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # --- Aggregation Layer ---
        # Multiple aggregation strategies
        self.cell_aggregator = nn.Sequential(
            nn.LayerNorm(128 * 3) if use_layer_norm else nn.Identity(),
            nn.Linear(128 * 3, 128),  # Concat of mean, max, attention-weighted
            nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            nn.ReLU()
        )
        
        # --- Final Combination ---
        self.combiner = nn.Sequential(
            nn.Linear(128 + 128, 256),  # global + aggregated cells
            nn.LayerNorm(256) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim) if use_layer_norm else nn.Identity()
        )
        
    def _create_cell_mask(self, cell_features):
        """
        Create a mask to identify valid (non-padded) cells.
        Assumes padded cells have all zeros.
        
        Args:
            cell_features: (batch_size, max_cells, n_cell_features)
        Returns:
            mask: (batch_size, max_cells) boolean tensor
        """
        # A cell is valid if it has any non-zero feature
        # Use a small threshold to handle numerical errors
        mask = (cell_features.abs().sum(dim=-1) > 1e-6)
        return mask
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # --- Split observations ---
        global_features = observations[:, :self.n_global_features]
        cell_features_flat = observations[:, self.n_global_features:]
        
        # Reshape cell features
        cell_features = cell_features_flat.view(
            batch_size, self.max_cells, self.n_cell_features
        )
        
        # Create mask for valid cells
        cell_mask = self._create_cell_mask(cell_features)  # (batch, max_cells)
        
        # --- Process Global Features ---
        processed_global = self.global_extractor(global_features)  # (batch, 128)
        
        # --- Process Cell Features ---
        # Initial embedding
        embedded_cells = self.cell_embedding(cell_features)  # (batch, max_cells, 128)
        
        # Self-attention among cells (with masking)
        attended_cells = self.cell_attention(
            embedded_cells, 
            mask=cell_mask
        )  # (batch, max_cells, 128)
        
        # Residual connection
        attended_cells = embedded_cells + attended_cells
        
        # Post-attention processing
        processed_cells = self.cell_post_attention(attended_cells)  # (batch, max_cells, 128)
        
        # --- Cross-Attention: Global context influences cells ---
        # Expand global features to sequence format
        global_seq = processed_global.unsqueeze(1)  # (batch, 1, 128)
        
        # Use global as query, cells as key and value
        cross_attended, _ = self.global_to_cell_attention(
            query=global_seq,
            key=processed_cells,
            value=processed_cells,
            key_padding_mask=~cell_mask  # Mask padded cells
        )  # Output shape: (batch, 1, 128)
        
        # --- Aggregate Cell Features ---
        # Apply mask before aggregation
        masked_cells = processed_cells * cell_mask.unsqueeze(-1).float()
        
        # Multiple aggregation strategies
        # 1. Mean pooling (excluding padded cells)
        cell_counts = cell_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        mean_cells = masked_cells.sum(dim=1) / cell_counts  # (batch, 128)
        
        # 2. Max pooling (excluding padded cells)
        masked_cells_for_max = masked_cells.clone()
        # Expand mask to match feature dimensions for proper broadcasting
        mask_expanded = cell_mask.unsqueeze(-1).expand_as(masked_cells_for_max)  # (batch, max_cells, 128)
        masked_cells_for_max[~mask_expanded] = float('-inf')
        max_cells = masked_cells_for_max.max(dim=1)[0]  # (batch, 128)
        # Replace -inf with zeros (happens when all cells are masked)
        max_cells = torch.where(
            torch.isinf(max_cells), 
            torch.zeros_like(max_cells), 
            max_cells
        )
        
        # 3. Attention-weighted (from cross-attention)
        attention_weighted = cross_attended.squeeze(1) if cross_attended.dim() == 3 else cross_attended  # (batch, 128)
        
        # Ensure all have the same shape before concatenation
        assert mean_cells.dim() == 2, f"mean_cells has wrong dims: {mean_cells.shape}"
        assert max_cells.dim() == 2, f"max_cells has wrong dims: {max_cells.shape}"
        assert attention_weighted.dim() == 2, f"attention_weighted has wrong dims: {attention_weighted.shape}"
        
        # Concatenate all aggregations
        aggregated = torch.cat([mean_cells, max_cells, attention_weighted], dim=-1)  # (batch, 384)
        aggregated = self.cell_aggregator(aggregated)  # (batch, 128)
        
        # --- Final Combination ---
        combined = torch.cat([processed_global, aggregated], dim=-1)  # (batch, 256)
        output = self.combiner(combined)  # (batch, features_dim)
        
        return output


class LightweightAttentionNetwork(BaseFeaturesExtractor):
    """
    Lighter version with fewer parameters, faster training.
    Good for environments with limited compute or when training is slow.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256,
                 max_cells: int = 57, n_cell_features: int = 12):
        super().__init__(observation_space, features_dim)
        
        self.max_cells = max_cells
        self.n_cell_features = n_cell_features
        self.n_global_features = 17 + 14
        
        # Simpler processing
        self.global_net = nn.Sequential(
            nn.Linear(self.n_global_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.cell_net = nn.Sequential(
            nn.Linear(self.n_cell_features, 64),
            nn.ReLU()
        )
        
        # Simple attention mechanism
        self.attention_weights = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.final_net = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        global_features = observations[:, :self.n_global_features]
        cell_features_flat = observations[:, self.n_global_features:]
        cell_features = cell_features_flat.view(batch_size, self.max_cells, self.n_cell_features)
        
        # Create mask
        cell_mask = (cell_features.abs().sum(dim=-1) > 1e-6).float()  # (batch, max_cells)
        
        # Process features
        proc_global = self.global_net(global_features)  # (batch, 64)
        proc_cells = self.cell_net(cell_features)  # (batch, max_cells, 64)
        
        # Compute attention weights
        attn_logits = self.attention_weights(proc_cells).squeeze(-1)  # (batch, max_cells)
        attn_logits = attn_logits.masked_fill(cell_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (batch, max_cells, 1)
        
        # Weighted sum of cells
        aggregated_cells = (proc_cells * attn_weights).sum(dim=1)  # (batch, 64)
        
        # Combine and output
        combined = torch.cat([proc_global, aggregated_cells], dim=-1)
        return self.final_net(combined)