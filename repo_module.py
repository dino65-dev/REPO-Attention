"""
RePo Position Learning Module

This is the heart of RePo: learning to assign continuous positions
based on semantic content rather than sequential order.

Mathematical formulation:
z_i = f_φ(h_i) = [Swish(h_i W_g) ⊙ (h_i W_c)] W_z

Components:
1. Gate pathway: Swish(h_i W_g) - learns "which features matter"
2. Content pathway: h_i W_c - learns "what position to assign"
3. Hadamard product: ⊙ - combines them multiplicatively
4. Final projection: W_z - collapses to scalar position
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """
    Swish activation: f(x) = x * σ(x) = x / (1 + exp(-x))
    
    Properties:
    - Smooth, non-monotonic
    - Unbounded above, bounded below
    - Self-gated: uses its own value as gate
    
    Derivative: f'(x) = σ(x) * (1 + x * (1 - σ(x)))
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class RePoModule(nn.Module):
    """
    RePo (Context Re-Positioning) Module.
    
    Learns to assign continuous position values to tokens based on
    their semantic content, enabling attention to focus on relevant
    information regardless of sequential distance.
    
    Architecture: SwiGLU (Swish-Gated Linear Unit)
    """
    
    def __init__(self, hidden_dim: int, position_dim: int = None):
        """
        Args:
            hidden_dim: Input hidden dimension (d)
            position_dim: Intermediate position dimension (d_p)
                         Default: hidden_dim // 8 (as in paper)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.position_dim = position_dim or (hidden_dim // 8)
        
        # Gate pathway: learns which semantic features are position-relevant
        # W_g ∈ ℝ^(d × d_p)
        self.W_gate = nn.Linear(hidden_dim, self.position_dim, bias=False)
        
        # Content pathway: learns the positional representation
        # W_c ∈ ℝ^(d × d_p)
        self.W_content = nn.Linear(hidden_dim, self.position_dim, bias=False)
        
        # Projection to scalar position
        # W_z ∈ ℝ^(d_p × 1)
        self.W_proj = nn.Linear(self.position_dim, 1, bias=False)
        
        # Swish activation for gating
        self.swish = Swish()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights with small values to start near identity.
        
        Intuition: Begin close to standard linear positions, then
        gradually learn to reorganize as training progresses.
        """
        nn.init.normal_(self.W_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_content.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_proj.weight, mean=0.0, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute continuous positions for input tokens.
        
        Mathematical formula:
        1. gate = Swish(h W_g)           [relevance scoring]
        2. content = h W_c                [position encoding]
        3. r = gate ⊙ content             [conditional features]
        4. z = r W_z                      [scalar position]
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        
        Returns:
            positions: (batch, seq_len) continuous position values
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # GATE PATHWAY: Which features matter for positioning?
        # gate_features: (batch, seq_len, position_dim)
        gate_features = self.W_gate(hidden_states)
        gate_activation = self.swish(gate_features)
        
        # CONTENT PATHWAY: What position should be assigned?
        # content_features: (batch, seq_len, position_dim)
        content_features = self.W_content(hidden_states)
        
        # HADAMARD PRODUCT: Conditional position representation
        # Each dimension is modulated by its relevance score
        # r: (batch, seq_len, position_dim)
        position_representation = gate_activation * content_features
        
        # PROJECT TO SCALAR: Weighted combination of positional hypotheses
        # positions: (batch, seq_len, 1) -> (batch, seq_len)
        positions = self.W_proj(position_representation).squeeze(-1)
        
        return positions
    
    def get_position_statistics(self, positions: torch.Tensor) -> dict:
        """
        Analyze learned position patterns (for debugging/visualization).
        
        Returns statistics about position density and range.
        """
        return {
            'min': positions.min().item(),
            'max': positions.max().item(),
            'mean': positions.mean().item(),
            'std': positions.std().item(),
            'range': (positions.max() - positions.min()).item(),
        }


if __name__ == "__main__":
    # Test RePo module
    repo = RePoModule(hidden_dim=512, position_dim=64)
    
    # Simulate token embeddings
    batch_size, seq_len, hidden_dim = 2, 100, 512
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Get learned positions
    positions = repo(hidden_states)
    
    print(f" RePo positions shape: {positions.shape}")
    print(f" Position statistics:")
    stats = repo.get_position_statistics(positions)
    for key, val in stats.items():
        print(f"    {key}: {val:.3f}")
    
    # Check that positions vary with input
    hidden_states_diff = torch.randn(batch_size, seq_len, hidden_dim)
    positions_diff = repo(hidden_states_diff)
    
    difference = (positions - positions_diff).abs().mean()
    print(f" Position sensitivity to input: {difference:.3f}")
