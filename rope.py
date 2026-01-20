"""
- For position p and dimension pair (2i, 2i+1), apply rotation:
  [[cos(θ_i * p), -sin(θ_i * p)],
   [sin(θ_i * p),  cos(θ_i * p)]]
  
- Base frequencies: θ_i = base^(-2i/d), typically base=10000
"""

import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    """
    RoPE: Encodes position via rotation matrices in complex plane.
    
    Key insight: Relative position (j-i) naturally emerges from 
    rotation difference between positions j and i.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        """
        Args:
            dim: Embedding dimension (must be even)
            max_seq_len: Maximum sequence length for cache
            base: Base for frequency calculation (10000 in original paper)
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies: θ_i = base^(-2i/d)
        # Shape: (dim//2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute for efficiency
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """
        Precompute cos and sin for all positions up to seq_len.
        
        Mathematical operation:
        For each position p ∈ [0, seq_len-1] and each frequency θ_i:
        - Compute angle: p * θ_i
        - Store cos(p * θ_i) and sin(p * θ_i)
        """
        # Position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.float32)
        
        # Outer product: positions ⊗ inv_freq
        # Shape: (seq_len, dim//2)
        freqs = torch.outer(positions, self.inv_freq)
        
        # Duplicate for both dimensions in each pair
        # Shape: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Cache cos and sin with broadcast-compatible shapes
        # Shape: (seq_len, 1, 1, dim) for (seq, batch, heads, dim)
        self.register_buffer('cos_cached', emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :], persistent=False)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half dimensions for complex multiplication.
        
        Transform [x1, x2, x3, x4, ...] → [-x2, x1, -x4, x3, ...]
        
        This implements: multiply by i in complex plane
        Real part (x1, x3, ...) → Imaginary part with sign flip
        Imaginary part (x2, x4, ...) → Real part
        """
        x1, x2 = x.chunk(2, dim=-1)  # Split into two halves
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        positions: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.
        
        Standard RoPE: positions = [0, 1, 2, ..., seq_len-1] (linear)
        RePo: positions = continuous learned values
        
        Args:
            q: Query (batch, seq_len, num_heads, head_dim)
            k: Key (batch, seq_len, num_heads, head_dim)
            positions: Optional continuous positions for RePo
        
        Returns:
            Rotated (q, k) tuple
        """
        seq_len = q.shape[1]
        
        if positions is None:
            # Standard RoPE with integer positions
            if seq_len > self.max_seq_len:
                self._build_cache(seq_len)
            
            cos = self.cos_cached[:, :seq_len, :, :]
            sin = self.sin_cached[:, :seq_len, :, :]
        else:
            # RePo: interpolate for continuous positions
            cos, sin = self._interpolate_rotary(positions)
        
        # Apply rotation:
        # rotated = x * cos + rotate_half(x) * sin
        #
        # This is equivalent to:
        # [x_real * cos - x_imag * sin,
        #  x_real * sin + x_imag * cos]
        # which is complex rotation by angle θ
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def _interpolate_rotary(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos/sin for CONTINUOUS positions (crucial for RePo!).
        
        Unlike standard RoPE which uses integer positions [0, 1, 2, ...],
        RePo learns continuous positions like [0.5, 2.3, 1.1, ...].
        
        Args:
            positions: (batch, seq_len) continuous position values
        
        Returns:
            cos, sin: (batch, seq_len, 1, dim) interpolated values
        """
        batch_size, seq_len = positions.shape
        
        # Compute angles: position * θ_i for each position and frequency
        # positions: (batch, seq_len, 1)
        # inv_freq: (dim//2,)
        # Result: (batch, seq_len, dim//2)
        freqs = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        
        # Duplicate for both dimensions in pair
        emb = torch.cat([freqs, freqs], dim=-1)  # (batch, seq_len, dim)
        
        # Compute cos and sin
        cos = emb.cos().unsqueeze(2)  # (batch, seq_len, 1, dim)
        sin = emb.sin().unsqueeze(2)
        
        return cos, sin


if __name__ == "__main__":
    # Test RoPE
    rope = RotaryPositionEmbedding(dim=64, max_seq_len=512)
    
    # Standard attention
    q = torch.randn(2, 10, 4, 64)  # batch=2, seq=10, heads=4, dim=64
    k = torch.randn(2, 10, 4, 64)
    
    q_rot, k_rot = rope.apply_rotary_pos_emb(q, k)
    print(f" Standard RoPE: {q_rot.shape}")
    
    # RePo with continuous positions
    positions = torch.rand(2, 10) * 20  # Random continuous positions
    q_rot_repo, k_rot_repo = rope.apply_rotary_pos_emb(q, k, positions)
    print(f"RePo with continuous positions: {q_rot_repo.shape}")
