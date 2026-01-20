"""
Integrates RePo position learning into the attention mechanism.

Key difference from standard attention:
- Standard: A_ij = q_i^T g_θ(j - i) k_j  (fixed positions)
- RePo: A_ij = q_i^T g_θ(z_j - z_i) k_j  (learned positions)

where z_i = f_φ(h_i) is the learned continuous position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rope import RotaryPositionEmbedding
from repo_module import RePoModule


class RePoMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Embedding (RePo variant).
    
    This combines:
    1. Standard multi-head attention mechanism
    2. RoPE for encoding relative positions
    3. RePo for learning optimal position assignments
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_repo: bool = True,
        position_dim: int = None
    ):
        """
        Args:
            hidden_dim: Model dimension (d_model)
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_repo: If True, use RePo; if False, use standard RoPE
            position_dim: Dimension for RePo module (default: hidden_dim // 8)
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_repo = use_repo
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Rotary position embedding
        self.rope = RotaryPositionEmbedding(dim=self.head_dim)
        
        # RePo position learning module (optional)
        if use_repo:
            self.repo = RePoModule(hidden_dim=hidden_dim, position_dim=position_dim)
        else:
            self.repo = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot product
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with RePo-enhanced attention.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            mask: Attention mask (batch, seq_len, seq_len) or None
            return_attention: If True, also return attention weights
        
        Returns:
            output: (batch, seq_len, hidden_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len) if return_attention
        """
        batch_size, seq_len, _ = x.shape
        
        # ============================================================
        # 1. COMPUTE QUERIES, KEYS, VALUES
        # ============================================================
        
        # Linear projections
        # Shape: (batch, seq_len, hidden_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        # Shape: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # ============================================================
        # 2. POSITION ASSIGNMENT (Standard RoPE vs RePo)
        # ============================================================
        
        if self.use_repo:
            # RePo: Learn continuous positions from content
            # positions: (batch, seq_len)
            learned_positions = self.repo(x)
            
            # Apply RoPE with learned positions
            Q, K = self.rope.apply_rotary_pos_emb(Q, K, positions=learned_positions)
        else:
            # Standard RoPE: Use linear positions [0, 1, 2, ..., seq_len-1]
            Q, K = self.rope.apply_rotary_pos_emb(Q, K, positions=None)
        
        # ============================================================
        # 3. SCALED DOT-PRODUCT ATTENTION
        # ============================================================
        
        # Transpose for attention computation
        # Q, K, V: (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        # scores: (batch, num_heads, seq_len, seq_len)
        # A_ij = (q_i^T k_j) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided (for causal/padding masking)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        # attention_weights: (batch, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # output: (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attention_weights, V)
        
        # ============================================================
        # 4. CONCATENATE HEADS AND PROJECT
        # ============================================================
        
        # Transpose back: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Concatenate heads: (batch, seq_len, hidden_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        if return_attention:
            return output, attention_weights
        return output


class TransformerBlock(nn.Module):
    """
    Complete Transformer block with RePo attention and FFN.
    
    Architecture:
    x -> LayerNorm -> RePo-Attention -> Residual -> LayerNorm -> FFN -> Residual
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int = None,
        dropout: float = 0.1,
        use_repo: bool = True
    ):
        super().__init__()
        
        ffn_dim = ffn_dim or (4 * hidden_dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # RePo attention
        self.attention = RePoMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_repo=use_repo
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with pre-normalization and residual connections.
        
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        # Attention block with residual
        x = x + self.attention(self.ln1(x), mask=mask)
        
        # FFN block with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


if __name__ == "__main__":
    # Test RePo attention
    batch_size, seq_len, hidden_dim = 2, 50, 512
    num_heads = 8
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Standard RoPE attention
    attn_standard = RePoMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        use_repo=False
    )
    out_standard = attn_standard(x)
    print(f" Standard RoPE attention: {out_standard.shape}")
    
    # RePo attention
    attn_repo = RePoMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        use_repo=True
    )
    out_repo, attn_weights = attn_repo(x, return_attention=True)
    print(f" RePo attention output: {out_repo.shape}")
    print(f" Attention weights: {attn_weights.shape}")
    
    # Test full transformer block
    block = TransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        use_repo=True
    )
    out_block = block(x)
    print(f" Transformer block output: {out_block.shape}")
