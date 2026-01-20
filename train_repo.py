"""
train_repo.py - RePo Training Example

Demonstrates continual pretraining with RePo on a language modeling task.

Training strategy (from paper):
- Start from pretrained checkpoint with standard RoPE
- Apply RePo from layer ℓ onward (e.g., ℓ=5)
- Lower layers keep RoPE for local syntax
- Upper layers use RePo for semantic reorganization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from repo_attention import TransformerBlock


class SimpleTransformer(nn.Module):
    """
    Simple transformer language model with RePo.
    
    Architecture choice:
    - Layers 0 to repo_start_layer-1: Standard RoPE
    - Layers repo_start_layer to num_layers-1: RePo
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        repo_start_layer: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # If repo_start_layer not specified, use RePo in upper half
        repo_start_layer = repo_start_layer or (num_layers // 2)
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Build transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_repo=(i >= repo_start_layer)  # RePo in upper layers only
            )
            for i in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie weights between embedding and output
        self.lm_head.weight = self.embedding.weight
        
        print(f"✓ Model created:")
        print(f"    Layers 0-{repo_start_layer-1}: Standard RoPE")
        print(f"    Layers {repo_start_layer}-{num_layers-1}: RePo")
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            input_ids: (batch, seq_len) token indices
            mask: Optional attention mask
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Token embeddings
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Final layer norm and projection
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0
    ):
        """
        Autoregressive generation.
        
        Args:
            input_ids: (batch, seq_len) prompt tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
        
        Returns:
            generated: (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def train_step(model, batch, optimizer, device):
    """Single training step."""
    model.train()
    
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss (cross-entropy)
    # Flatten: (batch * seq_len, vocab_size)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 50000
    hidden_dim = 768
    num_layers = 12
    num_heads = 12
    max_seq_len = 2048
    repo_start_layer = 6  # Apply RePo from layer 6 onward
    
    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        repo_start_layer=repo_start_layer
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    repo_params = sum(
        p.numel() for layer in model.layers[repo_start_layer:]
        for p in layer.attention.repo.parameters()
    )
    
    print(f"\n✓ Total parameters: {total_params:,}")
    print(f"✓ RePo parameters: {repo_params:,} ({100*repo_params/total_params:.2f}%)")
    
    # Test forward pass
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"\n✓ Forward pass successful: {logits.shape}")
    print(f"✓ Ready for training!")
