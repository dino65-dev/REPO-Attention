"""
visualize_positions.py - Visualize and analyze RePo's learned positions

This script helps understand WHAT RePo learns:
- Position density analysis
- Pattern classification (constant/monotonic/hybrid)
- Attention mass distribution
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from repo_attention import RePoMultiHeadAttention


def analyze_position_patterns(positions: torch.Tensor, threshold: float = 0.1):
    """
    Classify position patterns as constant, monotonic, or hybrid.
    
    Based on RePo paper analysis:
    - Constant: positions vary minimally
    - Monotonic: positions increase/decrease steadily
    - Hybrid: mix of constant and monotonic segments
    
    Args:
        positions: (batch, seq_len) learned positions
        threshold: Threshold for considering positions "constant"
    
    Returns:
        dict with pattern statistics
    """
    batch_size, seq_len = positions.shape
    patterns = {'constant': 0, 'monotonic': 0, 'hybrid': 0}
    
    for i in range(batch_size):
        pos = positions[i].cpu().numpy()
        
        # Check if constant (small variance)
        if np.std(pos) < threshold:
            patterns['constant'] += 1
            continue
        
        # Check if monotonic (correlation with sequence index)
        indices = np.arange(seq_len)
        correlation = np.corrcoef(pos, indices)[0, 1]
        
        if abs(correlation) > 0.9:
            patterns['monotonic'] += 1
        else:
            patterns['hybrid'] += 1
    
    return {k: v / batch_size for k, v in patterns.items()}


def compute_position_density(positions: torch.Tensor, num_bins: int = 50):
    """
    Analyze how densely positions are packed.
    
    RePo paper shows positions use much smaller range than sequence length
    (e.g., ~1000 positions for 4K tokens).
    """
    positions_flat = positions.flatten().cpu().numpy()
    
    hist, bin_edges = np.histogram(positions_flat, bins=num_bins)
    
    return {
        'range': positions_flat.max() - positions_flat.min(),
        'unique_ratio': len(np.unique(positions_flat)) / len(positions_flat),
        'histogram': (hist, bin_edges)
    }


def visualize_positions(model, input_text_embeddings):
    """
    Visualize learned positions for a sample input.
    
    Creates plots showing:
    1. Position vs sequence index
    2. Position density histogram
    3. Attention pattern with RePo
    """
    with torch.no_grad():
        # Get learned positions
        positions = model.repo(input_text_embeddings)
        
        # Analyze patterns
        pattern_stats = analyze_position_patterns(positions)
        density_stats = compute_position_density(positions)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Position vs sequence index
        ax = axes[0, 0]
        for i in range(min(5, positions.shape[0])):
            ax.plot(positions[i].cpu().numpy(), label=f'Sample {i+1}', alpha=0.7)
        ax.set_xlabel('Sequence Index')
        ax.set_ylabel('Learned Position')
        ax.set_title('RePo Position Assignment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Position density
        ax = axes[0, 1]
        hist, bin_edges = density_stats['histogram']
        ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7, color='blue')
        ax.set_xlabel('Position Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Position Density (range={density_stats["range"]:.1f})')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Pattern statistics
        ax = axes[1, 0]
        patterns = list(pattern_stats.keys())
        values = list(pattern_stats.values())
        ax.bar(patterns, values, color=['green', 'orange', 'blue'], alpha=0.7)
        ax.set_ylabel('Proportion')
        ax.set_title('Position Pattern Classification')
        ax.set_ylim([0, 1])
        for i, v in enumerate(values):
            ax.text(i, v + 0.05, f'{v:.2%}', ha='center')
        
        # Plot 4: Statistics summary
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
        RePo Position Statistics:
        
        Range: {density_stats['range']:.2f}
        Unique Ratio: {density_stats['unique_ratio']:.2%}
        
        Pattern Distribution:
        • Constant: {pattern_stats['constant']:.1%}
        • Monotonic: {pattern_stats['monotonic']:.1%}
        • Hybrid: {pattern_stats['hybrid']:.1%}
        
        Paper reported: 74% hybrid, 22% constant, 4% monotonic
        """
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace')
        
        plt.tight_layout()
        plt.savefig('repo_position_analysis.png', dpi=150)
        print("✓ Visualization saved to 'repo_position_analysis.png'")
        
        return pattern_stats, density_stats


if __name__ == "__main__":
    # Create model and test data
    hidden_dim = 512
    num_heads = 8
    batch_size = 32
    seq_len = 100
    
    model = RePoMultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        use_repo=True
    )
    
    # Random input (in practice, use real text embeddings)
    input_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Analyze positions
    print("Analyzing RePo position patterns...")
    pattern_stats, density_stats = visualize_positions(model, input_embeddings)
    
    print("\n✓ Analysis complete!")
    print(f"    Position range: {density_stats['range']:.2f}")
    print(f"    Hybrid patterns: {pattern_stats['hybrid']:.1%}")
