"""
GPT model configuration class.

Compatible with NanoChat checkpoint format.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class GPTConfig:
    """Configuration for GPT model.
    
    Compatible with NanoChat checkpoint format.
    """
    # Model architecture
    vocab_size: int = 50304  # GPT-2 vocab size
    n_layer: int = 12        # Number of transformer layers
    n_head: int = 12         # Number of attention heads
    n_embd: int = 768        # Embedding dimension
    block_size: int = 1024   # Context length
    
    # Optional parameters
    dropout: float = 0.0     # Dropout rate
    bias: bool = True        # Use bias in linear layers
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GPTConfig":
        """Create config from dictionary (NanoChat checkpoint format).
        
        Args:
            config_dict: Configuration dictionary from checkpoint
            
        Returns:
            GPTConfig instance
        """
        # Extract relevant parameters, use defaults for missing ones
        return cls(
            vocab_size=config_dict.get("vocab_size", 50304),
            n_layer=config_dict.get("n_layer", 12),
            n_head=config_dict.get("n_head", 12),
            n_embd=config_dict.get("n_embd", 768),
            block_size=config_dict.get("block_size", 1024),
            dropout=config_dict.get("dropout", 0.0),
            bias=config_dict.get("bias", True),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "dropout": self.dropout,
            "bias": self.bias,
        }
    
    def __str__(self) -> str:
        """String representation of config."""
        return f"GPTConfig(vocab_size={self.vocab_size}, n_layer={self.n_layer}, n_head={self.n_head}, n_embd={self.n_embd}, block_size={self.block_size})"
