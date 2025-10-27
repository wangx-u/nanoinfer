"""
GPT model implementation with KV Cache support.

Simplified version compatible with NanoChat format.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with KV cache support."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Register causal mask as buffer
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x: torch.Tensor, past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional KV cache.
        
        Args:
            x: Input tensor of shape (B, T, C)
            past_key_values: Tuple of (past_key, past_value) for caching
            
        Returns:
            Tuple of (output, (key, value)) for next iteration
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # If we have past key/values, concatenate them
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)  # (B, nh, T+past_T, hs)
            v = torch.cat([past_v, v], dim=2)  # (B, nh, T+past_T, hs)
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Calculate the correct mask size based on full sequence length (including cached keys)
        full_seq_len = k.size(2)  # This includes both cached and current tokens
        current_seq_len = T  # Current input length
        
        # Create proper causal mask for the full sequence
        if full_seq_len <= self.bias.size(2):
            # Use cached mask if available
            mask_slice = self.bias[:, :, :current_seq_len, :full_seq_len]
            # Broadcast mask to match attention shape
            mask_slice = mask_slice.expand(att.size(0), att.size(1), -1, -1)
            att = att.masked_fill(mask_slice == 0, float('-inf'))
        else:
            # Create dynamic mask for longer sequences
            causal_mask = torch.tril(torch.ones(current_seq_len, full_seq_len, device=att.device))
            # Broadcast mask to match attention shape
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(att.size(0), att.size(1), -1, -1)
            att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor, past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, T, C)
            past_key_values: Tuple of (past_key, past_value) for caching
            
        Returns:
            Tuple of (output, (key, value)) for next iteration
        """
        # Self-attention with residual connection
        attn_out, kv_cache = self.attn(self.ln_1(x), past_key_values)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x, kv_cache


class GPT(nn.Module):
    """GPT model with KV cache support."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params:,}")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx: torch.Tensor, past_key_values: Optional[list] = None) -> Tuple[torch.Tensor, Optional[list]]:
        """Forward pass with optional KV cache.
        
        Args:
            idx: Input token indices of shape (B, T)
            past_key_values: List of (key, value) tuples for each layer
            
        Returns:
            Tuple of (logits, new_past_key_values)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Calculate position encoding correctly for KV cache
        if past_key_values is not None and len(past_key_values) > 0:
            # When using KV cache, we need to calculate the position based on cached sequence length
            past_length = past_key_values[0][0].size(2) if past_key_values[0][0] is not None else 0
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device).unsqueeze(0)
        else:
            # First forward pass, use normal position encoding
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        
        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # Token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # Position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        new_past_key_values = []
        for i, block in enumerate(self.transformer.h):
            if past_key_values is not None:
                past_kv = past_key_values[i] if i < len(past_key_values) else None
            else:
                past_kv = None
            x, kv_cache = block(x, past_kv)
            new_past_key_values.append(kv_cache)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, new_past_key_values
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate text using the model.
        
        Args:
            idx: Input token indices of shape (B, T)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for no filtering)
            
        Returns:
            Generated token indices
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
