"""
Core inference engine with KV cache and sampling strategies.

The heart of NanoInfer - implements autoregressive generation with various sampling methods.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Generator
import time


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply temperature scaling to logits.
    
    Args:
        logits: Input logits
        temperature: Temperature value (>0)
        
    Returns:
        Scaled logits
    """
    if temperature == 1.0:
        return logits
    return logits / temperature


def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Filter logits to top-k tokens.
    
    Args:
        logits: Input logits
        k: Number of top tokens to keep
        
    Returns:
        Filtered logits
    """
    if k <= 0:
        return logits
    
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    logits[logits < min_values] = float('-inf')
    return logits


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering.
    
    Args:
        logits: Input logits
        p: Cumulative probability threshold (0 < p <= 1)
        
    Returns:
        Filtered logits
    """
    if p >= 1.0:
        return logits
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # Convert to probabilities
    probs = F.softmax(sorted_logits, dim=-1)
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Set filtered logits to -inf
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:
    """Sample next token from logits.
    
    Args:
        logits: Model output logits of shape (B, V)
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Top-p (nucleus) filtering
        
    Returns:
        Sampled token IDs of shape (B, 1)
    """
    # Apply temperature
    logits = apply_temperature(logits, temperature)
    
    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        logits = top_k_filtering(logits, top_k)
    
    # Apply top-p filtering
    if top_p is not None and 0 < top_p < 1:
        logits = top_p_filtering(logits, top_p)
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample from distribution
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    use_cache: bool = True,
    stream: bool = False,
    stop_tokens: Optional[List[int]] = None
) -> Union[torch.Tensor, Generator[torch.Tensor, None, None]]:
    """Generate text using autoregressive decoding with KV cache.
    
    Args:
        model: GPT model
        input_ids: Input token IDs of shape (B, T)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling threshold
        top_k: Top-k sampling (None for no filtering)
        use_cache: Whether to use KV cache
        stream: Whether to yield tokens one by one
        stop_tokens: List of token IDs to stop generation
        
    Returns:
        Generated token IDs or generator for streaming
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.size(0)
    
    # Initialize past key values for KV cache
    past_key_values = None
    
    # Track generation statistics
    start_time = time.time()
    tokens_generated = 0
    
    if stream:
        def _generate_stream():
            nonlocal past_key_values, tokens_generated
            
            for step in range(max_new_tokens):
                # Forward pass with KV cache
                if use_cache and past_key_values is not None:
                    # Use KV cache: only pass the last token
                    logits, past_key_values = model(input_ids[:, -1:], past_key_values=past_key_values)
                else:
                    # First forward pass: use full input sequence
                    logits, past_key_values = model(input_ids)
                
                # Sample next token
                next_token = sample_next_token(
                    logits[:, -1, :], 
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                tokens_generated += 1
                
                # Check for stop tokens
                if stop_tokens and next_token.item() in stop_tokens:
                    break
                
                # Yield the new token
                yield next_token
                
                # Check if we've hit the context limit
                if input_ids.size(1) >= model.config.block_size:
                    break
            
            # Final statistics
            elapsed = time.time() - start_time
            print(f"⏱️  Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.1f} tokens/sec)")
        
        return _generate_stream()
    
    else:
        # Non-streaming generation
        for step in range(max_new_tokens):
            # Forward pass with KV cache
            if use_cache and past_key_values is not None:
                # Use KV cache: only pass the last token
                logits, past_key_values = model(input_ids[:, -1:], past_key_values=past_key_values)
            else:
                # First forward pass: use full input sequence
                logits, past_key_values = model(input_ids)
            
            # Sample next token
            next_token = sample_next_token(
                logits[:, -1, :], 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            tokens_generated += 1
            
            # Check for stop tokens
            if stop_tokens and next_token.item() in stop_tokens:
                break
            
            # Check if we've hit the context limit
            if input_ids.size(1) >= model.config.block_size:
                break
        
        # Final statistics
        elapsed = time.time() - start_time
        print(f"⏱️  Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.1f} tokens/sec)")
        
        return input_ids


def generate_batch(
    model: torch.nn.Module,
    prompts: List[str],
    tokenizer,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    batch_size: int = 4
) -> List[str]:
    """Generate text for multiple prompts in batches.
    
    Args:
        model: GPT model
        prompts: List of input prompts
        tokenizer: Tokenizer instance
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling threshold
        top_k: Top-k sampling
        batch_size: Batch size for processing
        
    Returns:
        List of generated texts
    """
    model.eval()
    results = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        batch_input_ids = []
        for prompt in batch_prompts:
            tokens = tokenizer.encode(prompt)
            batch_input_ids.append(tokens)
        
        # Pad to same length
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids = []
        for ids in batch_input_ids:
            padded = ids + [tokenizer.pad_token_id or 0] * (max_len - len(ids))
            padded_ids.append(padded)
        
        # Convert to tensor
        input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=next(model.parameters()).device)
        
        # Generate
        with torch.no_grad():
            generated = generate(
                model, 
                input_tensor, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        
        # Decode results
        for j, output_ids in enumerate(generated):
            # Remove input tokens
            new_tokens = output_ids[len(batch_input_ids[j]):]
            text = tokenizer.decode(new_tokens)
            results.append(text)
    
    return results
