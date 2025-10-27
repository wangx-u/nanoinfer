"""
Model checkpoint loader for NanoInfer.

Supports both HuggingFace models and NanoChat checkpoint format.
"""

import os
from typing import Tuple, Optional, Union
import torch
from .gpt import GPT
from .config import GPTConfig

# HuggingFace imports
try:
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  HuggingFace transformers not available. Install with: pip install transformers")


def load_model(
    model_path: str, 
    device: str = "cuda", 
    dtype: torch.dtype = torch.float16,
    model_type: Optional[str] = None
) -> Tuple[GPT, GPTConfig, Optional[str]]:
    """Load model from HuggingFace or .pt checkpoint file.
    
    Args:
        model_path: HuggingFace model name/path or .pt checkpoint file
        device: Device to load model on ('cuda', 'cpu', 'mps')
        dtype: Data type for model weights (torch.float16, torch.float32, etc.)
        model_type: Force model type ('hf' or 'pt'), auto-detect if None
        
    Returns:
        Tuple of (model, config, tokenizer_path)
    """
    print(f"🧠 Loading model from {model_path}...")
    
    # Auto-detect model type if not specified
    if model_type is None:
        if os.path.exists(model_path) and model_path.endswith('.pt'):
            model_type = 'pt'
        elif HF_AVAILABLE:
            model_type = 'hf'
        else:
            raise ValueError("Cannot determine model type. Please specify model_type='pt' or 'hf'")
    
    if model_type == 'hf':
        return _load_huggingface_model(model_path, device, dtype)
    elif model_type == 'pt':
        return _load_checkpoint_file(model_path, device, dtype)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _load_huggingface_model(
    model_name: str, 
    device: str = "cuda", 
    dtype: torch.dtype = torch.float16
) -> Tuple[GPT, GPTConfig, Optional[str]]:
    """Load model from HuggingFace.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        dtype: Data type for model weights
        
    Returns:
        Tuple of (model, config, tokenizer_path)
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace transformers not available. Install with: pip install transformers")
    
    print(f"📥 Loading HuggingFace model: {model_name}")
    
    # Load HuggingFace model and config
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    hf_config = AutoConfig.from_pretrained(model_name)
    
    # Convert HF config to NanoInfer config
    config_dict = {
        'vocab_size': getattr(hf_config, 'vocab_size', 50304),
        'n_layer': getattr(hf_config, 'n_layer', getattr(hf_config, 'num_hidden_layers', 12)),
        'n_head': getattr(hf_config, 'n_head', getattr(hf_config, 'num_attention_heads', 12)),
        'n_embd': getattr(hf_config, 'n_embd', getattr(hf_config, 'hidden_size', 768)),
        'block_size': getattr(hf_config, 'n_positions', getattr(hf_config, 'max_position_embeddings', 1024)),
        'dropout': getattr(hf_config, 'resid_pdrop', getattr(hf_config, 'hidden_dropout', 0.0)),
        'bias': True,
    }
    
    config = GPTConfig.from_dict(config_dict)
    
    # Create NanoInfer model
    model = GPT(config)
    
    # Map HuggingFace weights to NanoInfer format
    hf_state_dict = hf_model.state_dict()
    nanoinfer_state_dict = _map_hf_to_nanoinfer_weights(hf_state_dict, config)
    
    # Load weights
    model.load_state_dict(nanoinfer_state_dict)
    
    # Move to device and set dtype
    model = model.to(device, dtype=dtype)
    model.eval()
    
    # Tokenizer path is the same as model name
    tokenizer_path = model_name
    
    print(f"✅ HuggingFace model loaded successfully!")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Device: {device}")
    print(f"   - Dtype: {dtype}")
    print(f"   - Tokenizer: {tokenizer_path}")
    
    return model, config, tokenizer_path


def _load_checkpoint_file(
    checkpoint_path: str, 
    device: str = "cuda", 
    dtype: torch.dtype = torch.float16
) -> Tuple[GPT, GPTConfig, Optional[str]]:
    """Load model from .pt checkpoint file.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on
        dtype: Data type for model weights
        
    Returns:
        Tuple of (model, config, tokenizer_path)
    """
    print(f"📥 Loading checkpoint file: {checkpoint_path}")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract configuration
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = GPTConfig.from_dict(config_dict)
    else:
        # Fallback: try to infer config from model state dict
        print("⚠️  No config found in checkpoint, using default config")
        config = GPTConfig()
    
    # Create model
    model = GPT(config)
    
    # Load model weights
    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        # Assume checkpoint is the state dict itself
        model_state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    
    # Move to device and set dtype
    model = model.to(device, dtype=dtype)
    model.eval()
    
    # Extract tokenizer path if available
    tokenizer_path = None
    if "tokenizer_path" in checkpoint:
        tokenizer_path = checkpoint["tokenizer_path"]
    elif "tokenizer" in checkpoint:
        tokenizer_path = checkpoint["tokenizer"]
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Device: {device}")
    print(f"   - Dtype: {dtype}")
    if tokenizer_path:
        print(f"   - Tokenizer: {tokenizer_path}")
    
    return model, config, tokenizer_path


def _map_hf_to_nanoinfer_weights(hf_state_dict: dict, config: GPTConfig) -> dict:
    """Map HuggingFace weights to NanoInfer format.
    
    Args:
        hf_state_dict: HuggingFace model state dict
        config: GPTConfig object
        
    Returns:
        Mapped state dict for NanoInfer
    """
    nanoinfer_state_dict = {}
    
    # Embedding layers
    if 'transformer.wte.weight' in hf_state_dict:
        nanoinfer_state_dict['transformer.wte.weight'] = hf_state_dict['transformer.wte.weight']
    
    if 'transformer.wpe.weight' in hf_state_dict:
        nanoinfer_state_dict['transformer.wpe.weight'] = hf_state_dict['transformer.wpe.weight']
    
    # Transformer blocks
    for i in range(config.n_layer):
        # Layer normalization
        for ln_key in ['ln_1', 'ln_2']:
            for param in ['weight', 'bias']:
                hf_key = f'transformer.h.{i}.{ln_key}.{param}'
                if hf_key in hf_state_dict:
                    nanoinfer_state_dict[hf_key] = hf_state_dict[hf_key]
        
        # Attention layers - 需要转置权重
        for attn_key in ['c_attn', 'c_proj']:
            weight_key = f'transformer.h.{i}.attn.{attn_key}.weight'
            bias_key = f'transformer.h.{i}.attn.{attn_key}.bias'
            
            if weight_key in hf_state_dict:
                # HuggingFace权重格式是(in_features, out_features)，需要转置为(out_features, in_features)
                nanoinfer_state_dict[weight_key] = hf_state_dict[weight_key].T
            
            if bias_key in hf_state_dict:
                nanoinfer_state_dict[bias_key] = hf_state_dict[bias_key]
            else:
                # 如果没有 bias，创建零向量
                if attn_key == 'c_attn':
                    # c_attn 的 bias 形状是 3 * n_embd
                    bias_shape = 3 * config.n_embd
                else:
                    # c_proj 的 bias 形状是 n_embd
                    bias_shape = config.n_embd
                
                nanoinfer_state_dict[bias_key] = torch.zeros(bias_shape)
        
        # 添加 attn.bias buffer（因果掩码）
        attn_bias_key = f'transformer.h.{i}.attn.bias'
        # 创建因果掩码 buffer
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        causal_mask = causal_mask.view(1, 1, config.block_size, config.block_size)
        nanoinfer_state_dict[attn_bias_key] = causal_mask
        
        # MLP layers - 需要转置权重
        for mlp_key in ['c_fc', 'c_proj']:
            weight_key = f'transformer.h.{i}.mlp.{mlp_key}.weight'
            bias_key = f'transformer.h.{i}.mlp.{mlp_key}.bias'
            
            if weight_key in hf_state_dict:
                # HuggingFace权重格式是(in_features, out_features)，需要转置为(out_features, in_features)
                nanoinfer_state_dict[weight_key] = hf_state_dict[weight_key].T
            
            if bias_key in hf_state_dict:
                nanoinfer_state_dict[bias_key] = hf_state_dict[bias_key]
            else:
                # 如果没有 bias，创建零向量
                if mlp_key == 'c_fc':
                    # c_fc 的 bias 形状是 4 * n_embd
                    bias_shape = 4 * config.n_embd
                else:
                    # c_proj 的 bias 形状是 n_embd
                    bias_shape = config.n_embd
                
                nanoinfer_state_dict[bias_key] = torch.zeros(bias_shape)
    
    # Final layer norm
    if 'transformer.ln_f.weight' in hf_state_dict:
        nanoinfer_state_dict['transformer.ln_f.weight'] = hf_state_dict['transformer.ln_f.weight']
        nanoinfer_state_dict['transformer.ln_f.bias'] = hf_state_dict['transformer.ln_f.bias']
    
    # Language modeling head
    if 'lm_head.weight' in hf_state_dict:
        nanoinfer_state_dict['lm_head.weight'] = hf_state_dict['lm_head.weight']
    elif 'transformer.wte.weight' in hf_state_dict:
        # Use tied weights if lm_head is not separate
        nanoinfer_state_dict['lm_head.weight'] = hf_state_dict['transformer.wte.weight']
    
    return nanoinfer_state_dict


# Backward compatibility
def load_checkpoint(
    checkpoint_path: str, 
    device: str = "cuda", 
    dtype: torch.dtype = torch.float16
) -> Tuple[GPT, GPTConfig, Optional[str]]:
    """Load model from checkpoint file (backward compatibility).
    
    This function is kept for backward compatibility.
    Use load_model() for new code.
    """
    return _load_checkpoint_file(checkpoint_path, device, dtype)


def save_checkpoint(
    model: GPT, 
    config: GPTConfig, 
    save_path: str,
    tokenizer_path: Optional[str] = None
) -> None:
    """Save model checkpoint.
    
    Args:
        model: GPT model to save
        config: Model configuration
        save_path: Path to save checkpoint
        tokenizer_path: Optional path to tokenizer
    """
    checkpoint = {
        "model": model.state_dict(),
        "config": config.to_dict(),
    }
    
    if tokenizer_path:
        checkpoint["tokenizer_path"] = tokenizer_path
    
    torch.save(checkpoint, save_path)
    print(f"💾 Model saved to {save_path}")


def get_model_info(model: GPT, config: GPTConfig) -> dict:
    """Get model information.
    
    Args:
        model: GPT model
        config: Model configuration
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "vocab_size": config.vocab_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "block_size": config.block_size,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }
