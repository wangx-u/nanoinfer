import torch
from typing import Tuple, Optional
from .base import BaseModel, HuggingFaceModel, NanoInferModel
from .loader import load_model as load_nanoinfer_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_model(
    model_name: str,
    model_type: str = "auto",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> Tuple[BaseModel, object]:
    """创建模型和tokenizer
    
    Args:
        model_name: 模型名称或路径
        model_type: 模型类型 ("hf", "nanoinfer", "auto")
        device: 设备
        dtype: 数据类型
        
    Returns:
        (model, tokenizer)
    """
    if model_type == "auto":
        model_type = _detect_model_type(model_name)
    
    if model_type == "hf":
        return _create_huggingface_model(model_name, device, dtype)
    elif model_type == "nanoinfer":
        return _create_nanoinfer_model(model_name, device, dtype)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _detect_model_type(model_name: str) -> str:
    """自动检测模型类型"""
    import os
    if os.path.exists(model_name) and model_name.endswith('.pt'):
        return "nanoinfer"
    else:
        return "hf"


def _create_huggingface_model(
    model_name: str, 
    device: str, 
    dtype: torch.dtype
) -> Tuple[BaseModel, object]:
    """创建HuggingFace模型"""
    print(f"📥 Loading HuggingFace model: {model_name}")
    
    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=dtype,
        device_map="auto" if device != "cpu" else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 移动到指定设备
    if device != "auto":
        model = model.to(device)
    
    # 创建适配器
    hf_model = HuggingFaceModel(model, tokenizer)
    
    print(f"✅ HuggingFace model loaded successfully!")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Device: {device}")
    print(f"   - Dtype: {dtype}")
    
    return hf_model, tokenizer


def _create_nanoinfer_model(
    model_path: str, 
    device: str, 
    dtype: torch.dtype
) -> Tuple[BaseModel, object]:
    """创建NanoInfer模型"""
    print(f"📥 Loading NanoInfer model: {model_path}")
    
    # 使用现有的loader
    model, config, tokenizer_path = load_nanoinfer_model(
        model_path, device=device, dtype=dtype
    )
    
    # 加载tokenizer
    from ..tokenizer.tokenizer import Tokenizer
    tokenizer = Tokenizer(tokenizer_path)
    
    # 创建适配器
    nanoinfer_model = NanoInferModel(model, config)
    
    print(f"✅ NanoInfer model loaded successfully!")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Device: {device}")
    print(f"   - Dtype: {dtype}")
    
    return nanoinfer_model, tokenizer
