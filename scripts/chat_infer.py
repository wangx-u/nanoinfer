#!/usr/bin/env python3

import argparse
import sys
import os
import torch
from pathlib import Path

from nanoinfer.model.factory import create_model
from nanoinfer.engine.inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="NanoInfer CLI Inference")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--model-type", type=str, choices=['auto', 'hf', 'nanoinfer'], default='auto', 
                       help="Model type")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=20, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    parser.add_argument("--top_k", type=int, help="Top-k sampling (optional)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda/cpu/mps)")
    parser.add_argument("--dtype", type=str, default="fp32", help="Data type (fp16/fp32)")
    parser.add_argument("--stream", action="store_true", help="Stream output")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("💡 CUDA not available, using CPU for inference")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("💡 MPS not available, using CPU for inference")
        args.device = "cpu"
    
    # Set dtype
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype, torch.float32)
    
    print("🧠 NanoInfer CLI Inference")
    print("=" * 50)
    
    # 1. 创建模型和tokenizer
    print(f"📁 Loading model from {args.model}...")
    model, tokenizer = create_model(
        args.model,
        model_type=args.model_type,
        device=args.device,
        dtype=dtype
    )
    
    # 2. 创建推理引擎
    print("🔧 Creating inference engine...")
    engine = InferenceEngine(model, tokenizer)
    
    # 3. 生成文本
    print(f"📝 Input prompt: {args.prompt}")
    print("=" * 50)
    
    if args.stream:
        print("🤖 Generating response (streaming):")
        print("-" * 30)
        
        for chunk in engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=True
        ):
            print(chunk, end="", flush=True)
        
        print("\n" + "-" * 30)
    else:
        print("🤖 Generating response...")
        
        response = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=False
        )
        
        print("🤖 Model response:")
        print("-" * 30)
        print(response)
        print("-" * 30)
    
    print("✅ Inference completed!")


if __name__ == "__main__":
    main()
