#!/usr/bin/env python3
"""
FastAPI server for NanoInfer.

Provides HTTP API for model inference with streaming support.
"""

import argparse
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from nanoinfer.model.loader import load_model
from nanoinfer.tokenizer.tokenizer import Tokenizer
from nanoinfer.engine.generator import generate
from nanoinfer.plugins.optimizer import optimize_for_inference


# Global model and tokenizer
model = None
tokenizer = None
config = None


class ChatRequest(BaseModel):
    """Chat request model."""
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: Optional[int] = None
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float


class ModelInfo(BaseModel):
    """Model information model."""
    model_path: str
    config: Dict[str, Any]
    device: str
    optimized: bool


# Create FastAPI app
app = FastAPI(
    title="NanoInfer API",
    description="Lightweight LLM inference API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model, tokenizer, config
    
    print("🚀 Starting NanoInfer API Server...")
    
    # Load model
    print(f"📁 Loading model from {args.model}...")
    model_type = None if args.model_type == 'auto' else args.model_type
    model, config, tokenizer_path = load_model(
        args.model, 
        device=args.device, 
        dtype=args.dtype,
        model_type=model_type
    )
    
    # Load tokenizer
    if args.tokenizer:
        tokenizer_path = args.tokenizer
    elif not tokenizer_path:
        raise RuntimeError("No tokenizer path provided and not found in checkpoint")
    
    print(f"🔤 Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(tokenizer_path)
    
    # Apply optimizations
    if args.optimize:
        model = optimize_for_inference(model, device=args.device)
    
    print("✅ Model loaded successfully!")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Device: {args.device}")
    print(f"   - Optimized: {args.optimize}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/api/model_info")
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_path=args.model,
        config=config.to_dict(),
        device=args.device,
        optimized=args.optimize
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode input
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(args.device)
        
        if request.stream:
            # Streaming response
            def generate_stream():
                for token in generate(
                    model=model,
                    input_ids=input_ids,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stream=True
                ):
                    token_text = tokenizer.decode(token.cpu())
                    yield f"data: {json.dumps({'token': token_text})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        else:
            # Non-streaming response
            import time
            start_time = time.time()
            
            output_ids = generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stream=False
            )
            
            # Decode response
            new_tokens = output_ids[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens)
            
            generation_time = time.time() - start_time
            tokens_generated = len(new_tokens)
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            return ChatResponse(
                response=response,
                tokens_generated=tokens_generated,
                generation_time=generation_time,
                tokens_per_second=tokens_per_second
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/batch")
async def chat_batch(requests: List[ChatRequest]):
    """Batch chat endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        responses = []
        
        for request in requests:
            # Encode input
            input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(args.device)
            
            # Generate
            output_ids = generate(
                model=model,
                input_ids=input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stream=False
            )
            
            # Decode response
            new_tokens = output_ids[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens)
            
            responses.append({
                "response": response,
                "tokens_generated": len(new_tokens)
            })
        
        return {"responses": responses}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    global args
    
    parser = argparse.ArgumentParser(description="NanoInfer FastAPI Server")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path to .pt checkpoint")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer (optional)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    parser.add_argument("--dtype", type=str, default="fp16", help="Data type (fp16/fp32)")
    parser.add_argument("--optimize", action="store_true", help="Apply optimizations")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--model-type", type=str, choices=['auto', 'hf', 'pt'], default='auto', 
                       help="Model type: auto-detect, HuggingFace, or .pt checkpoint")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("💡 CUDA not available, using CPU for inference")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("💡 MPS not available, using CPU for inference")
        args.device = "cpu"
    
    # Set dtype based on device
    if args.device == "cpu":
        # CPU 上默认使用 FP32 以获得更好的性能
        if args.dtype == "fp16":
            print("💡 Using FP16 on CPU (may be slower than FP32)")
        dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
        args.dtype = dtype_map.get(args.dtype, torch.float32)  # 默认 FP32
    else:
        # GPU 上使用 FP16 以节省显存
        dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
        args.dtype = dtype_map.get(args.dtype, torch.float16)  # 默认 FP16
    
    print("🌐 NanoInfer FastAPI Server")
    print("=" * 50)
    print(f"📁 Model: {args.model}")
    print(f"⚙️  Device: {args.device}")
    print(f"🔧 Optimized: {args.optimize}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print("=" * 50)
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
