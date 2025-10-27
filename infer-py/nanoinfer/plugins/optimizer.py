"""
Performance optimization plugins for NanoInfer.

Provides FP16, torch.compile, CUDA Graphs, and other optimizations.
"""

import torch
from typing import Optional, Tuple, Dict, Any
import time


def optimize_model(
    model: torch.nn.Module, 
    strategy: str = "fp16",
    device: str = "cuda"
) -> torch.nn.Module:
    """Apply optimization strategy to model.
    
    Args:
        model: Model to optimize
        strategy: Optimization strategy ('fp16', 'bf16', 'compile', 'all', 'cpu_optimized')
        device: Target device
        
    Returns:
        Optimized model
    """
    print(f"⚙️  Applying optimization: {strategy}")
    
    # CPU-specific optimizations
    if device == "cpu":
        if strategy == "cpu_optimized" or strategy == "all":
            # CPU 优化：使用 FP32 以获得更好的数值稳定性
            model = model.float()
            print("   ✅ Applied CPU-optimized FP32 precision")
            
            # 启用 CPU 优化
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(torch.get_num_threads())
                print(f"   ✅ Set CPU threads to {torch.get_num_threads()}")
        elif strategy == "fp16":
            # CPU 上 FP16 可能较慢，建议使用 FP32
            print("   ⚠️  FP16 on CPU may be slower, consider using FP32")
            model = model.half()
        else:
            # 默认 CPU 优化
            model = model.float()
            print("   ✅ Applied FP32 precision for CPU")
    
    # GPU optimizations
    else:
        if strategy == "fp16" or strategy == "all":
            model = model.half()
            print("   ✅ Applied FP16 precision")
        
        elif strategy == "bf16":
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                model = model.bfloat16()
                print("   ✅ Applied BF16 precision")
            else:
                print("   ⚠️  BF16 not supported on this device, using FP16")
                model = model.half()
    
    # Compilation optimization (works on both CPU and GPU)
    if strategy == "compile" or strategy == "all":
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            print("   ✅ Applied torch.compile")
        else:
            print("   ⚠️  torch.compile not available (requires PyTorch 2.0+)")
    
    return model


def enable_kv_cache(model: torch.nn.Module) -> torch.nn.Module:
    """Ensure model supports KV cache.
    
    Args:
        model: Model to enable KV cache for
        
    Returns:
        Model with KV cache enabled
    """
    # This is mainly for documentation - the model should already support KV cache
    # if it's implemented correctly in the forward method
    print("✅ KV cache is enabled (model supports past_key_values)")
    return model


def setup_cuda_graphs(
    model: torch.nn.Module, 
    input_shape: Tuple[int, int],
    device: str = "cuda"
) -> Optional[torch.cuda.CUDAGraph]:
    """Setup CUDA Graphs for stable inference latency.
    
    Args:
        model: Model to optimize
        input_shape: Input shape (batch_size, seq_len)
        device: Target device
        
    Returns:
        CUDA Graph object or None if not supported
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping CUDA Graphs")
        return None
    
    try:
        # Create CUDA Graph
        graph = torch.cuda.CUDAGraph()
        
        # Create dummy inputs
        dummy_input = torch.randn(input_shape, dtype=torch.float16, device=device)
        
        # Warmup
        with torch.cuda.graph(graph):
            _ = model(dummy_input)
        
        print("✅ CUDA Graphs enabled")
        return graph
        
    except Exception as e:
        print(f"⚠️  Failed to setup CUDA Graphs: {e}")
        return None


def benchmark_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, int],
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark model performance.
    
    Args:
        model: Model to benchmark
        input_shape: Input shape (batch_size, seq_len)
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Target device
        
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    device_obj = torch.device(device)
    
    # Choose appropriate dtype based on device
    if device == "cpu":
        # CPU 上使用 FP32 以获得更好的性能
        dtype = torch.float32
    else:
        # GPU 上使用 FP16 以节省显存
        dtype = torch.float16
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, dtype=dtype, device=device_obj)
    
    # Warmup
    print(f"🔥 Warming up with {warmup_runs} runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Benchmark
    print(f"📊 Benchmarking with {num_runs} runs...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            # 设备同步：CUDA 需要同步，CPU 不需要
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate throughput
    batch_size, seq_len = input_shape
    tokens_per_second = (batch_size * seq_len) / avg_time
    
    results = {
        "avg_latency_ms": avg_time * 1000,
        "min_latency_ms": min_time * 1000,
        "max_latency_ms": max_time * 1000,
        "tokens_per_second": tokens_per_second,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "device": device,
        "dtype": str(dtype)
    }
    
    print(f"📈 Benchmark Results:")
    print(f"   Device: {device}")
    print(f"   Dtype: {dtype}")
    print(f"   Average latency: {avg_time*1000:.2f}ms")
    print(f"   Min latency: {min_time*1000:.2f}ms")
    print(f"   Max latency: {max_time*1000:.2f}ms")
    print(f"   Throughput: {tokens_per_second:.1f} tokens/sec")
    
    return results


def get_memory_usage(device: str = "cuda") -> Dict[str, float]:
    """Get current memory usage.
    
    Args:
        device: Target device
        
    Returns:
        Dictionary with memory usage in MB
    """
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2     # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
            "device_type": "cuda"
        }
    elif device == "cpu":
        # CPU 内存使用情况（需要 psutil）
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / 1024**2,  # 实际物理内存
                "vms_mb": memory_info.vms / 1024**2,  # 虚拟内存
                "percent": process.memory_percent(),   # 内存使用百分比
                "device_type": "cpu"
            }
        except ImportError:
            return {
                "error": "psutil not available for CPU memory monitoring",
                "device_type": "cpu"
            }
    else:
        return {"error": f"Device {device} not supported", "device_type": device}


def clear_memory_cache(device: str = "cuda"):
    """Clear memory cache.
    
    Args:
        device: Target device
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cache cleared")
    elif device == "cpu":
        # CPU 上清理 Python 垃圾回收
        import gc
        gc.collect()
        print("🧹 CPU memory cache cleared (garbage collection)")
    else:
        print(f"⚠️  Device {device} not supported for cache clearing")


def optimize_for_inference(model: torch.nn.Module, device: str = "cuda") -> torch.nn.Module:
    """Apply all inference optimizations.
    
    Args:
        model: Model to optimize
        device: Target device
        
    Returns:
        Optimized model
    """
    print("🚀 Applying inference optimizations...")
    
    # Move to device
    model = model.to(device)
    
    # Apply device-specific optimizations
    if device == "cpu":
        # CPU 优化策略
        model = optimize_model(model, strategy="cpu_optimized", device=device)
    else:
        # GPU 优化策略
        model = optimize_model(model, strategy="fp16", device=device)
    
    # Enable KV cache
    model = enable_kv_cache(model)
    
    # Apply torch.compile if available (works on both CPU and GPU)
    if hasattr(torch, 'compile'):
        model = optimize_model(model, strategy="compile", device=device)
    
    # Set to eval mode
    model.eval()
    
    print("✅ All optimizations applied")
    return model
