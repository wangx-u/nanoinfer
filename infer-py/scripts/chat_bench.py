#!/usr/bin/env python3
"""
Performance benchmark script for NanoInfer.

Tests different configurations for throughput and latency.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from nanoinfer.model.loader import load_model
from nanoinfer.tokenizer.tokenizer import Tokenizer
from nanoinfer.plugins.optimizer import benchmark_model, get_memory_usage, optimize_model


def run_benchmark(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    batch_sizes: List[int],
    seq_lengths: List[int],
    dtypes: List[torch.dtype],
    strategies: List[str],
    device: str = "cuda",
    num_runs: int = 5
) -> Dict[str, Any]:
    """Run comprehensive benchmark tests.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer instance
        batch_sizes: List of batch sizes to test
        seq_lengths: List of sequence lengths to test
        dtypes: List of data types to test
        strategies: List of optimization strategies
        device: Target device
        num_runs: Number of runs per test
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "device": device,
        "model_info": {
            "total_params": sum(p.numel() for p in model.parameters()),
            "vocab_size": tokenizer.vocab_size
        },
        "benchmarks": []
    }
    
    print("🚀 Starting comprehensive benchmark...")
    print("=" * 60)
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            for dtype in dtypes:
                for strategy in strategies:
                    print(f"\n📊 Testing: batch={batch_size}, seq={seq_len}, dtype={dtype}, strategy={strategy}")
                    
                    try:
                        # Create test model copy
                        test_model = model.clone() if hasattr(model, 'clone') else model
                        test_model = test_model.to(dtype)
                        
                        # Apply optimization strategy
                        if strategy != "none":
                            test_model = optimize_model(test_model, strategy=strategy, device=device)
                        
                        # Create dummy input
                        input_shape = (batch_size, seq_len)
                        dummy_input = torch.randn(input_shape, dtype=dtype, device=device)
                        
                        # Benchmark
                        benchmark_results = benchmark_model(
                            test_model,
                            input_shape,
                            num_runs=num_runs,
                            warmup_runs=2,
                            device=device
                        )
                        
                        # Get memory usage
                        memory_usage = get_memory_usage(device)
                        
                        # Store results
                        test_result = {
                            "batch_size": batch_size,
                            "seq_length": seq_len,
                            "dtype": str(dtype),
                            "strategy": strategy,
                            "latency_ms": benchmark_results["avg_latency_ms"],
                            "min_latency_ms": benchmark_results["min_latency_ms"],
                            "max_latency_ms": benchmark_results["max_latency_ms"],
                            "tokens_per_second": benchmark_results["tokens_per_second"],
                            "memory_allocated_mb": memory_usage.get("allocated_mb", 0),
                            "memory_reserved_mb": memory_usage.get("reserved_mb", 0)
                        }
                        
                        results["benchmarks"].append(test_result)
                        
                        print(f"   ✅ Latency: {benchmark_results['avg_latency_ms']:.2f}ms")
                        print(f"   ✅ Throughput: {benchmark_results['tokens_per_second']:.1f} tokens/sec")
                        print(f"   ✅ Memory: {memory_usage.get('allocated_mb', 0):.1f}MB")
                        
                    except Exception as e:
                        print(f"   ❌ Failed: {e}")
                        continue
    
    return results


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark results and provide insights.
    
    Args:
        results: Benchmark results
        
    Returns:
        Analysis summary
    """
    benchmarks = results["benchmarks"]
    
    if not benchmarks:
        return {
            "error": "No successful benchmarks",
            "total_tests": 0,
            "successful_tests": 0,
            "best_latency": None,
            "best_throughput": None,
            "best_memory": None,
            "strategy_comparison": {}
        }
    
    # Find best configurations
    best_latency = min(benchmarks, key=lambda x: x["latency_ms"])
    best_throughput = max(benchmarks, key=lambda x: x["tokens_per_second"])
    best_memory = min(benchmarks, key=lambda x: x["memory_allocated_mb"])
    
    # Calculate averages by strategy
    strategy_stats = {}
    for benchmark in benchmarks:
        strategy = benchmark["strategy"]
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                "latencies": [],
                "throughputs": [],
                "memories": []
            }
        
        strategy_stats[strategy]["latencies"].append(benchmark["latency_ms"])
        strategy_stats[strategy]["throughputs"].append(benchmark["tokens_per_second"])
        strategy_stats[strategy]["memories"].append(benchmark["memory_allocated_mb"])
    
    # Calculate averages
    for strategy in strategy_stats:
        stats = strategy_stats[strategy]
        stats["avg_latency"] = sum(stats["latencies"]) / len(stats["latencies"])
        stats["avg_throughput"] = sum(stats["throughputs"]) / len(stats["throughputs"])
        stats["avg_memory"] = sum(stats["memories"]) / len(stats["memories"])
    
    analysis = {
        "best_latency": best_latency,
        "best_throughput": best_throughput,
        "best_memory": best_memory,
        "strategy_comparison": strategy_stats,
        "total_tests": len(benchmarks),
        "successful_tests": len([b for b in benchmarks if b["latency_ms"] > 0])
    }
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Print benchmark analysis.
    
    Args:
        analysis: Analysis results
    """
    print("\n📈 Benchmark Analysis")
    print("=" * 60)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    print(f"📊 Total tests: {analysis['total_tests']}")
    print(f"✅ Successful: {analysis['successful_tests']}")
    
    print("\n🏆 Best Configurations:")
    print("-" * 30)
    
    best_latency = analysis["best_latency"]
    print(f"⚡ Best Latency: {best_latency['latency_ms']:.2f}ms")
    print(f"   Config: batch={best_latency['batch_size']}, seq={best_latency['seq_length']}, dtype={best_latency['dtype']}, strategy={best_latency['strategy']}")
    
    best_throughput = analysis["best_throughput"]
    print(f"🚀 Best Throughput: {best_throughput['tokens_per_second']:.1f} tokens/sec")
    print(f"   Config: batch={best_throughput['batch_size']}, seq={best_throughput['seq_length']}, dtype={best_throughput['dtype']}, strategy={best_throughput['strategy']}")
    
    best_memory = analysis["best_memory"]
    print(f"💾 Best Memory: {best_memory['memory_allocated_mb']:.1f}MB")
    print(f"   Config: batch={best_memory['batch_size']}, seq={best_memory['seq_length']}, dtype={best_memory['dtype']}, strategy={best_memory['strategy']}")
    
    print("\n🔧 Strategy Comparison:")
    print("-" * 30)
    for strategy, stats in analysis["strategy_comparison"].items():
        print(f"{strategy:>10}: latency={stats['avg_latency']:.1f}ms, throughput={stats['avg_throughput']:.1f} tok/s, memory={stats['avg_memory']:.1f}MB")


def main():
    parser = argparse.ArgumentParser(description="NanoInfer Performance Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model name or path (HuggingFace model ID or .pt checkpoint)")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer (optional)")
    parser.add_argument("--model-type", type=str, choices=['auto', 'hf', 'pt'], default='auto', 
                       help="Model type: auto-detect, HuggingFace, or .pt checkpoint")
    parser.add_argument("--output", type=str, help="Path to save results (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8], help="Batch sizes to test")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[128, 512], help="Sequence lengths to test")
    parser.add_argument("--dtypes", type=str, nargs="+", default=["fp16", "fp32"], help="Data types to test")
    parser.add_argument("--strategies", type=str, nargs="+", default=["none", "fp16", "compile", "all"], help="Optimization strategies")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs per test")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        args.device = "cpu"
    
    # Convert dtype strings to torch dtypes based on device
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    if args.device == "cpu":
        # CPU 上过滤掉不合适的 dtype
        cpu_dtypes = []
        for dtype_str in args.dtypes:
            if dtype_str == "fp16":
                print("⚠️  FP16 on CPU may be slower, consider using FP32")
            cpu_dtypes.append(dtype_map.get(dtype_str, torch.float32))
        dtypes = cpu_dtypes
    else:
        dtypes = [dtype_map.get(dtype, torch.float16) for dtype in args.dtypes]
    
    print("⚡ NanoInfer Performance Benchmark")
    print("=" * 60)
    
    # Load model
    print(f"📁 Loading model from {args.model}...")
    model_type = None if args.model_type == 'auto' else args.model_type
    model, config, tokenizer_path = load_model(
        args.model, 
        device=args.device, 
        dtype=torch.float16,
        model_type=model_type
    )
    
    # Load tokenizer
    if args.tokenizer:
        tokenizer_path = args.tokenizer
    elif not tokenizer_path:
        print("❌ No tokenizer path provided and not found in checkpoint")
        sys.exit(1)
    
    print(f"🔤 Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(tokenizer_path)
    
    print(f"🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"⚙️  Device: {args.device}")
    print(f"📊 Tests: {len(args.batch_sizes)} × {len(args.seq_lengths)} × {len(dtypes)} × {len(args.strategies)} = {len(args.batch_sizes) * len(args.seq_lengths) * len(dtypes) * len(args.strategies)} total")
    print("=" * 60)
    
    # Run benchmark
    results = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        dtypes=dtypes,
        strategies=args.strategies,
        device=args.device,
        num_runs=args.num_runs
    )
    
    # Analyze results
    analysis = analyze_results(results)
    print_analysis(analysis)
    
    # Save results
    if args.output:
        full_results = {
            "benchmark_results": results,
            "analysis": analysis,
            "config": {
                "model_path": args.model,
                "device": args.device,
                "batch_sizes": args.batch_sizes,
                "seq_lengths": args.seq_lengths,
                "dtypes": args.dtypes,
                "strategies": args.strategies,
                "num_runs": args.num_runs
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\n💾 Results saved to {args.output}")
    
    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()
