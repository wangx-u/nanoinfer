#!/usr/bin/env python3
"""
Evaluation script for NanoInfer.

Computes perplexity, BLEU, ROUGE and other metrics on evaluation datasets.
"""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from nanoinfer.model.loader import load_checkpoint
from nanoinfer.tokenizer.tokenizer import Tokenizer
from nanoinfer.evaluation.metrics import compute_metrics, save_eval_results, load_eval_data
from nanoinfer.plugins.optimizer import optimize_for_inference


def main():
    parser = argparse.ArgumentParser(description="NanoInfer Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer (optional)")
    parser.add_argument("--eval_set", type=str, required=True, help="Path to evaluation dataset JSON")
    parser.add_argument("--output", type=str, help="Path to save results (optional)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    parser.add_argument("--dtype", type=str, default="fp16", help="Data type (fp16/fp32)")
    parser.add_argument("--optimize", action="store_true", help="Apply optimizations")
    parser.add_argument("--metrics", type=str, default="all", help="Metrics to compute (all/ppl/bleu/rouge)")
    
    args = parser.parse_args()
    
    print("📊 NanoInfer Evaluation")
    print("=" * 50)
    
    # Load model
    print(f"📁 Loading model from {args.model}...")
    model, config, tokenizer_path = load_checkpoint(
        args.model, 
        device=args.device, 
        dtype=args.dtype
    )
    
    # Load tokenizer
    if args.tokenizer:
        tokenizer_path = args.tokenizer
    elif not tokenizer_path:
        print("❌ No tokenizer path provided and not found in checkpoint")
        sys.exit(1)
    
    print(f"🔤 Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer(tokenizer_path)
    
    # Apply optimizations
    if args.optimize:
        model = optimize_for_inference(model, device=args.device)
    
    # Load evaluation data
    print(f"📋 Loading evaluation data from {args.eval_set}...")
    eval_data = load_eval_data(args.eval_set)
    
    print(f"📊 Evaluating on {len(eval_data)} samples...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Temperature: {args.temperature}")
    print("=" * 50)
    
    # Compute metrics
    results = compute_metrics(
        model=model,
        tokenizer=tokenizer,
        eval_data=eval_data,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Add evaluation metadata
    results['evaluation_config'] = {
        'model_path': args.model,
        'eval_set': args.eval_set,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'device': args.device,
        'optimized': args.optimize
    }
    
    # Save results
    if args.output:
        save_eval_results(results, args.output)
    
    # Print summary
    print("\n📈 Evaluation Summary:")
    print("=" * 50)
    print(f"📊 Dataset: {len(eval_data)} samples")
    print(f"🧠 Model: {config}")
    print(f"⚙️  Device: {args.device}")
    print(f"🔧 Optimized: {args.optimize}")
    print("-" * 30)
    print(f"📉 Perplexity: {results.get('perplexity', 0):.2f}")
    print(f"📊 BLEU-1: {results.get('bleu_1', 0):.3f}")
    print(f"📊 BLEU-2: {results.get('bleu_2', 0):.3f}")
    print(f"📊 BLEU-3: {results.get('bleu_3', 0):.3f}")
    print(f"📊 BLEU-4: {results.get('bleu_4', 0):.3f}")
    print(f"📊 ROUGE-1: {results.get('rouge1', 0):.3f}")
    print(f"📊 ROUGE-2: {results.get('rouge2', 0):.3f}")
    print(f"📊 ROUGE-L: {results.get('rougeL', 0):.3f}")
    print(f"📏 Avg pred length: {results.get('avg_pred_length', 0):.1f}")
    print(f"📏 Avg ref length: {results.get('avg_ref_length', 0):.1f}")
    
    if args.output:
        print(f"\n💾 Results saved to {args.output}")
    
    print("✅ Evaluation completed!")


if __name__ == "__main__":
    main()
