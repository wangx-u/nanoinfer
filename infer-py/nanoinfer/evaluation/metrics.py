"""
Evaluation metrics for NanoInfer.

Implements perplexity, BLEU, ROUGE, and other evaluation metrics.
"""

import math
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    text_dataset: List[str],
    device: str = "cuda",
    batch_size: int = 8
) -> float:
    """Compute perplexity on a text dataset.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer instance
        text_dataset: List of text samples
        device: Target device
        batch_size: Batch size for processing
        
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    print(f"📊 Computing perplexity on {len(text_dataset)} samples...")
    
    with torch.no_grad():
        for i in range(0, len(text_dataset), batch_size):
            batch_texts = text_dataset[i:i + batch_size]
            
            # Tokenize batch
            batch_tokens = []
            for text in batch_texts:
                tokens = tokenizer.encode(text)
                batch_tokens.append(tokens)
            
            # Pad to same length
            max_len = max(len(tokens) for tokens in batch_tokens)
            padded_tokens = []
            for tokens in batch_tokens:
                padded = tokens + [tokenizer.pad_token_id or 0] * (max_len - len(tokens))
                padded_tokens.append(padded)
            
            # Convert to tensor
            input_ids = torch.tensor(padded_tokens, dtype=torch.long, device=device)
            
            # Forward pass
            logits, _ = model(input_ids)
            
            # Compute loss (cross-entropy)
            # Shift logits and targets for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Flatten for loss computation
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id or 0,
                reduction='sum'
            )
            
            # Count non-padding tokens
            valid_tokens = (shift_labels != (tokenizer.pad_token_id or 0)).sum().item()
            
            total_loss += loss.item()
            total_tokens += valid_tokens
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"✅ Perplexity: {perplexity:.2f}")
    return perplexity


def compute_bleu(
    predictions: List[str], 
    references: List[str],
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """Compute BLEU scores.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        weights: BLEU weights for different n-grams
        
    Returns:
        Dictionary with BLEU scores
    """
    if weights is None:
        weights = [(1.0,), (0.5, 0.5), (0.33, 0.33, 0.33), (0.25, 0.25, 0.25, 0.25)]
    
    # Tokenize texts
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [ref.split() for ref in references]
    
    # Compute BLEU scores
    bleu_scores = {}
    smoothing = SmoothingFunction().method1
    
    for i, weight in enumerate(weights):
        if len(weight) == 1:
            # BLEU-1
            scores = [sentence_bleu([ref], pred, weights=weight, smoothing_function=smoothing) 
                     for pred, ref in zip(pred_tokens, ref_tokens)]
        elif len(weight) == 2:
            # BLEU-2
            scores = [sentence_bleu([ref], pred, weights=weight, smoothing_function=smoothing) 
                     for pred, ref in zip(pred_tokens, ref_tokens)]
        elif len(weight) == 3:
            # BLEU-3
            scores = [sentence_bleu([ref], pred, weights=weight, smoothing_function=smoothing) 
                     for pred, ref in zip(pred_tokens, ref_tokens)]
        elif len(weight) == 4:
            # BLEU-4
            scores = [sentence_bleu([ref], pred, weights=weight, smoothing_function=smoothing) 
                     for pred, ref in zip(pred_tokens, ref_tokens)]
        
        avg_score = sum(scores) / len(scores)
        bleu_scores[f"bleu_{len(weight)}"] = avg_score
    
    print(f"✅ BLEU scores: {bleu_scores}")
    return bleu_scores


def compute_rouge(
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]:
    """Compute ROUGE scores.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeL'] += scores['rougeL'].fmeasure
    
    # Average scores
    num_samples = len(predictions)
    for key in rouge_scores:
        rouge_scores[key] /= num_samples
    
    print(f"✅ ROUGE scores: {rouge_scores}")
    return rouge_scores


def compute_metrics(
    model: torch.nn.Module,
    tokenizer,
    eval_data: List[Dict[str, str]],
    device: str = "cuda",
    batch_size: int = 8
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer instance
        eval_data: List of dicts with 'prompt' and 'reference' keys
        device: Target device
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with all metrics
    """
    print(f"📊 Computing metrics on {len(eval_data)} samples...")
    
    # Extract prompts and references
    prompts = [item['prompt'] for item in eval_data]
    references = [item['reference'] for item in eval_data]
    
    # Generate predictions
    from ..engine.generator import generate_batch
    
    predictions = generate_batch(
        model=model,
        prompts=prompts,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.8,
        batch_size=batch_size
    )
    
    # Compute metrics
    results = {}
    
    # BLEU scores
    bleu_scores = compute_bleu(predictions, references)
    results.update(bleu_scores)
    
    # ROUGE scores
    rouge_scores = compute_rouge(predictions, references)
    results.update(rouge_scores)
    
    # Perplexity (on references)
    ppl = compute_perplexity(model, tokenizer, references, device, batch_size)
    results['perplexity'] = ppl
    
    # Additional statistics
    results['num_samples'] = len(eval_data)
    results['avg_pred_length'] = sum(len(pred.split()) for pred in predictions) / len(predictions)
    results['avg_ref_length'] = sum(len(ref.split()) for ref in references) / len(references)
    
    print(f"📈 Evaluation Results:")
    print(f"   Perplexity: {ppl:.2f}")
    print(f"   BLEU-4: {bleu_scores.get('bleu_4', 0):.3f}")
    print(f"   ROUGE-L: {rouge_scores.get('rougeL', 0):.3f}")
    print(f"   Avg pred length: {results['avg_pred_length']:.1f}")
    print(f"   Avg ref length: {results['avg_ref_length']:.1f}")
    
    return results


def save_eval_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save results
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to {output_path}")


def load_eval_data(data_path: str) -> List[Dict[str, str]]:
    """Load evaluation data from JSON file.
    
    Args:
        data_path: Path to JSON file with evaluation data
        
    Returns:
        List of evaluation samples
    """
    import json
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Ensure data has required keys
    for item in data:
        if 'prompt' not in item or 'reference' not in item:
            raise ValueError("Evaluation data must contain 'prompt' and 'reference' keys")
    
    print(f"📁 Loaded {len(data)} evaluation samples from {data_path}")
    return data
