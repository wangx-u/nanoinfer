"""
NanoInfer: The smallest, clearest, and most educational inference pipeline for LLMs.

From checkpoint to conversation — in 100 lines of code.
"""

__version__ = "0.1.0"
__author__ = "WangXu"

# Core imports for easy access
from .model.gpt import GPT
from .model.config import GPTConfig
from .model.loader import load_checkpoint
from .engine.generator import generate
from .tokenizer.tokenizer import Tokenizer

__all__ = [
    "GPT",
    "GPTConfig", 
    "load_checkpoint",
    "generate",
    "Tokenizer"
]
