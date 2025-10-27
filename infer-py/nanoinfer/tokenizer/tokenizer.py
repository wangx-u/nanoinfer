"""
Tokenizer wrapper for NanoInfer.

Supports loading existing tokenizers (NanoChat format, HuggingFace format).
"""

import os
from typing import List, Optional, Union
import torch
from transformers import AutoTokenizer


class Tokenizer:
    """Tokenizer wrapper supporting multiple formats."""
    
    def __init__(self, tokenizer_path: str):
        """Initialize tokenizer from path.
        
        Args:
            tokenizer_path: Path to tokenizer file or directory
        """
        self.tokenizer_path = tokenizer_path
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load tokenizer based on file extension or HuggingFace model name."""
        # Try HuggingFace format first (works for both local paths and model names)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self._is_hf = True
            print(f"✅ Loaded HuggingFace tokenizer from {self.tokenizer_path}")
        except Exception:
            # Try sentencepiece format (only for local files)
            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")
            
            try:
                import sentencepiece as spm
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.load(self.tokenizer_path)
                self._is_hf = False
                print(f"✅ Loaded SentencePiece tokenizer from {self.tokenizer_path}")
            except Exception as e:
                raise ValueError(f"Failed to load tokenizer from {self.tokenizer_path}: {e}")
    
    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to encode
            bos: Add beginning-of-sequence token
            eos: Add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        if self._is_hf:
            # HuggingFace tokenizer
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if bos and self.tokenizer.bos_token_id is not None:
                tokens = [self.tokenizer.bos_token_id] + tokens
            if eos and self.tokenizer.eos_token_id is not None:
                tokens = tokens + [self.tokenizer.eos_token_id]
        else:
            # SentencePiece tokenizer
            tokens = self.sp_model.encode(text, out_type=int)
            if bos:
                tokens = [self.sp_model.bos_id()] + tokens
            if eos:
                tokens = tokens + [self.sp_model.eos_id()]
        
        return tokens
    
    def decode(self, tokens: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs to text.
        
        Args:
            tokens: Token IDs to decode
            
        Returns:
            Decoded text
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        if self._is_hf:
            # HuggingFace tokenizer
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            # SentencePiece tokenizer
            return self.sp_model.decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._is_hf:
            return len(self.tokenizer)
        else:
            return self.sp_model.get_piece_size()
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """Get BOS token ID."""
        if self._is_hf:
            return self.tokenizer.bos_token_id
        else:
            return self.sp_model.bos_id()
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Get EOS token ID."""
        if self._is_hf:
            return self.tokenizer.eos_token_id
        else:
            return self.sp_model.eos_id()
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Get PAD token ID."""
        if self._is_hf:
            return self.tokenizer.pad_token_id
        else:
            return self.sp_model.pad_id()
    
    def __call__(self, text: str, return_tensors: Optional[str] = None, **kwargs) -> Union[List[int], torch.Tensor]:
        """Call tokenizer directly.
        
        Args:
            text: Input text
            return_tensors: If 'pt', return PyTorch tensor
            **kwargs: Additional arguments
            
        Returns:
            Token IDs as list or tensor
        """
        tokens = self.encode(text, **kwargs)
        
        if return_tensors == "pt":
            return torch.tensor(tokens, dtype=torch.long)
        return tokens
