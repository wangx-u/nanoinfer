"""
统一的模型接口，抽象不同模型实现
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Union
import torch


class BaseModel(ABC):
    """统一的模型接口"""
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, past_key_values: Optional[List] = None) -> Tuple[torch.Tensor, Optional[List]]:
        """模型前向传播
        
        Args:
            input_ids: 输入token IDs，形状 (batch_size, seq_len)
            past_key_values: KV cache，可选
            
        Returns:
            (logits, new_past_key_values)
            - logits: 形状 (batch_size, seq_len, vocab_size)
            - new_past_key_values: 新的KV cache
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        pass
    
    @abstractmethod
    def get_max_length(self) -> int:
        """获取最大序列长度"""
        pass
    
    @abstractmethod
    def to(self, device: str, dtype: torch.dtype = None):
        """移动模型到指定设备"""
        pass
    
    @abstractmethod
    def eval(self):
        """设置为评估模式"""
        pass


class HuggingFaceModel(BaseModel):
    """HuggingFace模型适配器"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._vocab_size = len(tokenizer)
        self._max_length = getattr(model.config, 'max_position_embeddings', 1024)
    
    def forward(self, input_ids: torch.Tensor, past_key_values: Optional[List] = None) -> Tuple[torch.Tensor, Optional[List]]:
        """HuggingFace模型前向传播"""
        with torch.no_grad():
            if past_key_values is not None:
                # 使用KV cache
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
                # 首次推理
                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=True
                )
            
            logits = outputs.logits
            new_past_key_values = outputs.past_key_values
            
            return logits, new_past_key_values
    
    def get_vocab_size(self) -> int:
        return self._vocab_size
    
    def get_max_length(self) -> int:
        return self._max_length
    
    def to(self, device: str, dtype: torch.dtype = None):
        self.model = self.model.to(device, dtype=dtype)
        return self
    
    def eval(self):
        self.model.eval()
        return self


class NanoInferModel(BaseModel):
    """NanoInfer自定义模型适配器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self._vocab_size = config.vocab_size
        self._max_length = config.block_size
    
    def forward(self, input_ids: torch.Tensor, past_key_values: Optional[List] = None) -> Tuple[torch.Tensor, Optional[List]]:
        """NanoInfer模型前向传播"""
        with torch.no_grad():
            return self.model(input_ids, past_key_values)
    
    def get_vocab_size(self) -> int:
        return self._vocab_size
    
    def get_max_length(self) -> int:
        return self._max_length
    
    def to(self, device: str, dtype: torch.dtype = None):
        self.model = self.model.to(device, dtype=dtype)
        return self
    
    def eval(self):
        self.model.eval()
        return self
