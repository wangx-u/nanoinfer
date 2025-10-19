"""
采样策略模块
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional


class SamplingStrategy(ABC):
    """采样策略基类"""
    
    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """从logits中采样token
        
        Args:
            logits: 形状 (batch_size, vocab_size)
            
        Returns:
            采样的token IDs，形状 (batch_size, 1)
        """
        pass


class GreedySampling(SamplingStrategy):
    """贪心采样"""
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True)


class TemperatureSampling(SamplingStrategy):
    """温度采样"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature == 1.0:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
        
        return torch.multinomial(probs, num_samples=1)


class TopKSampling(SamplingStrategy):
    """Top-K采样"""
    
    def __init__(self, base_strategy: SamplingStrategy, k: int):
        self.base_strategy = base_strategy
        self.k = k
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # 应用Top-K过滤
        if self.k > 0:
            values, _ = torch.topk(logits, min(self.k, logits.size(-1)))
            min_values = values[..., -1, None]
            logits = logits.clone()
            logits[logits < min_values] = float('-inf')
        
        # 使用基础策略采样
        return self.base_strategy.sample(logits)


class TopPSampling(SamplingStrategy):
    """Top-P (Nucleus) 采样"""
    
    def __init__(self, base_strategy: SamplingStrategy, p: float):
        self.base_strategy = base_strategy
        self.p = p
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # 应用Top-P过滤
        if 0 < self.p < 1.0:
            # 排序
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            
            # 计算累积概率
            cumulative_probs = torch.cumsum(probs, dim=-1)
            
            # 移除超过阈值的tokens
            sorted_indices_to_remove = cumulative_probs > self.p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            # 应用过滤
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.clone()
            logits[indices_to_remove] = float('-inf')
        
        # 使用基础策略采样
        return self.base_strategy.sample(logits)


class RepetitionPenaltySampling(SamplingStrategy):
    """重复惩罚采样"""
    
    def __init__(self, base_strategy: SamplingStrategy, penalty: float = 1.0):
        self.base_strategy = base_strategy
        self.penalty = penalty
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # 这里需要访问历史tokens，暂时简化实现
        return self.base_strategy.sample(logits)
