"""
统一的推理引擎，处理文本生成流程
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Generator
import time
from .sampling import SamplingStrategy, GreedySampling, TopKSampling, TopPSampling, TemperatureSampling


class InferenceEngine:
    """统一的推理引擎"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.model.parameters()).device if hasattr(model, 'model') else 'cpu'
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stream: bool = False,
        stop_tokens: Optional[List[str]] = None
    ) -> Union[str, Generator[str, None, None]]:
        """生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_k: Top-K采样
            top_p: Top-P采样
            repetition_penalty: 重复惩罚
            stream: 是否流式输出
            stop_tokens: 停止token列表
            
        Returns:
            生成的文本或生成器
        """
        # 1. 编码输入
        input_ids = self._encode_prompt(prompt)
        
        # 2. 设置采样策略
        sampling_strategy = self._create_sampling_strategy(
            temperature, top_k, top_p, repetition_penalty
        )
        
        # 3. 生成tokens
        if stream:
            return self._generate_stream(
                input_ids, max_new_tokens, sampling_strategy, stop_tokens
            )
        else:
            return self._generate_batch(
                input_ids, max_new_tokens, sampling_strategy, stop_tokens
            )
    
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """编码输入提示"""
        # 使用tokenizer编码，不添加特殊token
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def _create_sampling_strategy(
        self, 
        temperature: float, 
        top_k: Optional[int], 
        top_p: Optional[float], 
        repetition_penalty: float
    ) -> SamplingStrategy:
        """创建采样策略"""
        if temperature == 0.0:
            return GreedySampling()
        else:
            strategy = TemperatureSampling(temperature)
            if top_k is not None:
                strategy = TopKSampling(strategy, top_k)
            if top_p is not None:
                strategy = TopPSampling(strategy, top_p)
            return strategy
    
    def _generate_batch(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        sampling_strategy: SamplingStrategy,
        stop_tokens: Optional[List[str]]
    ) -> str:
        """批量生成"""
        generated_ids = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # 前向传播
            logits, past_key_values = self.model.forward(
                generated_ids[:, -1:] if past_key_values is not None else generated_ids,
                past_key_values
            )
            
            # 采样下一个token
            next_token = sampling_strategy.sample(logits[:, -1, :])
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # 检查停止条件
            if self._should_stop(generated_ids, stop_tokens):
                break
        
        # 解码并返回新生成的文本
        return self._decode_new_tokens(generated_ids, input_ids.shape[1])
    
    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        sampling_strategy: SamplingStrategy,
        stop_tokens: Optional[List[str]]
    ) -> Generator[str, None, None]:
        """流式生成"""
        generated_ids = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # 前向传播
            logits, past_key_values = self.model.forward(
                generated_ids[:, -1:] if past_key_values is not None else generated_ids,
                past_key_values
            )
            
            # 采样下一个token
            next_token = sampling_strategy.sample(logits[:, -1, :])
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # 解码并yield新token
            new_text = self._decode_new_tokens(generated_ids, input_ids.shape[1])
            yield new_text
            
            # 检查停止条件
            if self._should_stop(generated_ids, stop_tokens):
                break
    
    def _should_stop(self, generated_ids: torch.Tensor, stop_tokens: Optional[List[str]]) -> bool:
        """检查是否应该停止生成"""
        if stop_tokens is None:
            return False
        
        # 获取最后生成的文本
        last_text = self._decode_new_tokens(generated_ids, generated_ids.shape[1] - 1)
        
        # 检查是否包含停止token
        for stop_token in stop_tokens:
            if stop_token in last_text:
                return True
        
        return False
    
    def _decode_new_tokens(self, generated_ids: torch.Tensor, input_length: int) -> str:
        """解码新生成的tokens"""
        new_tokens = generated_ids[0, input_length:]
        return self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
