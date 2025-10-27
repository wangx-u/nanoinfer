# 🧠 NanoInfer 推理教程

> 从零理解 LLM 推理的全过程

## 目录

1. [推理原理](#推理原理)
2. [KV Cache 机制](#kv-cache-机制)
3. [采样策略详解](#采样策略详解)
4. [性能优化指南](#性能优化指南)
5. [实战示例](#实战示例)

---

## 推理原理

### 自回归生成

LLM 推理的核心是**自回归生成**：模型逐个预测下一个 token，直到生成结束。

```python
# 伪代码：自回归生成循环
def generate(model, prompt):
    tokens = tokenize(prompt)
    
    for i in range(max_length):
        # 1. 前向传播
        logits = model(tokens)
        
        # 2. 采样下一个 token
        next_token = sample(logits[-1])
        
        # 3. 添加到序列
        tokens.append(next_token)
        
        # 4. 检查结束条件
        if is_end_token(next_token):
            break
    
    return detokenize(tokens)
```

### 关键概念

- **Logits**: 模型输出的原始分数，表示每个 token 的"可能性"
- **Temperature**: 控制生成的随机性（0.1=确定性，1.0=标准，2.0=高随机性）
- **Top-p (Nucleus)**: 只从累积概率达到 p 的 token 中采样
- **Top-k**: 只从概率最高的 k 个 token 中采样

---

## KV Cache 机制

### 问题：重复计算

在自回归生成中，每次只生成一个新 token，但模型需要重新计算整个序列的注意力：

```python
# 没有 KV Cache：每次都重新计算
for i in range(3):
    # 第1次：计算 [token1] 的注意力
    # 第2次：计算 [token1, token2] 的注意力  ← 重复计算 token1
    # 第3次：计算 [token1, token2, token3] 的注意力  ← 重复计算 token1, token2
```

### 解决方案：缓存 Key 和 Value

```python
# 有 KV Cache：只计算新 token
past_key_values = None
for i in range(3):
    # 只计算新 token 的 K, V
    new_k, new_v = compute_kv(new_token)
    
    # 拼接历史 K, V
    k = concat([past_k, new_k])
    v = concat([past_v, new_v])
    
    # 缓存供下次使用
    past_key_values = (k, v)
```

### 实现细节

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x, past_key_values=None):
        # 计算当前 token 的 Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 如果有历史 KV，拼接
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)  # 拼接时间维度
            v = torch.cat([past_v, v], dim=2)
        
        # 计算注意力
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.bias == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
        
        # 返回输出和新的 KV
        return out, (k, v)
```

### 性能提升

- **时间复杂度**: O(N²) → O(N)
- **实际加速**: 2-4× 推理速度提升
- **内存开销**: 增加 ~50% 显存使用

---

## 采样策略详解

### 1. Greedy 采样

选择概率最高的 token：

```python
def greedy_sampling(logits):
    return torch.argmax(logits, dim=-1)
```

**特点**: 确定性，适合需要一致性的场景

### 2. Temperature 采样

通过温度参数控制随机性：

```python
def temperature_sampling(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**温度效果**:
- `temperature = 0.1`: 几乎确定性，重复性高
- `temperature = 1.0`: 标准随机性
- `temperature = 2.0`: 高随机性，创意性强

### 3. Top-k 采样

只从概率最高的 k 个 token 中采样：

```python
def top_k_sampling(logits, k):
    values, indices = torch.topk(logits, k)
    logits[logits < values[:, -1:]] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 4. Top-p (Nucleus) 采样

只从累积概率达到 p 的 token 中采样：

```python
def top_p_sampling(logits, p):
    # 排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    # 移除超出阈值的 token
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 应用掩码
    logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 采样策略对比

| 策略 | 确定性 | 多样性 | 适用场景 |
|------|--------|--------|----------|
| Greedy | 高 | 低 | 代码生成、数学问题 |
| Temperature | 中 | 中 | 通用对话 |
| Top-k | 中 | 中 | 平衡质量和多样性 |
| Top-p | 低 | 高 | 创意写作、头脑风暴 |

---

## 性能优化指南

### 1. 精度优化

#### GPU 优化
```python
# FP16 半精度（GPU 推荐）
model = model.half()  # 显存减半，速度提升 1.5-2×

# BF16 脑浮点（GPU 推荐）
model = model.bfloat16()  # 更好的数值稳定性
```

#### CPU 优化
```python
# CPU 上使用 FP32 以获得更好的性能
model = model.float()  # CPU 上 FP32 通常比 FP16 更快

# 设置 CPU 线程数
torch.set_num_threads(4)  # 根据 CPU 核心数调整
```

### 2. 编译优化

#### Torch Compile
```python
model = torch.compile(model)  # PyTorch 2.0+ 自动图优化
```

### 3. 批处理优化

```python
# 单条推理
for prompt in prompts:
    generate(model, prompt)

# 批处理推理
batch_prompts = [prompt1, prompt2, prompt3]
generate_batch(model, batch_prompts)  # 3-5× 吞吐提升
```

### 4. 内存优化

#### GPU 内存管理
```python
# 清理显存
torch.cuda.empty_cache()

# 监控显存使用
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")
```

#### CPU 内存管理
```python
# 清理 CPU 内存
import gc
gc.collect()

# 监控 CPU 内存使用
import psutil
process = psutil.Process()
print(f"RSS: {process.memory_info().rss / 1024**2:.1f}MB")
print(f"VMS: {process.memory_info().vms / 1024**2:.1f}MB")
```

### 5. CUDA Graphs（高级）

```python
# 预编译推理图，减少启动开销
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(input_tensor)
```

---

## 实战示例

### 基础推理

```python
from nanoinfer import load_checkpoint, Tokenizer, generate

# 加载模型
model, config, tokenizer_path = load_checkpoint("model.pt")
tokenizer = Tokenizer(tokenizer_path)

# 单轮推理
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = generate(
    model=model,
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)
response = tokenizer.decode(output_ids[0])
print(response)
```

### 流式生成

```python
# 流式输出
for token in generate(model, input_ids, stream=True):
    token_text = tokenizer.decode(token.cpu())
    print(token_text, end="", flush=True)
```

### 批量推理

```python
from nanoinfer.engine.generator import generate_batch

prompts = [
    "Explain quantum computing",
    "Write a haiku about AI",
    "What is machine learning?"
]

responses = generate_batch(
    model=model,
    prompts=prompts,
    tokenizer=tokenizer,
    max_new_tokens=100,
    batch_size=4
)
```

### 性能基准测试

```python
from nanoinfer.plugins.optimizer import benchmark_model

# 测试不同配置
results = benchmark_model(
    model=model,
    input_shape=(4, 512),  # batch_size=4, seq_len=512
    num_runs=10
)

print(f"Average latency: {results['avg_latency_ms']:.2f}ms")
print(f"Throughput: {results['tokens_per_second']:.1f} tokens/sec")
```

### 评估指标

```python
from nanoinfer.evaluation.metrics import compute_metrics

# 加载评估数据
eval_data = [
    {"prompt": "What is AI?", "reference": "AI is artificial intelligence..."},
    # ... 更多样本
]

# 计算指标
results = compute_metrics(model, tokenizer, eval_data)
print(f"Perplexity: {results['perplexity']:.2f}")
print(f"BLEU-4: {results['bleu_4']:.3f}")
print(f"ROUGE-L: {results['rougeL']:.3f}")
```

---

## 常见问题

### Q: 为什么生成速度慢？
A: 检查以下几点：
1. 是否启用了 KV Cache
2. 是否使用了 FP16 精度
3. 是否应用了 torch.compile
4. 批处理大小是否合适

### Q: 显存不足怎么办？
A: 尝试以下方法：
1. 使用 FP16 或 BF16
2. 减小批处理大小
3. 使用梯度检查点
4. 考虑模型量化

### Q: CPU 推理太慢怎么办？
A: 优化策略：
1. 使用 FP32 而不是 FP16（CPU 上 FP32 通常更快）
2. 调整 CPU 线程数：`torch.set_num_threads(4)`
3. 启用 torch.compile 优化
4. 使用较小的批处理大小
5. 考虑使用更小的模型

### Q: 生成质量不好？
A: 调整采样参数：
1. 降低 temperature（更确定性）
2. 调整 top_p 值（0.9-0.95）
3. 使用 top_k 限制候选
4. 检查 prompt 质量

### Q: 如何提高吞吐量？
A: 优化策略：
1. 增加批处理大小
2. 使用批处理推理
3. 启用 CUDA Graphs
4. 考虑模型并行

---

## 进阶技巧

### 1. 动态停止条件

```python
def generate_with_stop(model, input_ids, stop_tokens=[], max_tokens=200):
    for token in generate(model, input_ids, stream=True):
        if token.item() in stop_tokens:
            break
        yield token
```

### 2. 条件生成

```python
def conditional_generate(model, input_ids, condition_fn):
    for token in generate(model, input_ids, stream=True):
        if condition_fn(token):
            break
        yield token
```

### 3. 多轮对话

```python
def multi_turn_chat(model, tokenizer, conversation_history):
    # 构建完整上下文
    context = tokenizer.encode(conversation_history)
    
    # 生成回复
    response = generate(model, context)
    
    return tokenizer.decode(response)
```

---

## 总结

NanoInfer 的设计哲学是**从零理解推理**：

1. **透明性**: 每个步骤都可以追踪和调试
2. **教育性**: 代码清晰，注释详细
3. **实用性**: 支持生产级优化和部署
4. **扩展性**: 模块化设计，易于定制

通过这个教程，你应该能够：
- 理解 LLM 推理的核心机制
- 掌握 KV Cache 的工作原理
- 选择合适的采样策略
- 应用性能优化技巧
- 构建完整的推理系统

记住：**理解原理比追求性能更重要**。先让代码工作，再让它跑得更快。
