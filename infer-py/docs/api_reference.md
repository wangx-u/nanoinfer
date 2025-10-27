# 📚 NanoInfer API 参考

> 完整的 API 文档和使用示例

## 目录

1. [核心模块](#核心模块)
2. [推理引擎](#推理引擎)
3. [优化工具](#优化工具)
4. [评估指标](#评估指标)
5. [脚本工具](#脚本工具)
6. [使用示例](#使用示例)

---

## 核心模块

### Model Loader

#### `load_checkpoint(path, device='cuda', dtype=torch.float16)`

加载模型检查点。

**参数**:
- `path` (str): 检查点文件路径
- `device` (str): 目标设备 ('cuda', 'cpu', 'mps')
- `dtype` (torch.dtype): 数据类型

**返回**:
- `model`: GPT 模型实例
- `config`: GPTConfig 配置对象
- `tokenizer_path`: Tokenizer 路径

**示例**:
```python
from nanoinfer import load_checkpoint

model, config, tokenizer_path = load_checkpoint(
    "model.pt", 
    device="cuda", 
    dtype=torch.float16
)
```

#### `GPTConfig`

模型配置类。

**属性**:
- `vocab_size` (int): 词汇表大小
- `n_layer` (int): Transformer 层数
- `n_head` (int): 注意力头数
- `n_embd` (int): 嵌入维度
- `block_size` (int): 上下文长度

**方法**:
- `from_dict(config_dict)`: 从字典创建配置
- `to_dict()`: 转换为字典

### Tokenizer

#### `Tokenizer(tokenizer_path)`

Tokenizer 包装器，支持多种格式。

**参数**:
- `tokenizer_path` (str): Tokenizer 文件路径

**方法**:
- `encode(text, bos=True, eos=False)`: 编码文本
- `decode(tokens)`: 解码 token 序列
- `__call__(text, return_tensors=None)`: 直接调用

**属性**:
- `vocab_size`: 词汇表大小
- `bos_token_id`: BOS token ID
- `eos_token_id`: EOS token ID
- `pad_token_id`: PAD token ID

**示例**:
```python
from nanoinfer import Tokenizer

tokenizer = Tokenizer("tokenizer.model")
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)
```

---

## 推理引擎

### `generate(model, input_ids, **kwargs)`

核心推理函数，支持多种生成选项。

**参数**:
- `model`: GPT 模型
- `input_ids` (torch.Tensor): 输入 token IDs
- `max_new_tokens` (int): 最大生成 token 数
- `temperature` (float): 采样温度
- `top_p` (float): Top-p 采样阈值
- `top_k` (int): Top-k 采样
- `use_cache` (bool): 是否使用 KV Cache
- `stream` (bool): 是否流式输出
- `stop_tokens` (List[int]): 停止 token 列表

**返回**:
- 生成结果或生成器（流式模式）

**示例**:
```python
from nanoinfer.engine import generate

# 非流式生成
output_ids = generate(
    model=model,
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)

# 流式生成
for token in generate(model, input_ids, stream=True):
    print(tokenizer.decode(token), end="")
```

### 采样函数

#### `apply_temperature(logits, temperature)`

应用温度缩放。

#### `top_k_filtering(logits, k)`

Top-k 过滤。

#### `top_p_filtering(logits, p)`

Top-p (nucleus) 过滤。

#### `sample_next_token(logits, temperature, top_k, top_p)`

采样下一个 token。

---

## 优化工具

### `optimize_model(model, strategy, device)`

应用优化策略。

**参数**:
- `model`: 模型实例
- `strategy` (str): 优化策略 ('fp16', 'bf16', 'compile', 'all', 'cpu_optimized')
- `device` (str): 目标设备

**示例**:
```python
from nanoinfer.plugins import optimize_model

# GPU 优化
model = optimize_model(model, strategy="fp16", device="cuda")
model = optimize_model(model, strategy="compile", device="cuda")

# CPU 优化
model = optimize_model(model, strategy="cpu_optimized", device="cpu")
model = optimize_model(model, strategy="compile", device="cpu")

# 全部优化
model = optimize_model(model, strategy="all", device="cuda")
```

### `benchmark_model(model, input_shape, num_runs)`

性能基准测试。

**参数**:
- `model`: 模型实例
- `input_shape` (tuple): 输入形状 (batch_size, seq_len)
- `num_runs` (int): 测试轮数

**返回**:
- 基准测试结果字典

### `get_memory_usage(device)`

获取显存使用情况。

### `clear_memory_cache(device)`

清理显存缓存。

---

## 评估指标

### `compute_perplexity(model, tokenizer, text_dataset)`

计算困惑度。

**参数**:
- `model`: GPT 模型
- `tokenizer`: Tokenizer 实例
- `text_dataset` (List[str]): 文本数据集

**返回**:
- 困惑度分数

### `compute_bleu(predictions, references)`

计算 BLEU 分数。

**参数**:
- `predictions` (List[str]): 预测文本
- `references` (List[str]): 参考文本

**返回**:
- BLEU 分数字典

### `compute_rouge(predictions, references)`

计算 ROUGE 分数。

### `compute_metrics(model, tokenizer, eval_data)`

计算综合评估指标。

**参数**:
- `model`: GPT 模型
- `tokenizer`: Tokenizer 实例
- `eval_data` (List[Dict]): 评估数据

**返回**:
- 评估结果字典

---

## 脚本工具

### CLI 推理

```bash
python -m scripts.chat_infer \
  --model model.pt \
  --prompt "Hello world" \
  --max_tokens 100 \
  --temperature 0.8 \
  --stream
```

### 评估脚本

```bash
python -m scripts.chat_eval_infer \
  --model model.pt \
  --eval_set eval.json \
  --output results.json
```

### 性能基准

```bash
python -m scripts.chat_bench \
  --model model.pt \
  --batch_sizes 1 4 8 \
  --seq_lengths 128 512 \
  --strategies none fp16 compile
```

### FastAPI 服务

```bash
python -m scripts.chat_server \
  --model model.pt \
  --host 0.0.0.0 \
  --port 8000
```

**API 端点**:
- `POST /api/chat`: 聊天接口
- `GET /api/health`: 健康检查
- `GET /api/model_info`: 模型信息

---

## 使用示例

### 基础推理流程

```python
from nanoinfer import load_checkpoint, Tokenizer, generate

# 1. 加载模型
model, config, tokenizer_path = load_checkpoint("model.pt")
tokenizer = Tokenizer(tokenizer_path)

# 2. 准备输入
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 3. 生成
output_ids = generate(
    model=model,
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8
)

# 4. 解码输出
response = tokenizer.decode(output_ids[0])
print(response)
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

### 流式生成

```python
# 流式输出
for token in generate(model, input_ids, stream=True):
    token_text = tokenizer.decode(token.cpu())
    print(token_text, end="", flush=True)
```

### 性能优化

```python
from nanoinfer.plugins import optimize_for_inference

# 应用所有优化
model = optimize_for_inference(model, device="cuda")

# 基准测试
from nanoinfer.plugins import benchmark_model
results = benchmark_model(model, (4, 512), num_runs=10)
print(f"Throughput: {results['tokens_per_second']:.1f} tokens/sec")
```

### 评估指标

```python
from nanoinfer.evaluation import compute_metrics

eval_data = [
    {"prompt": "What is AI?", "reference": "AI is artificial intelligence..."},
    # ... 更多样本
]

results = compute_metrics(model, tokenizer, eval_data)
print(f"Perplexity: {results['perplexity']:.2f}")
print(f"BLEU-4: {results['bleu_4']:.3f}")
```

### FastAPI 集成

```python
from fastapi import FastAPI
from nanoinfer import load_checkpoint, Tokenizer, generate

app = FastAPI()
model, config, tokenizer_path = load_checkpoint("model.pt")
tokenizer = Tokenizer(tokenizer_path)

@app.post("/chat")
async def chat(request: dict):
    prompt = request["prompt"]
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = generate(model, input_ids, max_new_tokens=100)
    response = tokenizer.decode(output_ids[0])
    return {"response": response}
```

---

## 错误处理

### 常见错误

1. **CUDA 内存不足**
   ```python
   # 解决方案：使用 FP16 或减小批处理大小
   model = model.half()
   ```

2. **Tokenizer 加载失败**
   ```python
   # 检查路径和格式
   tokenizer = Tokenizer("path/to/tokenizer.model")
   ```

3. **模型配置不匹配**
   ```python
   # 检查配置参数
   print(config)
   ```

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查模型状态
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# 监控显存使用
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
```

---

## 最佳实践

### 1. 模型加载

```python
# 推荐：使用 FP16 和优化
model, config, tokenizer_path = load_checkpoint(
    "model.pt", 
    device="cuda", 
    dtype=torch.float16
)
model = optimize_for_inference(model)
```

### 2. 推理配置

```python
# 平衡质量和速度
output_ids = generate(
    model=model,
    input_ids=input_ids,
    max_new_tokens=200,
    temperature=0.8,      # 适中的随机性
    top_p=0.9,            # 保持多样性
    use_cache=True        # 启用 KV Cache
)
```

### 3. 批处理优化

```python
# 批量处理提高吞吐量
responses = generate_batch(
    model=model,
    prompts=prompts,
    tokenizer=tokenizer,
    batch_size=8,         # 根据显存调整
    max_new_tokens=100
)
```

### 4. 性能监控

```python
# 定期检查性能
results = benchmark_model(model, (4, 512))
if results['tokens_per_second'] < 100:
    print("⚠️  Performance degradation detected")
```

---

## 扩展开发

### 自定义采样策略

```python
def custom_sampling(logits, **kwargs):
    # 实现自定义采样逻辑
    pass

# 在生成中使用
output_ids = generate(model, input_ids, custom_sampling=custom_sampling)
```

### 自定义优化器

```python
def custom_optimizer(model, **kwargs):
    # 实现自定义优化逻辑
    return model
```

### 自定义评估指标

```python
def custom_metric(predictions, references):
    # 实现自定义评估指标
    return score
```

---

## 总结

NanoInfer 提供了完整的 LLM 推理解决方案：

- **简单易用**: 几行代码即可开始推理
- **功能完整**: 支持各种采样策略和优化
- **性能优秀**: 内置多种性能优化技术
- **扩展性强**: 模块化设计，易于定制

通过这个 API 参考，你应该能够：
- 快速上手 NanoInfer
- 理解各个模块的功能
- 应用最佳实践
- 进行扩展开发

如有问题，请参考 [GitHub Issues](https://github.com/wangx-u/nanoinfer/issues) 或提交新的 issue。
