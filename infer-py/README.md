# 🧠 NanoInfer  
> The smallest, clearest, and most educational inference pipeline for LLMs.  
> **From checkpoint to conversation — in 100 lines of code.**

---

## ✨ What is NanoInfer?

**NanoInfer** 是一个极简推理框架 / 教程，旨在从底层剖析 **LLM 推理阶段** 的全过程。  
它继承了 [NanoChat](https://github.com/karpathy/nanochat) 的理念：  
> “Train small, understand deeply.”

NanoInfer 让你仅使用 PyTorch 理解一个完整的自回归推理循环：  
- 如何从 HF模型 或 `.safetensors` 权重文件加载模型  
- 如何执行 `generate()` 推理循环（含 KV cache）  
- 如何构建 tokenizer + sampling  
- 如何做推理评估与部署  

目标：**理解推理 → 优化推理 → 服务推理**

---

## 🧩 Features

| 模块 | 功能 | 技术关键词 |
|------|------|-------------|
| 🧠 Model Loader | 支持 HuggingFace 和 .pt 格式 | HuggingFace / PyTorch |
| 🔡 Tokenizer | 兼容 HuggingFace tokenizer | AutoTokenizer / BPE |
| 🔁 Generator | 自回归生成 | KV Cache / Sampling |
| 📊 Evaluator | 计算 PPL / BLEU / CORE | 轻量指标计算 |
| 🌐 Chat Server | 快速部署 Web 聊天 | FastAPI / HuggingFace |
| ⚙️ Accelerator | 性能优化 | FP16 / Torch Compile / CUDA Graphs |

---

## 📦 Installation

```bash
git clone https://github.com/yourname/nanoinfer.git
cd nanoinfer
pip install -r requirements.txt
````

Requirements:

* Python ≥ 3.10
* PyTorch ≥ 2.1
* transformers, numpy, tqdm, fastapi, uvicorn

**新增**: 现在默认支持 HuggingFace 模型，无需转换！

---

## 🚀 Quick Start

运行单轮推理（最小示例）：

```bash
# 使用 HuggingFace 模型（推荐）
python -m scripts.chat_infer --model gpt2 --prompt "Explain transformers in simple terms."

# 使用 .pt 检查点文件
python -m scripts.chat_infer --model model.pt --prompt "Explain transformers in simple terms."

# CPU 推理
python -m scripts.chat_infer --model gpt2 --device cpu --dtype fp32 --prompt "Explain transformers in simple terms."
```

输出：

```
🧠 Loading tokenizer and model...
🗣️ Model output:
Transformers are neural networks that learn context by attending to all words at once.
⏱️ Inference time: 0.83s for 200 tokens
```

---


## 🤗 HuggingFace 模型支持

NanoInfer 现在默认支持 HuggingFace 模型，无需任何转换！

### 支持的模型

- **GPT-2 系列**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **DialoGPT 系列**: `microsoft/DialoGPT-small`, `microsoft/DialoGPT-medium`
- **其他 GPT 架构模型**

### 快速使用

```bash
# 直接使用 HuggingFace 模型
python -m scripts.chat_infer --model gpt2 --prompt "Hello world"

# 使用更大的模型
python -m scripts.chat_infer --model gpt2-medium --prompt "The future of AI"

# 启动 API 服务器
python -m scripts.chat_server --model gpt2 --host 0.0.0.0 --port 8000
```

### 编程接口

```python
from nanoinfer.model.loader import load_model
from nanoinfer.tokenizer.tokenizer import Tokenizer

# 加载 HuggingFace 模型
model, config, tokenizer_path = load_model("gpt2", device="cuda")
tokenizer = Tokenizer(tokenizer_path)

# 使用模型进行推理
input_ids = tokenizer.encode("Hello world", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=50)
response = tokenizer.decode(output[0])
```

---

## 🧠 Inference Pipeline

```python
# pseudo-code overview

from nanoinfer.model.loader import load_model
from nanoinfer.engine.generator import generate
from nanoinfer.tokenizer.tokenizer import Tokenizer

# 加载 HuggingFace 模型（推荐）
model, config, tokenizer_path = load_model("gpt2", device="cuda")
tokenizer = Tokenizer(tokenizer_path)

# 或者加载 .pt 检查点
# model, config, tokenizer_path = load_model("model.pt", device="cuda")

prompt = "The theory of attention is"
ids = tokenizer.encode(prompt, return_tensors="pt")

out = generate(model, ids, max_new_tokens=128, temperature=0.8)
print(tokenizer.decode(out[0]))
```

支持以下功能：

* KV Cache（O(N) 推理）
* Temperature / Top-p / Top-k 采样
* FP16 + CUDA Graphs 加速
* 批量生成（Batch Decode）

---

## 🧪 Evaluation

```bash
# 使用 HuggingFace 模型进行评估
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval_infer \
  --model gpt2 \
  --eval_set ~/nanobase/eval_bundle/core_eval.json

# 或使用 .pt 检查点
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval_infer \
  --model ~/nanobase/models/sft_model.pt \
  --eval_set ~/nanobase/eval_bundle/core_eval.json
```

Metrics:

| 指标                   | 含义       |
| -------------------- | -------- |
| **Perplexity (PPL)** | 语言建模困惑度  |
| **CORE metric**      | 复杂推理任务评估 |
| **BLEU / ROUGE**     | 自然语言生成质量 |

---

## ⚡ Performance Tips

| 优化项                 | 描述                | GPU 效果      | CPU 效果      |
| ------------------- | ----------------- | --------- | --------- |
| **KV cache**        | 缓存注意力层历史状态        | ⏱️ 2–4×   | ⏱️ 2–3×   |
| **FP16**            | 半精度推理             | 💾 显存减半   | ⚠️ 可能更慢   |
| **FP32**            | 单精度推理（CPU 推荐）     | 💾 显存更多   | 🚀 通常更快   |
| **torch.compile()** | PyTorch 2.x 自动图优化 | 🚀 20–40% | 🚀 30–50% |
| **Batch Decode**    | 多 prompt 并行生成     | 🔁 吞吐提升   | 🔁 吞吐提升   |
| **CUDA Graphs**     | 预编译推理图（仅 GPU）    | ⚙️ 稳定延迟   | ❌ 不支持     |

---

## 🌐 Serve It

启动交互界面：

```bash
python -m scripts.chat_server
```

访问 👉 [http://localhost:8000](http://localhost:8000)

示例：

```python
POST /api/chat
{
  "prompt": "Write a haiku about GPU inference."
}
```

---

## 🧭 Roadmap

* [x] 基础推理循环 (CPU/GPU)
* [ ] CORE 任务评估
* [ ] FastAPI 服务接口
* [ ] CPU 优化支持
* [ ] 性能基准测试
* [x] Streaming 生成接口
* [ ] Tensor Parallel Serving
* [ ] 动态 KV cache 管理

---

## 🧩 Relationship to NanoChat

| 项目                                               | 功能阶段 | 核心目标       |
| ------------------------------------------------ | ---- | ---------- |
| [NanoChat](https://github.com/karpathy/nanochat) | 训练   | 从零理解语言模型训练 |
| **NanoInfer**                                    | 推理   | 从零理解语言模型推理 |

> “NanoChat teaches your model to think.
> NanoInfer teaches it to speak.”
