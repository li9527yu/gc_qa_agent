# Easy-RAG 模型更换指南

本文档说明如何在 Easy-RAG 项目中更换**向量模型 (Embedding)**、**重排模型 (Reranker)** 和**大语言模型 (LLM)**。

---

## 一、向量模型 (Embedding Model) 更换

### 1.1 配置文件

编辑 `app/embedding_config.py`：

```python
# 模型路径
EMBEDDING_MODEL_PATH = "/home/tpc/suda/rag/easy-rag/models/Qwen3-Embedding-0.6B"

# 加载器类型
EMBEDDING_TYPE = "sentence_transformer"

# 向量维度 (必须与模型输出维度一致，用于 Milvus 建表)
EMBEDDING_DIMENSION = 1024
```

### 1.2 支持的模型与加载器

| 模型 | 类型 | 维度 | 显存占用 | 推荐加载器 |
|------|------|------|----------|------------|
| `bge-large-zh-v1.5` | 通用中文 | 1024 | ~2GB | `sentence_transformer` |
| `bge-m3` | 多语言 | 1024 | ~8GB | `sentence_transformer` |
| `gte-large-zh` | 通用中文 | 1024 | ~4GB | `sentence_transformer` / `modelscope` |
| `qwen3-embedding-0.6b` | Qwen3 | 1024 | ~1.5GB | `sentence_transformer` / `qwen3` |
| `qwen3-embedding-4b` | Qwen3 | 2560 (支持 MRL 降维) | ~8GB | `qwen3` |
| `xiaobu-embedding` | 通用中文 | 4096 | ~16GB | `sentence_transformer` |

### 1.3 加载器类型说明

- **`sentence_transformer`** (推荐)：通过 `sentence-transformers` 库加载，兼容性最好，支持绝大多数开源 Embedding 模型。
- **`modelscope`**：通过 `modelscope` 的 `AutoModel` / `AutoTokenizer` 加载，适合国内模型（如 GTE、部分 BGE）。
- **`qwen3`**：专用加载器，仅用于 Qwen3-Embedding 系列，支持 **MRL 降维** 和 **32K 长文本**。

### 1.4 更换步骤

1. **下载新模型**到 `models/` 目录（或指定绝对路径）。
2. 修改 `app/embedding_config.py`：
   - `EMBEDDING_MODEL_PATH`：新模型路径
   - `EMBEDDING_TYPE`：选择合适的加载器
   - `EMBEDDING_DIMENSION`：与模型实际输出维度一致
3. **重建向量数据库**：
   ```bash
   # 删除缓存和 Milvus 集合，启动时会自动重建
   rm tokenized_docs.pkl
   # 如使用 Milvus standalone，需手动删除对应 collection
   ```
4. 重启应用服务。

### 1.5 注意事项

- **维度必须一致**：`EMBEDDING_DIMENSION` 必须与模型实际输出维度严格一致，否则 Milvus 写入会报错。
- **Qwen3 MRL 降维**：若使用 `qwen3-embedding-4b` 且想降维，设置 `EMBEDDING_DIMENSION < 2560`（如 1024），并在 `EMBEDDING_TYPE = "qwen3"` 模式下会自动生效。
- **显存分配**：可通过 `EMBEDDING_DEVICE = "cuda:1"` 指定模型加载到非 LLM 占用的 GPU。

---

## 二、重排模型 (Reranker) 更换

### 2.1 配置文件

同样在 `app/embedding_config.py` 中修改：

```python
# 模型路径
RERANKER_MODEL_PATH = "/home/tpc/suda/rag/easy-rag/models/Qwen3-Reranker-0.6B"

# 是否启用重排
USE_RERANKER = True

# 设备分配
RERANKER_DEVICE = "cuda:1"
```

### 2.2 支持的模型

| 模型 | 架构 | 显存占用 | 特点 |
|------|------|----------|------|
| `bge-reranker-large` | Cross-Encoder | ~2GB | 经典重排模型，稳定可靠 |
| `qwen3-reranker-0.6b` | Cross-Encoder | ~1.5GB | 轻量，效果优异，推荐 |
| `qwen3-reranker-4b` | Cross-Encoder | ~8GB | 大参数，精度更高 |

### 2.3 自动检测机制

`app/reranker.py` 会根据模型路径**自动识别**模型类型：

- 路径中包含 `qwen3`（不区分大小写）→ 使用 `Qwen3Reranker` 专用封装
- 其他路径 → 使用标准 `AutoTokenizer` + `AutoModelForSequenceClassification` 加载（BGE 系列）

### 2.4 更换步骤

1. 下载新模型到本地目录。
2. 修改 `app/embedding_config.py` 中的 `RERANKER_MODEL_PATH`。
3. 重启应用服务即可（**无需重建向量库**）。

### 2.5 注意事项

- 若 GPU 显存紧张，可将 `RERANKER_DEVICE` 设为 `"cpu"`，但重排速度会显著下降。
- 如不需要重排阶段，设置 `USE_RERANKER = False` 可直接跳过，减少推理耗时。

---

## 三、大语言模型 (LLM) 更换

### 3.1 配置文件

编辑 `app/config.py`：

```python
# 选择 LLM 提供商/部署方式
USE_LLM = "local_vllm_quantized"

# DeepSeek API 配置
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_API_KEY = "sk-xxx"
DEEPSEEK_MODEL = "deepseek-chat"

# 本地原始模型
LOCAL_API_BASE = "http://localhost:9839/v1"
LOCAL_MODEL = "Qwen3-32B"

# AWQ 量化
LOCAL_AWQ_API_BASE = "http://localhost:9843/v1"
LOCAL_AWQ_MODEL = "qwen3-30b-a3b-instruct-awq"

# GGUF 量化
LOCAL_QUANTIZED_API_BASE = "http://localhost:9841/v1"
LOCAL_QUANTIZED_MODEL = "qwen3-30b-a3b-instruct-gguf-q4km"

# vLLM 量化
LOCAL_VLLM_QUANTIZED_API_BASE = "http://localhost:9844/v1"
LOCAL_VLLM_QUANTIZED_MODEL = "./models/quantized-qwen3-30b-a3b-instruct-vllm"
```

### 3.2 支持的 LLM 模式

| `USE_LLM` 值 | 说明 | 适用场景 |
|--------------|------|----------|
| `"deepseek"` | DeepSeek 官方 API | 快速接入，无需本地 GPU |
| `"local"` | 本地部署的原始模型 | 有充足显存，追求最佳效果 |
| `"local_awq"` | AWQ 量化模型 | 显存有限，需要较快推理 |
| `"local_quantized"` | GGUF 量化模型 (llama.cpp) | CPU/GPU 混合，低显存 |
| `"local_vllm_quantized"` | vLLM 量化模型 (GPTQ/AWQ) | **推荐**，响应最快 (~0.7s) |

### 3.3 更换步骤

#### 方式 A：切换到其他 API 服务

1. 修改 `app/config.py` 中的 `USE_LLM` 为 `"deepseek"` 或 `"local"` 等。
2. 配置对应的 `API_BASE`、`API_KEY`、`MODEL` 名称。
3. 重启应用即可。

#### 方式 B：部署新的本地模型

1. **准备模型文件**：
   - 原始模型：直接下载 HuggingFace 完整权重
   - vLLM 量化：使用 `quantize_with_vllm_fixed.py` 生成量化模型
   - GGUF：下载 `.gguf` 文件，通过 `llama.cpp` 启动
2. **启动模型服务**（需 OpenAI 兼容 API）：
   ```bash
   # vLLM 示例
   vllm serve ./models/your-new-model \
       --port 9844 \
       --tensor-parallel-size 1 \
       --dtype half \
       --max-model-len 4096
   ```
3. 修改 `app/config.py`：
   - 新增/修改对应的 `LOCAL_XXX_API_BASE` 和 `LOCAL_XXX_MODEL`
   - 设置 `USE_LLM = "local_vllm_quantized"`（或对应模式）
4. 重启应用。

### 3.4 注意事项

- **OpenAI 兼容接口**：所有本地模型必须通过提供 `/v1/chat/completions` 接口的服务启动（vLLM、llama.cpp server、TGI 等），因为 `llm_deepseek.py` 和 `llm_openai.py` 底层使用 OpenAI 客户端调用。
- **模型名称**：`LLM_MODEL_NAME` 会传递给 API 的 `model` 参数，需与模型服务注册的名称一致。
- **显存要求**：
  - vLLM 量化模型：建议 ≥ 16GB GPU
  - 原始 30B+ 模型：建议 ≥ 60GB GPU 或多卡并行
- **上下文长度**：确保 `--max-model-len` 或模型配置能覆盖 RAG 拼接后的 Prompt 长度。

---

## 四、快速检查清单

更换任意模型后，请按以下清单确认：

| 检查项 | Embedding | Reranker | LLM |
|--------|-----------|----------|-----|
| 修改配置文件 | `embedding_config.py` | `embedding_config.py` | `config.py` |
| 确认路径/名称正确 | ✅ | ✅ | ✅ |
| 确认维度一致 | ✅ | - | - |
| 重建向量数据库 | ✅ | - | - |
| 重启应用服务 | ✅ | ✅ | ✅ |
| 测试一次完整问答 | ✅ | ✅ | ✅ |

---

## 五、常见问题

### Q1: 更换 Embedding 模型后 Milvus 报错 "dimension mismatch"
**A**: `EMBEDDING_DIMENSION` 与新模型实际输出维度不一致。请核对模型文档中的输出维度，并删除旧向量库后重建。

### Q2: 本地 LLM 服务已启动，但应用提示连接失败
**A**: 检查 `app/config.py` 中的 `OPENAI_API_BASE`（由 `USE_LLM` 自动推导）是否与模型服务实际端口一致，并确认服务支持 `/v1/chat/completions`。

### Q3: 想同时使用新 Embedding 和旧 LLM，只换一部分可以吗？
**A**: 可以。三类模型配置完全独立，单独更换其中任意一类不会影响其他模型。

### Q4: Reranker 可以关闭吗？
**A**: 可以。设置 `USE_RERANKER = False`，检索结果将直接返回，不再经过重排阶段。
