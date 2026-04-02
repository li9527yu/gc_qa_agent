# Easy-RAG 项目指南

## 项目概述

Easy-RAG 是一个基于 RAG (Retrieval-Augmented Generation) 的智能问答系统，专注于工程造价领域的知识问答和价格推荐。

### 核心功能

1. **知识问答**: 基于上传的文档语料库进行智能问答
2. **价格推荐**: 从多个渠道（信息价、厂商报价、智诚信息价）查询材料价格
3. **意图识别**: 自动识别用户问题是知识问答还是价格查询
4. **文件管理**: 支持文档上传、删除和知识库自动更新
5. **流式响应**: 支持流式返回 LLM 生成的回答

### 技术架构

- **后端框架**: FastAPI + Python 3.9
- **向量数据库**: Milvus (支持 standalone 和 lite 模式)
- **检索组件**: BM25 + 向量检索 (混合检索)
- **嵌入模型**: BGE-large-zh-v1.5
- **重排序模型**: BGE-reranker-large
- **LLM 支持**: DeepSeek API / 本地量化模型 (GGUF/AWQ/vLLM)
- **部署方式**: Docker + Docker Compose

## 项目结构

```
/home/tpc/suda/rag/easy-rag/
├── app/                          # 主应用代码
│   ├── api/                      # API 路由
│   │   └── endpoints.py          # REST API 接口定义
│   ├── core/                     # 核心组件
│   │   └── logger.py             # 日志配置
│   ├── schemas/                  # Pydantic 数据模型
│   │   ├── files.py              # 文件相关模型
│   │   ├── material.py           # 材料数据模型
│   │   ├── price.py              # 价格查询模型
│   │   ├── rag.py                # RAG 请求/响应模型
│   │   └── request_response.py   # 通用请求响应模型
│   ├── services/                 # 业务逻辑层
│   │   ├── file_processor.py     # 文件处理服务
│   │   └── rag_service.py        # RAG 核心服务
│   ├── utils/                    # 工具函数
│   │   ├── data_filter_helper.py # 数据量检查和 Token 估算
│   │   ├── fillter_tools.py      # 数据过滤工具
│   │   ├── generate_signature.py # API 签名生成
│   │   ├── markdown_converter.py # Markdown 转换工具
│   │   ├── price_tools.py        # 价格分析工具
│   │   └── prompt_template.py    # 提示词模板
│   ├── api_server.py             # FastAPI 应用入口
│   ├── better_split.py           # 文档分块优化
│   ├── config.py                 # 全局配置
│   ├── llm_deepseek.py           # LLM 客户端封装
│   ├── llm_openai.py             # OpenAI 风格 API 封装
│   ├── read_corpus.py            # 语料库读取
│   ├── reranker.py               # 重排序模块
│   ├── retriever.py              # 检索模块
│   └── tools.py                  # 其他工具函数
├── tests/                        # 测试代码
│   ├── test_data_volume_control.py
│   └── test_mock_api.py
├── text2sql/                     # Text-to-SQL 相关
├── models/                       # 模型文件目录
├── logs/                         # 日志目录
├── volumes/                      # Docker 数据卷
├── docs/                         # 文档目录
├── docker-compose.yml            # Milvus 服务编排
├── Dockerfile                    # Docker 构建文件
├── schema.sql                    # PostgreSQL 数据库初始化脚本
├── requirements.txt              # Python 依赖
├── start_vllm_quantized_model.sh # 启动 vLLM 量化模型
├── start_awq_model.sh            # 启动 AWQ 量化模型
├── start_original_model.sh       # 启动原始模型
├── start_all_models.sh           # 启动所有模型服务
└── quantize_with_vllm_fixed.py   # vLLM 模型量化脚本
```

## 核心技术栈

### Python 依赖 (requirements.txt)

- **Web 框架**: FastAPI 0.104.1, Uvicorn 0.24.0
- **AI/ML**: 
  - PyTorch 2.7.1 (CUDA 12.x)
  - Transformers 4.53.0
  - LangChain 0.2.0 + LangChain-Milvus 0.1.10
  - sentence-transformers 4.1.0
  - vLLM (用于量化模型推理)
- **向量数据库**: pymilvus 2.3.4, milvus-lite 2.4.12
- **数据处理**: pandas 1.5.3, numpy 1.23.5, jieba 0.42.1
- **HTTP 客户端**: httpx 0.27.0, requests 2.32.3
- **其他**: pydantic 2.5.0, gradio 4.32.0

### 模型配置

1. **嵌入模型**: `/home/tpc/suda/rag/model/bge-large-zh-v1.5`
2. **重排序模型**: `/home/tpc/suda/rag/model/bge-reranker-large`
3. **LLM 模型**: Qwen3-30B-A3B-Instruct (支持多种量化格式)

## 配置说明

### 1. LLM 配置 (app/config.py)

```python
# 选择 LLM 提供商: "deepseek", "local", "local_awq", "local_quantized", "local_vllm_quantized"
USE_LLM = "deepseek"

# DeepSeek API 配置
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_API_KEY = "your-api-key"

# vLLM 量化模型 (推荐，响应最快)
LOCAL_VLLM_QUANTIZED_API_BASE = "http://localhost:9844/v1"

# GGUF 量化模型
LOCAL_QUANTIZED_API_BASE = "http://localhost:9841/v1"

# AWQ 量化模型
LOCAL_AWQ_API_BASE = "http://localhost:9843/v1"
```

### 2. 数据源配置

```python
# 价格查询 API
USE_MOCK_API = False  # True 使用 Mock 数据，False 使用真实 API
MOCK_API_BASE_URL = "https://m1.apifoxmock.com/..."
REAL_API_BASE_URL = "http://113.31.103.48:8801/backend/largeModelMaterial"

# 知识库数据路径
DATA_PATH = "/home/tpc/suda/rag/easy-rag/app/dataset/data/"
```

### 3. Milvus 配置

```yaml
# docker-compose.yml 中配置
- standalone 模式: 使用 Docker 部署完整 Milvus 服务
- lite 模式: 使用 milvus_rag.db 本地文件
```

## 启动与运行

### 1. 启动 Milvus 向量数据库

```bash
cd /home/tpc/suda/rag/easy-rag
docker-compose up -d
```

### 2. 启动 LLM 服务 (选择一种)

```bash
# 方案1: vLLM 量化模型 (推荐，响应时间 ~0.72秒)
bash start_vllm_quantized_model.sh

# 方案2: GGUF 量化模型 (响应时间 ~6.77秒)
bash start_awq_model.sh

# 方案3: 原始模型
bash start_original_model.sh

# 方案4: 启动所有模型
bash start_all_models.sh
```

### 3. 启动主应用

```bash
# 开发模式
python -m app.api_server

# 或使用 uvicorn
uvicorn app.api_server:app --host 0.0.0.0 --port 8001 --reload
```

### 4. 服务端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| Easy-RAG API | 8001 | 主应用服务 |
| Milvus | 19530 | 向量数据库 |
| GGUF 模型 | 9841 | llama.cpp 服务 |
| vLLM 量化 | 9844 | vLLM 服务 |
| AWQ 模型 | 9843 | AWQ 量化服务 |

## API 接口说明

### 核心接口流程

1. **意图识别**: `POST /api/v1/query/intent`
   - 判断用户问题是知识问答还是价格推荐
   - 返回: `knowledge_qa` 或 `price_recommendation`

2. **知识问答流式接口**: `POST /api/v1/query/stream`
   - 基于文档语料库进行 RAG 问答
   - 支持流式返回和上下文记忆

3. **价格推荐流式接口**: `POST /api/v1/query/price`
   - 从信息价/厂商报价/智诚信息价查询材料价格
   - 支持数据量检查和 Token 优化

4. **直接价格查询**: `POST /api/v1/query/price/direct`
   - 接收材料信息列表直接查询价格
   - 用于前端直接传递材料数据场景

### 文件管理接口

- `POST /api/v1/upload` - 批量上传文件
- `GET /api/v1/files` - 获取文件列表
- `DELETE /api/v1/delete/batch` - 批量删除文件
- `GET /api/v1/task_status?task_id=xxx` - 查询任务状态

### 测试脚本

```bash
# 测试 Mock API
python tests/test_mock_api.py

# 测试响应时间
python test_response_time.py

# 性能测试
python comprehensive_performance_test.py
```

## 开发规范

### 代码风格

- 使用 **中文注释** 和 **中文文档**
- 遵循 PEP 8 代码规范
- 使用类型注解提高代码可读性
- 异步函数使用 `async/await` 模式

### 日志规范

```python
import logging
logger = logging.getLogger("easy_rag_api")

# 性能统计日志格式
logger.info("=" * 80)
logger.info("📊 性能统计标题")
logger.info("=" * 80)
logger.info(f"⏱️  耗时: {time:.3f}s")
```

### 错误处理

```python
try:
    result = await process()
except Exception as e:
    logger.error(f"操作失败: {e}")
    return {
        "success": False,
        "error_message": str(e)
    }
```

## 模型量化说明

### 支持的量化格式

| 格式 | 路径 | 大小 | 响应时间 | 推荐指数 |
|------|------|------|----------|----------|
| 原始模型 | Qwen3-30B-A3B-Instruct-2507 | ~60GB | ~30s+ | ⭐ |
| GGUF Q4_K_M | qwen3-30b-a3b-instruct.Q4_K_M.gguf | ~18GB | ~6.77s | ⭐⭐⭐ |
| vLLM GPTQ | quantized-qwen3-30b-a3b-instruct-vllm | ~15GB | ~0.72s | ⭐⭐⭐⭐⭐ |

### 量化模型启动

```bash
# 创建量化模型 (使用 vLLM)
python quantize_with_vllm_fixed.py

# 启动量化服务
vllm serve ./models/quantized-qwen3-30b-a3b-instruct-vllm \
    --port 9844 \
    --tensor-parallel-size 1 \
    --dtype half \
    --max-model-len 4096
```

## 数据库设计

### PostgreSQL 数据库 (schema.sql)

主要表结构:
- `document_version` - 文档版本管理
- `document_chunk` - 文档分块存储
- `document_update_task` - 文档更新任务
- `sys_user` / `sys_role` - 用户权限系统
- `query_history` - 查询历史记录
- `price_query_record` - 价格查询记录

### Milvus 集合

- **Collection**: `easy_rag_milvus`
- **Embedding 维度**: 1024 (BGE-large-zh)
- **索引类型**: IVF_FLAT (standalone) / FLAT (lite)
- **度量方式**: L2 距离

## 部署说明

### Docker 部署

```bash
# 构建镜像
docker build -t easy-rag:latest .

# 运行容器
docker run -d \
  --gpus all \
  -p 8001:8001 \
  -p 19530:19530 \
  -v $(pwd)/volumes:/app/volumes \
  easy-rag:latest
```

### 环境变量

```bash
export HF_ENDPOINT=https://hf-mirror.com  # HuggingFace 镜像
export CUDA_VISIBLE_DEVICES=0              # GPU 设备选择
export DOCKER_VOLUME_DIRECTORY=./volumes   # Docker 数据卷路径
```

## 注意事项

1. **GPU 内存**: 运行量化模型需要至少 16GB GPU 内存
2. **语料库更新**: 上传/删除文件后会自动触发知识库重建
3. **Token 限制**: 价格查询会检查数据量，超过 1000 条会提示用户细化条件
4. **上下文记忆**: 保留最近 5 轮对话历史 (最多 10 条消息)
5. **服务依赖**: 启动前确保 Milvus 和 LLM 服务已就绪

## 故障排查

### 常见问题

1. **Milvus 连接失败**
   ```bash
   # 检查服务状态
   docker-compose ps
   # 重启服务
   docker-compose restart standalone
   ```

2. **LLM 服务无响应**
   ```bash
   # 检查端口占用
   netstat -tulnp | grep 9844
   # 查看日志
   cat logs/vllm_quantized.log
   ```

3. **语料库哈希不匹配**
   - 系统会自动检测并重建向量库
   - 手动重建: 删除 `tokenized_docs.pkl` 和 Milvus 集合

## 相关文档

- `QUICK_START_GUIDE.md` - 快速使用指南
- `MODEL_PERFORMANCE_REPORT.md` - 模型性能对比报告
- `ALTERNATIVE_SOLUTIONS.md` - 备选部署方案
