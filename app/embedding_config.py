#!/usr/bin/env python3
"""
Embedding 模型配置中心
统一管理 Embedding 和 Reranker 模型配置
"""

# ==================== Embedding 模型配置 ====================

# 当前使用的 Embedding 模型
# 可选值:
#   - "/home/tpc/suda/rag/model/bge-large-zh-v1.5"      # 当前使用 (326M, ~2GB)
#   - "/home/tpc/suda/rag/model/bge-m3"                  # BGE-M3 (2.7B, ~8GB)
#   - "/home/tpc/suda/rag/model/gte-large-zh"            # GTE-Large (1.3B, ~4GB)
#   - "/home/tpc/suda/rag/model/qwen3-embedding-0.6b"    # Qwen3-0.6B (~1.5GB, 推荐)
#   - "/home/tpc/suda/rag/model/qwen3-embedding-4b"      # Qwen3-4B (~8GB, 需释放显存)
#   - "/home/tpc/suda/rag/model/xiaobu-embedding"        # Xiaobu (7B, ~16GB)
EMBEDDING_MODEL_PATH = "/home/tpc/suda/rag/easy-rag/models/Qwen3-Embedding-0.6B"

# Embedding 模型类型
# 可选值: "sentence_transformer", "modelscope", "qwen3"
# 说明:
#   - sentence_transformer: 推荐使用，兼容性好
#   - modelscope: 使用 ModelScope 加载，适合国内模型
#   - qwen3: Qwen3 Embedding 专用加载器
EMBEDDING_TYPE = "sentence_transformer"

# 向量维度配置 (用于 Milvus 建表)
# 不同模型的维度:
#   - bge-large-zh-v1.5: 1024
#   - bge-m3: 1024
#   - gte-large-zh: 1024
#   - qwen3-embedding-0.6b: 1024
#   - qwen3-embedding-4b: 2560 (支持 MRL 降维)
#   - xiaobu-embedding: 4096
EMBEDDING_DIMENSION = 1024

# Qwen3 特有配置
QWEN3_EMBEDDING_DIM = 2560  # 完整维度，使用 MRL 可降维

# ==================== Reranker 模型配置 ====================

# Reranker 模型路径
# 可选值:
#   - "/home/tpc/suda/rag/model/bge-reranker-large"      # 当前使用 (326M, ~2GB)
#   - "/home/tpc/suda/rag/model/qwen3-reranker-0.6b"      # Qwen3-0.6B (~1.5GB, 推荐)
#   - "/home/tpc/suda/rag/model/qwen3-reranker-4b"        # Qwen3-4B (~8GB, 需释放显存)
RERANKER_MODEL_PATH = "/home/tpc/suda/rag/easy-rag/models/Qwen3-Reranker-0.6B"

# 是否使用 Reranker
USE_RERANKER = True

# ==================== 设备配置 ====================

# 指定 Embedding 模型使用的设备
# 可选值: "cuda", "cuda:0", "cuda:1", "cpu"
# 说明:
#   - 如果 GPU 0 显存紧张，可以指定 "cuda:1" 使用第二张卡
#   - "cuda" 自动选择，"cpu" 使用 CPU 推理（慢但省显存）
EMBEDDING_DEVICE = "cuda:1"  # 默认使用 GPU 1，通常空闲更多

# 指定 Reranker 模型使用的设备
# 由于 GPU 1 已被本地 LLM 服务占满，Reranker 改到 CPU 运行
# Qwen3-Reranker-0.6B 模型很小（~1.5GB），CPU 推理 20 条文档仅需几百毫秒
RERANKER_DEVICE = "cpu"

# 指定 LLM 使用的设备 (如果需要)
LLM_DEVICE = "cuda:0"

# ==================== 模型特性配置 ====================

# BGE-M3 特有配置 (仅在使用 bge-m3 时生效)
BGE_M3_CONFIG = {
    # 检索模式: "dense", "sparse", "multi_vector", "hybrid"
    "retrieval_mode": "hybrid",
    
    # 是否使用 fp16 加速
    "use_fp16": True,
    
    # 批量编码大小
    "batch_size": 32,
}

