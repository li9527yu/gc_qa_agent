# Qwen3-Embedding-4B / Qwen3-Reranker-4B 使用指南

## 📊 模型对比分析

### 规格对比

| 特性 | BGE-large-zh-v1.5 | **Qwen3-Embedding-4B** | 提升 |
|------|-------------------|------------------------|------|
| 参数量 | 326M | **4B** (12x) | ✅ 12倍 |
| 维度 | 1024 | **2560** | ✅ 2.5倍 |
| 最大长度 | 512 | **32K** (64x) | ✅ 64倍 |
| 语言支持 | 中文 | **100+种** | ✅ 多语言 |
| MRL支持 | ❌ | ✅ | ✅ 灵活降维 |
| 指令感知 | ❌ | ✅ | ✅ 任务定制 |
| MTEB排名 | 第5名 | **超越Gemini** | ✅ SOTA |

| 特性 | bge-reranker-large | **Qwen3-Reranker-4B** | 提升 |
|------|--------------------|------------------------|------|
| 参数量 | 326M | **4B** (12x) | ✅ 12倍 |
| 最大长度 | 512 | **32K** (64x) | ✅ 64倍 |
| 指令感知 | ❌ | ✅ | ✅ 任务定制 |
| CMTEB-R | baseline | **77+分** | ✅ 领先 |

---

## ✅ 兼容性分析

### 优势

1. **架构兼容**
   - Qwen3-Embedding: 双编码器 (Bi-Encoder) - 与当前 BGE 架构一致
   - Qwen3-Reranker: 交叉编码器 (Cross-Encoder) - 与当前 Reranker 架构一致

2. **显存充足**
   - Embedding-4B: ~8GB (FP16)
   - Reranker-4B: ~8GB (FP16)
   - 你的 96GB 显存完全足够 ✅

3. **序列长度优势**
   - 32K 上下文 vs 当前 512
   - 可以处理更长文档，无需分块

### 需要注意的问题

1. **维度不同**
   - 当前: 1024 维
   - Qwen3: 2560 维
   - **必须**重建 Milvus 向量库

2. **加载方式**
   - 目前主要通过 `transformers` 或 `modelscope` 加载
   - `sentence-transformers` 支持正在完善中
   - 需要修改 `TextEmbedding` 类

3. **MRL 特性**
   - 支持输出维度自定义 (32-2560)
   - 可以通过 MRL 降维到 1024 以节省存储

---

## 🚀 使用方案

### 方案 1: 直接使用 2560 维 (推荐)

**优点**: 完整保留模型能力  
**缺点**: 向量存储增加 2.5 倍

```python
# Milvus 配置
EMBEDDING_DIMENSION = 2560
```

### 方案 2: 使用 MRL 降维到 1024 维

**优点**: 兼容现有存储，节省空间  
**缺点**: 略微损失精度 (~2%)

```python
# 在模型输出时截断到 1024 维
embedding = model.encode(text)[:1024]
```

### 方案 3: 使用 MRL 降维到 768 维

**优点**: 存储更节省  
**缺点**: 精度损失约 5%

---

## 📦 实施步骤

### 步骤 1: 下载模型

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 创建模型目录
mkdir -p /home/tpc/suda/rag/model/qwen3-embedding-4b
mkdir -p /home/tpc/suda/rag/model/qwen3-reranker-4b

# 下载 Embedding 模型
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-Embedding-4B', cache_dir='/home/tpc/suda/rag/model/qwen3-embedding-4b')
"

# 下载 Reranker 模型
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-Reranker-4B', cache_dir='/home/tpc/suda/rag/model/qwen3-reranker-4b')
"
```

### 步骤 2: 创建 Qwen3 Embedding 封装类

创建 `app/qwen3_embedding.py`:

```python
import torch
from typing import List
from langchain.schema.embeddings import Embeddings
from modelscope import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class Qwen3Embedding(Embeddings):
    """
    Qwen3-Embedding-4B 封装类
    支持 MRL (Matryoshka Representation Learning) 降维
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str = 'cuda',
        embedding_dim: int = 2560,  # 默认使用完整维度
        use_fp16: bool = True,
        batch_size: int = 32
    ):
        """
        Args:
            model_path: 模型路径
            device: 计算设备
            embedding_dim: 输出维度 (32-2560, 支持 MRL)
            use_fp16: 是否使用 FP16
            batch_size: 批处理大小
        """
        self.device = device
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        logger.info(f"Loading Qwen3-Embedding model from {model_path}")
        logger.info(f"Output dimension: {embedding_dim}")
        
        # 加载 tokenizer 和 model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        
        if use_fp16 and device == 'cuda':
            self.model = self.model.half()
            
        self.model = self.model.to(device).eval()
        
        logger.info("Qwen3-Embedding model loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """编码文档列表"""
        texts = [t.replace("\n", " ") for t in texts]
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt',
                    max_length=32768  # Qwen3 支持 32K
                ).to(self.device)
                
                # 获取最后一层隐藏状态
                outputs = self.model(**inputs)
                
                # 使用 [EOS] token 的 embedding 作为句子表示
                # Qwen3-Embedding 使用最后 token
                batch_embeddings = outputs.last_hidden_state[:, -1, :]
                
                # 归一化
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                # MRL: 截取到指定维度
                if self.embedding_dim < batch_embeddings.shape[-1]:
                    batch_embeddings = batch_embeddings[:, :self.embedding_dim]
                    # 重新归一化
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                all_embeddings.extend(batch_embeddings.cpu().tolist())
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """编码单个查询"""
        text = text.replace("\n", " ")
        
        with torch.no_grad():
            inputs = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=32768
            ).to(self.device)
            
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, -1, :]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            # MRL 降维
            if self.embedding_dim < embedding.shape[-1]:
                embedding = embedding[:, :self.embedding_dim]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            return embedding[0].cpu().tolist()
```

### 步骤 3: 创建 Qwen3 Reranker 封装类

创建 `app/qwen3_reranker.py`:

```python
import torch
from typing import List, Dict
from modelscope import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class Qwen3Reranker:
    """
    Qwen3-Reranker-4B 封装类
    支持指令感知的重排序
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str = 'cuda',
        use_fp16: bool = True,
        max_length: int = 32768,
        instruction: str = None
    ):
        """
        Args:
            model_path: 模型路径
            device: 计算设备
            use_fp16: 是否使用 FP16
            max_length: 最大序列长度
            instruction: 任务指令 (可选，用于指令感知)
        """
        self.device = device
        self.max_length = max_length
        self.instruction = instruction or "Given a query and a passage, determine whether the passage contains an answer to the query by responding with 'Yes' or 'No'."
        
        logger.info(f"Loading Qwen3-Reranker model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            trust_remote_code=True,
            num_labels=1  # 回归任务
        )
        
        if use_fp16 and device == 'cuda':
            self.model = self.model.half()
            
        self.model = self.model.to(device).eval()
        
        logger.info("Qwen3-Reranker model loaded successfully")
    
    def rerank(self, docs: List[Dict], query: str, k: int) -> List[Dict]:
        """
        对文档进行重排序
        
        Args:
            docs: 文档列表，每个文档包含 page_content
            query: 查询字符串
            k: 返回前 k 个结果
            
        Returns:
            排序后的文档列表
        """
        if not docs:
            return []
        
        texts = [item["page_content"] for item in docs]
        
        # 构造指令感知的输入
        # Qwen3-Reranker 支持指令前缀
        query_with_instruction = f"{self.instruction}\nQuery: {query}"
        
        pairs = [[query_with_instruction, text] for text in texts]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length
            ).to(self.device)
            
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
        
        # 打包并排序
        scored_docs = [(docs[i], scores[i]) for i in range(len(docs))]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in scored_docs[:k]]
```

### 步骤 4: 修改配置

更新 `app/embedding_config.py`:

```python
# Qwen3 Embedding 配置
USE_QWEN3_EMBEDDING = True
QWEN3_EMBEDDING_PATH = "/home/tpc/suda/rag/model/qwen3-embedding-4b"
QWEN3_EMBEDDING_DIM = 2560  # 或 1024 (MRL降维)

# Qwen3 Reranker 配置
USE_QWEN3_RERANKER = True
QWEN3_RERANKER_PATH = "/home/tpc/suda/rag/model/qwen3-reranker-4b"
```

### 步骤 5: 修改 Retriever

更新 `app/retriever.py` 中的模型初始化逻辑:

```python
# 在初始化时根据配置选择模型
if embedding_type == 'qwen3':
    from app.qwen3_embedding import Qwen3Embedding
    self.emb_model = Qwen3Embedding(
        model_path=emb_model_name_or_path,
        device=device,
        embedding_dim=QWEN3_EMBEDDING_DIM
    )
elif embedding_type == 'sentence_transformer':
    self.emb_model = SentenceTransformerEmbedding(...)
```

### 步骤 6: 清理并重建向量库

```bash
# 停止服务
pkill -f "python -m app.api_server"

# 清理数据
rm -f tokenized_docs.pkl
rm -f milvus_rag.db
rm -f ~/.easy_rag_cache/easy_rag_milvus_hash.txt

# 启动服务 (自动重建)
python -m app.api_server
```

---

## 📈 性能预期

### 检索质量提升

| 指标 | 当前 (BGE) | Qwen3-4B | 提升 |
|------|-----------|----------|------|
| 语义理解 | baseline | +15-20% | ✅ 显著 |
| 长文档处理 | 512 tokens | 32K tokens | ✅ 64x |
| 跨语言检索 | 有限 | 100+语言 | ✅ 多语言 |
| 代码检索 | 一般 | 优秀 | ✅ 专业 |

### 速度影响

| 操作 | BGE-large | Qwen3-4B | 比例 |
|------|-----------|----------|------|
| 文档编码 | 1000 docs/s | ~300 docs/s | ~30% |
| 查询编码 | 10ms | ~30ms | ~30% |
| 重排序 | 50 pairs/s | ~20 pairs/s | ~40% |

**说明**: 速度降低但质量显著提升，对于 RAG 场景是可接受的 trade-off。

---

## 🔧 进阶优化

### 1. 使用 MRL 动态降维

```python
# 根据不同的检索阶段使用不同维度
# - 粗排: 256 维 (快速筛选)
# - 精排: 1024 维 (质量排序)
# - 最终: 2560 维 (精确匹配)

class AdaptiveQwen3Embedding(Qwen3Embedding):
    def embed_for_stage(self, texts: List[str], stage: str = "fine") -> List[List[float]]:
        dim_map = {"coarse": 256, "medium": 1024, "fine": 2560}
        self.embedding_dim = dim_map.get(stage, 2560)
        return self.embed_documents(texts)
```

### 2. 缓存优化

```python
# 使用 LRU 缓存查询 embedding
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed_query(self, text: str) -> tuple:
    return tuple(self.embed_query(text))
```

### 3. 批处理优化

```python
# 动态调整 batch_size 基于序列长度
# 短文本用大的 batch_size，长文本用小的

def adaptive_batch_size(texts: List[str]) -> int:
    avg_len = sum(len(t) for t in texts) / len(texts)
    if avg_len < 100:
        return 64
    elif avg_len < 1000:
        return 32
    else:
        return 8
```

---

## ❓ 常见问题

### Q1: 2560 维向量会占用多少存储？

**A**: 
- 原始 BGE (1024维): 100万文档 ≈ 4GB
- Qwen3 (2560维): 100万文档 ≈ 10GB
- 使用 MRL 1024维: 100万文档 ≈ 4GB (相同)

### Q2: 32K 长度真的能用吗？

**A**: 
- 可以，但需要注意：
  - 长文本编码更慢
  - 显存占用更高 (激活值)
  - 建议 >8K 的文档分块处理

### Q3: 与 vLLM 部署的 Qwen3-32B 有冲突吗？

**A**: 
- **无冲突**。Embedding/Reranker 与 LLM 是独立的
- 可以在同一 GPU 上共存
- 也可以分配到不同 GPU (推荐)

### Q4: 如何评估升级效果？

**A**: 
```bash
# 使用项目测试脚本
python test_embedding_models.py

# 人工评估检索质量
# 准备 50 个测试查询，对比前后检索结果的相关性
```

---

## 📚 参考资源

- [Qwen3-Embedding GitHub](https://github.com/QwenLM/Qwen3-Embedding)
- [HuggingFace Collection](https://huggingface.co/collections/Qwen/qwen3-embedding)
- [论文: Qwen3 Embedding](https://arxiv.org/abs/2506.05176)
- [ModelScope 镜像](https://www.modelscope.cn/organization/Qwen)

---

**建议**: 鉴于你的 96GB 显存充足，强烈建议升级到 Qwen3-Embedding-4B + Qwen3-Reranker-4B 组合，可以获得显著的检索质量提升！
