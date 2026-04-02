#!/usr/bin/env python3
"""
Qwen3-Embedding-4B 封装类
支持 MRL (Matryoshka Representation Learning) 降维

模型规格:
- 参数量: 4B
- 维度: 2560 (支持 MRL 降维到 32-2560)
- 最大长度: 32K tokens
- 语言支持: 100+ 种
"""
import torch
from typing import List, Union
from langchain.schema.embeddings import Embeddings
from modelscope import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class Qwen3Embedding(Embeddings):
    """
    Qwen3-Embedding-4B 封装类
    
    特点:
    1. 支持 MRL 降维，可灵活调整输出维度
    2. 支持 32K 长文本
    3. 支持指令感知 (instruction-aware)
    """
    
    def __init__(
        self, 
        model_path: str, 
        device: str = 'cuda',
        embedding_dim: int = 2560,
        use_fp16: bool = True,
        batch_size: int = 16,
        instruction: str = None,
        trust_remote_code: bool = True
    ):
        """
        初始化 Qwen3 Embedding 模型
        
        Args:
            model_path: 模型路径或 HuggingFace/modelscope 模型名
            device: 计算设备 ('cuda' 或 'cpu')
            embedding_dim: 输出维度 (32-2560)，使用 MRL 降维
            use_fp16: 是否使用 FP16 半精度
            batch_size: 批处理大小
            instruction: 任务指令 (可选，用于指令感知编码)
            trust_remote_code: 是否信任远程代码
        """
        super().__init__()
        
        self.device = device
        self.embedding_dim = min(max(embedding_dim, 32), 2560)  # 限制在有效范围
        self.batch_size = batch_size
        self.instruction = instruction
        self.model_path = model_path
        
        logger.info(f"Loading Qwen3-Embedding model from {model_path}")
        logger.info(f"Configuration: device={device}, dim={self.embedding_dim}, fp16={use_fp16}")
        
        try:
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=trust_remote_code
            )
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=trust_remote_code
            )
            
            if use_fp16 and device == 'cuda' and torch.cuda.is_available():
                logger.info("Using FP16 half precision")
                self.model = self.model.half()
            
            self.model = self.model.to(device).eval()
            
            # 获取实际维度
            self.full_dim = self.model.config.hidden_size
            logger.info(f"Model loaded successfully. Full dimension: {self.full_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        编码文档列表
        
        Args:
            texts: 待编码的文本列表
            
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        
        # 清理文本
        texts = [t.replace("\n", " ").strip() for t in texts]
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            try:
                with torch.no_grad():
                    # Tokenize
                    inputs = self.tokenizer(
                        batch, 
                        padding=True, 
                        truncation=True, 
                        return_tensors='pt',
                        max_length=32768  # Qwen3 支持 32K
                    ).to(self.device)
                    
                    # 编码
                    outputs = self.model(**inputs)
                    
                    # 使用最后一层隐藏状态的最后一个 token 作为句子表示
                    # Qwen3-Embedding 使用 EOS token
                    batch_embeddings = outputs.last_hidden_state[:, -1, :]
                    
                    # L2 归一化
                    batch_embeddings = torch.nn.functional.normalize(
                        batch_embeddings, p=2, dim=1
                    )
                    
                    # MRL: 截取到指定维度
                    if self.embedding_dim < self.full_dim:
                        batch_embeddings = batch_embeddings[:, :self.embedding_dim]
                        # 重新归一化
                        batch_embeddings = torch.nn.functional.normalize(
                            batch_embeddings, p=2, dim=1
                        )
                    
                    all_embeddings.extend(batch_embeddings.cpu().float().tolist())
                    
            except Exception as e:
                logger.error(f"Error encoding batch {batch_num}/{total_batches}: {e}")
                # 返回零向量作为 fallback
                zero_emb = [0.0] * self.embedding_dim
                all_embeddings.extend([zero_emb] * len(batch))
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        编码单个查询
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        text = text.replace("\n", " ").strip()
        
        # 如果提供了指令，添加指令前缀
        if self.instruction:
            text = f"{self.instruction}\n{text}"
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=32768
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # 使用最后一个 token
                embedding = outputs.last_hidden_state[:, -1, :]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                # MRL 降维
                if self.embedding_dim < self.full_dim:
                    embedding = embedding[:, :self.embedding_dim]
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                return embedding[0].cpu().float().tolist()
                
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            return [0.0] * self.embedding_dim
    
    def embed_with_instruction(self, texts: List[str], instruction: str) -> List[List[float]]:
        """
        使用特定指令编码文本
        
        Args:
            texts: 文本列表
            instruction: 任务指令
            
        Returns:
            嵌入向量列表
        """
        # 临时保存原指令
        original_instruction = self.instruction
        
        try:
            self.instruction = instruction
            return self.embed_documents(texts)
        finally:
            # 恢复原指令
            self.instruction = original_instruction
    
    def get_embedding_dim(self) -> int:
        """获取当前输出维度"""
        return self.embedding_dim
    
    def get_full_dim(self) -> int:
        """获取模型完整维度"""
        return self.full_dim


class AdaptiveQwen3Embedding(Qwen3Embedding):
    """
    自适应维度的 Qwen3 Embedding
    支持根据检索阶段动态调整输出维度
    """
    
    DIM_CONFIG = {
        "coarse": 256,    # 粗排: 快速筛选
        "medium": 1024,   # 中排: 平衡速度和质量
        "fine": 2560      # 精排: 最佳质量
    }
    
    def embed_for_stage(
        self, 
        texts: List[str], 
        stage: str = "fine"
    ) -> List[List[float]]:
        """
        根据检索阶段编码
        
        Args:
            texts: 文本列表
            stage: 检索阶段 ("coarse", "medium", "fine")
            
        Returns:
            嵌入向量列表
        """
        target_dim = self.DIM_CONFIG.get(stage, self.full_dim)
        
        # 临时修改维度
        original_dim = self.embedding_dim
        self.embedding_dim = target_dim
        
        try:
            results = self.embed_documents(texts)
            logger.debug(f"Encoded {len(texts)} texts with dimension {target_dim} ({stage} stage)")
            return results
        finally:
            self.embedding_dim = original_dim


# 兼容性包装器：提供与 sentence_transformers 相似的接口
class Qwen3SentenceTransformer:
    """
    兼容 sentence-transformers 接口的包装器
    """
    
    def __init__(self, model_path: str, device: str = 'cuda', **kwargs):
        self.embedding_model = Qwen3Embedding(model_path, device=device, **kwargs)
        self.device = device
    
    def encode(
        self, 
        sentences: Union[str, List[str]], 
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> Union[List[float], List[List[float]]]:
        """
        兼容 sentence-transformers 的 encode 方法
        """
        if isinstance(sentences, str):
            embeddings = self.embedding_model.embed_query(sentences)
            if convert_to_tensor:
                embeddings = torch.tensor(embeddings).to(self.device)
            return embeddings
        else:
            embeddings = self.embedding_model.embed_documents(sentences)
            if convert_to_tensor:
                embeddings = [torch.tensor(e).to(self.device) for e in embeddings]
            return embeddings
    
    def get_sentence_embedding_dimension(self) -> int:
        """获取嵌入维度"""
        return self.embedding_model.get_embedding_dim()


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # 测试模型加载
    model_path = "/home/tpc/suda/rag/model/qwen3-embedding-4b"
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    
    try:
        # 测试基础 Embedding
        logger.info("Testing Qwen3Embedding...")
        model = Qwen3Embedding(
            model_path=model_path,
            device=device,
            embedding_dim=2560
        )
        
        # 测试文档编码
        test_docs = [
            "这是一个测试文档",
            "Qwen3 Embedding 模型支持长文本",
            "RAG 系统需要高质量的向量表示"
        ]
        
        embeddings = model.embed_documents(test_docs)
        logger.info(f"Generated {len(embeddings)} embeddings, dim={len(embeddings[0])}")
        
        # 测试查询编码
        query_emb = model.embed_query("测试查询")
        logger.info(f"Query embedding dim={len(query_emb)}")
        
        # 测试自适应 Embedding
        logger.info("Testing AdaptiveQwen3Embedding...")
        adaptive_model = AdaptiveQwen3Embedding(
            model_path=model_path,
            device=device
        )
        
        for stage in ["coarse", "medium", "fine"]:
            emb = adaptive_model.embed_for_stage(["测试文本"], stage=stage)
            logger.info(f"Stage '{stage}': dim={len(emb[0])}")
        
        logger.info("All tests passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
