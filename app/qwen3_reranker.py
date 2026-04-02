#!/usr/bin/env python3
"""
Qwen3-Reranker-4B 封装类
支持指令感知的重排序

模型规格:
- 参数量: 4B
- 架构: Cross-Encoder (交叉编码器)
- 最大长度: 32K tokens
- 语言支持: 100+ 种
- 指令感知: 支持
"""
import torch
from typing import List, Dict, Optional
from modelscope import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)


class Qwen3Reranker:
    """
    Qwen3-Reranker-4B 封装类
    
    特点:
    1. 交叉编码器架构，比双编码器更精准
    2. 支持 32K 长文本对比
    3. 支持指令感知 (instruction-aware)
    4. 在 MTEB-R/CMTEB-R 榜单上表现优异
    """
    
    # 默认任务指令
    DEFAULT_INSTRUCTION = (
        "Given a query and a passage, determine whether the passage "
        "contains an answer to the query."
    )
    
    # 领域特定指令模板
    INSTRUCTION_TEMPLATES = {
        "default": "Given a query and a passage, determine whether the passage contains an answer to the query.",
        "retrieval": "Given a web search query and a web page, determine whether the web page contains relevant information to answer the query.",
        "qa": "Given a question and a context, determine whether the context contains the answer to the question.",
        "code": "Given a code query and a code snippet, determine whether the code snippet is relevant to the query.",
        "construction": "Given a construction material query and a document, determine whether the document contains relevant price or specification information.",
    }
    
    def __init__(
        self, 
        model_path: str, 
        device: str = 'cuda',
        use_fp16: bool = True,
        max_length: int = 32768,
        instruction: str = None,
        instruction_template: str = None,
        trust_remote_code: bool = True
    ):
        """
        初始化 Qwen3 Reranker 模型
        
        Args:
            model_path: 模型路径或 HuggingFace/modelscope 模型名
            device: 计算设备 ('cuda' 或 'cpu')
            use_fp16: 是否使用 FP16 半精度
            max_length: 最大序列长度 (默认 32K)
            instruction: 自定义任务指令
            instruction_template: 预设指令模板名称 ("default", "retrieval", "qa", "code", "construction")
            trust_remote_code: 是否信任远程代码
        """
        self.device = device
        self.max_length = max_length
        self.model_path = model_path
        
        # 设置指令
        if instruction:
            self.instruction = instruction
        elif instruction_template and instruction_template in self.INSTRUCTION_TEMPLATES:
            self.instruction = self.INSTRUCTION_TEMPLATES[instruction_template]
        else:
            self.instruction = self.DEFAULT_INSTRUCTION
        
        logger.info(f"Loading Qwen3-Reranker model from {model_path}")
        logger.info(f"Configuration: device={device}, max_length={max_length}, fp16={use_fp16}")
        logger.info(f"Instruction: {self.instruction[:50]}...")
        
        try:
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
            # 修复 padding token 问题
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
            else:
                logger.info(f"Using existing pad_token: {self.tokenizer.pad_token}")
            # 加载模型 (回归任务，输出相关性分数)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                num_labels=1  # 回归任务
            )
            
            # 为模型设置 pad_token_id（解决批处理错误）
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                logger.info(f"Set model.config.pad_token_id to: {self.model.config.pad_token_id}")
            
            if use_fp16 and device == 'cuda' and torch.cuda.is_available():
                logger.info("Using FP16 half precision")
                self.model = self.model.half()
            
            self.model = self.model.to(device).eval()
            
            logger.info("Qwen3-Reranker model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def rerank(
        self, 
        docs: List[Dict], 
        query: str, 
        k: int = 10,
        batch_size: int = 8
    ) -> List[Dict]:
        """
        对文档进行重排序
        
        Args:
            docs: 文档列表，每个文档是包含 'page_content' 的字典
            query: 查询字符串
            k: 返回前 k 个结果
            batch_size: 批处理大小 (根据显存调整)
            
        Returns:
            按相关性排序后的文档列表
        """
        if not docs:
            logger.warning("Empty document list provided")
            return []
        
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return docs[:k]
        
        texts = [item.get("page_content", "") for item in docs]
        
        # 构造指令感知的查询
        query_with_instruction = f"{self.instruction}\nQuery: {query}"
        
        all_scores = []
        
        # 批处理打分
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            # 构造 [查询, 文档] 对
            pairs = [[query_with_instruction, text] for text in batch_texts]
            
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length
                    ).to(self.device)
                    
                    outputs = self.model(**inputs, return_dict=True)
                    
                    # 获取相关性分数 (回归输出)
                    batch_scores = outputs.logits.view(-1).float().cpu().tolist()
                    all_scores.extend(batch_scores)
                    
                    logger.debug(f"Batch {batch_num}/{total_batches} scored")
                    
            except Exception as e:
                logger.error(f"Error scoring batch {batch_num}/{total_batches}: {e}")
                # 返回中性分数作为 fallback
                all_scores.extend([0.0] * len(batch_texts))
        
        # 打包并排序 (分数高的在前)
        scored_docs = [(docs[i], all_scores[i]) for i in range(len(docs))]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 添加分数到 metadata
        result = []
        for doc, score in scored_docs[:k]:
            doc_copy = doc.copy()
            if "metadata" not in doc_copy:
                doc_copy["metadata"] = {}
            doc_copy["metadata"]["rerank_score"] = float(score)
            result.append(doc_copy)
        
        logger.info(f"Reranked {len(docs)} docs, returning top {k}. Best score: {scored_docs[0][1]:.4f}")
        
        return result
    
    def compute_score(
        self, 
        query: str, 
        passages: List[str],
        batch_size: int = 8
    ) -> List[float]:
        """
        计算查询与多个文档的相关性分数
        
        Args:
            query: 查询字符串
            passages: 文档内容列表
            batch_size: 批处理大小
            
        Returns:
            相关性分数列表
        """
        if not passages:
            return []
        
        query_with_instruction = f"{self.instruction}\nQuery: {query}"
        all_scores = []
        
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            pairs = [[query_with_instruction, p] for p in batch]
            
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length
                    ).to(self.device)
                    
                    outputs = self.model(**inputs, return_dict=True)
                    batch_scores = outputs.logits.view(-1).float().cpu().tolist()
                    all_scores.extend(batch_scores)
                    
            except Exception as e:
                logger.error(f"Error computing scores: {e}")
                all_scores.extend([0.0] * len(batch))
        
        return all_scores
    
    def set_instruction(self, instruction: str):
        """
        动态设置任务指令
        
        Args:
            instruction: 新的指令文本
        """
        self.instruction = instruction
        logger.info(f"Instruction updated: {instruction[:50]}...")
    
    def set_instruction_template(self, template_name: str):
        """
        使用预设的指令模板
        
        Args:
            template_name: 模板名称 ("default", "retrieval", "qa", "code", "construction")
        """
        if template_name in self.INSTRUCTION_TEMPLATES:
            self.instruction = self.INSTRUCTION_TEMPLATES[template_name]
            logger.info(f"Using instruction template: {template_name}")
        else:
            logger.warning(f"Unknown template: {template_name}, using default")
            self.instruction = self.DEFAULT_INSTRUCTION


class Qwen3RerankerVLLM:
    """
    使用 vLLM 部署的 Qwen3-Reranker (如果 vLLM 支持)
    当前主要为占位，等待 vLLM 官方支持
    """
    
    def __init__(self, api_base: str = "http://localhost:8000/v1"):
        """
        通过 API 调用 vLLM 部署的 Reranker
        
        Args:
            api_base: vLLM API 地址
        """
        self.api_base = api_base
        logger.info(f"Qwen3RerankerVLLM initialized (placeholder). API: {api_base}")
        logger.warning("vLLM support for Qwen3-Reranker is not yet available, use Qwen3Reranker instead")
    
    def rerank(self, docs: List[Dict], query: str, k: int = 10) -> List[Dict]:
        raise NotImplementedError("vLLM support for Qwen3-Reranker is not yet available")


# 用于兼容旧版 Reranker 接口的包装器
def create_reranker(
    model_type: str = "qwen3",
    model_path: str = None,
    device: str = "cuda",
    **kwargs
):
    """
    工厂函数：创建 Reranker 实例
    
    Args:
        model_type: "qwen3" 或 "bge"
        model_path: 模型路径
        device: 计算设备
        **kwargs: 额外参数
        
    Returns:
        Reranker 实例
    """
    if model_type == "qwen3":
        return Qwen3Reranker(model_path, device=device, **kwargs)
    else:
        # 导入原有的 Reranker
        from app.reranker import Reranker
        return Reranker(model_path, device=device)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    model_path = "/home/tpc/suda/rag/model/qwen3-reranker-4b"
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    
    try:
        logger.info("Testing Qwen3Reranker...")
        
        # 初始化模型
        reranker = Qwen3Reranker(
            model_path=model_path,
            device=device,
            instruction_template="construction"  # 使用工程造价领域指令
        )
        
        # 测试文档
        test_docs = [
            {"page_content": "钢筋 HRB400 直径 20mm，价格 4500元/吨，产地上海"},
            {"page_content": "混凝土 C30 配合比说明及施工注意事项"},
            {"page_content": "螺纹钢 HRB400E 直径 25mm，价格 4600元/吨"},
            {"page_content": "BIM 技术在建筑工程中的应用"},
        ]
        
        query = "钢筋价格"
        
        # 重排序
        ranked_docs = reranker.rerank(test_docs, query, k=3)
        
        logger.info("\nReranking results:")
        for i, doc in enumerate(ranked_docs, 1):
            score = doc["metadata"].get("rerank_score", 0)
            content = doc["page_content"][:50]
            logger.info(f"{i}. Score: {score:.4f} | {content}...")
        
        # 测试分数计算
        logger.info("\nTesting compute_score...")
        passages = [doc["page_content"] for doc in test_docs]
        scores = reranker.compute_score(query, passages)
        
        logger.info("Scores:")
        for p, s in zip(passages, scores):
            logger.info(f"  {s:.4f} | {p[:40]}...")
        
        logger.info("\nAll tests passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
