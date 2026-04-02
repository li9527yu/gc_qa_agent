import torch
from typing import List, Dict
from modelscope import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranker 包装类
    支持 BGE-Reranker 和 Qwen3-Reranker
    """
    
    def __init__(self, rerank_model_name_or_path, device='cuda'):
        self.device = device
        
        # 检测是否是 Qwen3 模型
        if 'qwen3' in rerank_model_name_or_path.lower():
            logger.info(f"Loading Qwen3 Reranker from {rerank_model_name_or_path}")
            from app.qwen3_reranker import Qwen3Reranker
            self._qwen3_reranker = Qwen3Reranker(
                model_path=rerank_model_name_or_path,
                device=device,
                use_fp16=True
            )
            self._use_qwen3 = True
            logger.info('Qwen3 Reranker loaded successfully')
        else:
            # 原有的 BGE Reranker 加载逻辑
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name_or_path)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name_or_path)\
                .half().to(device).eval()
            self._use_qwen3 = False
            logger.info('BGE Reranker loaded successfully')

    def rerank(self, docs: List[Dict], query: str, k: int) -> List[Dict]:
        """
        对文档进行重排序
        
        Args:
            docs: 文档列表
            query: 查询字符串
            k: 返回前 k 个
            
        Returns:
            排序后的文档列表
        """
        if self._use_qwen3:
            # 使用 Qwen3 Reranker
            return self._qwen3_reranker.rerank(docs, query, k)
        
        # 原有的 BGE Reranker 逻辑
        # 提取 page_content 用于排序
        texts = [item["page_content"] for item in docs]

        # 构造输入对
        pairs = [[query, text] for text in texts]

        # 模型打分
        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs, padding=True, truncation=True, return_tensors='pt', max_length=512
            ).to(self.device)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

        # 将文档和分数打包并排序
        scored_docs = [(docs[i], scores[i]) for i in range(len(docs))]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 返回排序后的原始对象（保留 metadata）
        return [item[0] for item in scored_docs[:k]]
