import time
import asyncio
import logging
import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, Any, Optional, Tuple, List
import json

from app.llm_deepseek import LLMPredictor
from app.retriever import Retriever
from app.incremental_retriever import IncrementalRetriever
from app.reranker import Reranker
from app.read_corpus import Reader
from app.config import RELATED_DATA_PATH
from app.config import DATA_PATH
from app.config import ROOT_PATH
from app.config import USE_MOCK_API, MOCK_API_BASE_URL, REAL_API_BASE_URL, API_SECRET_KEY, PRICE_API_ACCOUNT_ID
from app.embedding_config import (
    EMBEDDING_MODEL_PATH,
    EMBEDDING_TYPE,
    EMBEDDING_DIMENSION,
    RERANKER_MODEL_PATH,
    USE_RERANKER,
    EMBEDDING_DEVICE,
    RERANKER_DEVICE
)
from app.utils.generate_signature import generate_signature
from app.utils.prompt_template import Price_Channel_TEMPLATE, Entity_Extract_TEMPLATE, DIRECT_QUERY_EXTRACTION_TEMPLATE, better_template, Price_Answer_TEMPLATE
from app.utils.price_tools import remove_outliers, analyze_prices, analyze_by_unit
from app.utils.fillter_tools import filter_items
from app.utils.data_filter_helper import check_data_volume_and_guide, TokenEstimator
from app.utils.channel_inferencer import (
    ChannelInferencer, 
    get_channel_inferencer,
    ChannelInferenceResult,
    ChannelType as InferencerChannelType
)
from app.utils.dialogue_manager import (
    DialogueManager,
    get_dialogue_manager,
    DialogueSession
)
from app.utils.conversation_manager import (
    get_conversation_manager,
    ConversationManager,
    ConversationType
)
from app.utils.query_rewriter import QueryRewriter, create_query_rewriter, RewriteResult
from app.services.kb_manager import get_kb_manager
# 画图函数：区间频数分布图
# from app.tools import plot_price_distribution
from datetime import datetime
import os
from enum import Enum


class ChannelType(str, Enum):
    ZC_PRICE = "zc_price"
    MANUFACTURER_PRICE = "manufacturer_price"
    INFORMATION_PRICE = "information_price"
    UNKNOWN = "unknown"  # 无法识别渠道



class RAGService:
    # 初始化
    def __init__(self):
        self.logger = logging.getLogger("easy_rag_api")
        self.retriever = None
        self.reranker = None
        self.llm = None
        self.reader = None
        self.corpus = None
        self.semaphore = asyncio.Semaphore(10)
        self.analyze_by_unit = analyze_by_unit
        self.better_template = None
        self.price_channel_template = None  
        self.entity_extract_template = None
        self.intent_template = None
        self.fillter_items = filter_items
        self._last_price_metadata = {}
        # 新增：渠道推断器
        self.channel_inferencer = get_channel_inferencer()
        # 新增：对话管理器
        self.dialogue_manager = get_dialogue_manager()
        # 新增：统一会话管理器
        self.conversation_manager = get_conversation_manager()
        # 新增：知识库索引管理器
        self.kb_manager = get_kb_manager()
        # 新增：知识库重建状态管理
        self._is_rebuilding = False
        self._rebuild_lock = asyncio.Lock()
        self._rebuild_progress = {"stage": "idle", "message": "就绪", "percent": 100}
        # 新增：查询重写器
        self.query_rewriter = None

    def _safe_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def _summarize_price_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "materialName",
            "province",
            "city",
            "materialModelSpec",
            "brand",
            "startReleaseDate",
            "endReleaseDate",
        ]
        return {
            key: value for key, value in entities.items()
            if key in keys and value not in (None, "", [], {})
        }

    def _normalize_price_query_entities(self, entities: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """统一价格查询时间字段，兼容历史别名。"""
        normalized = dict(entities or {})

        alias_map = {
            "startReleaseTime": "startReleaseDate",
            "endReleaseTime": "endReleaseDate",
            "start_release_date": "startReleaseDate",
            "end_release_date": "endReleaseDate",
        }
        for old_key, new_key in alias_map.items():
            if normalized.get(old_key) and not normalized.get(new_key):
                normalized[new_key] = normalized[old_key]

        release_date = normalized.get("releaseDate")
        if release_date and not normalized.get("startReleaseDate"):
            normalized["startReleaseDate"] = release_date

        return normalized

    def _price_trace_log(self, trace_id: Optional[str], stage: str, **kwargs):
        prefix = f"[Price Trace][{trace_id or 'no-trace'}] {stage}"
        detail = ", ".join(
            f"{key}={self._safe_json(value)}"
            for key, value in kwargs.items()
            if value is not None
        )
        self.logger.info(f"{prefix} | {detail}" if detail else prefix)

    # 获取服务状态（用于前端检查知识库重建状态）
    def get_service_status(self) -> Dict[str, Any]:
        """获取当前服务状态，包括知识库重建状态"""
        return {
            "is_rebuilding": self._is_rebuilding,
            "rebuild_progress": self._rebuild_progress,
            "retriever_ready": self.retriever is not None,
            "llm_ready": self.llm is not None,
            "corpus_count": len(self.corpus) if self.corpus else 0
        }

    # 异步初始化（支持双缓冲，重建期间不影响现有服务）
    async def initialize(self):
        self.logger.info("开始初始化RAG服务...")
        async with self._rebuild_lock:
            self._is_rebuilding = True
            self._rebuild_progress = {"stage": "loading", "message": "正在加载语料库...", "percent": 10}
            
            try:
                # ========== 双缓冲机制：保留旧实例，先在新变量上构建 ==========
                
                # 1. 加载语料库（在新变量上操作，不影响现有服务）
                new_reader = Reader(RELATED_DATA_PATH)
                new_corpus = new_reader.corpus
                self.logger.info(f"语料库加载完成，共{len(new_corpus)}个文档")
                self._rebuild_progress = {"stage": "checking", "message": "正在检查语料库变化...", "percent": 20}

                # 2. 自动检测是否需要强制重建
                from app.retriever import compute_corpus_hash, load_corpus_hash
                current_hash = compute_corpus_hash(new_corpus)
                cached_hash = load_corpus_hash("easy_rag_milvus")
                
                force_rebuild = False
                if current_hash != cached_hash:
                    self.logger.info(f"检测到语料库变化，当前哈希: {current_hash}, 缓存哈希: {cached_hash}")
                    self.logger.info("将自动重建向量库以同步最新语料库")
                    force_rebuild = True
                else:
                    self.logger.info("语料库未发生变化，使用现有缓存")
                    # 语料库未变，只需更新 reader 和 corpus 即可
                    self.reader = new_reader
                    self.corpus = new_corpus
                    
                    # 初始化 LLM（如果是首次初始化）- 必须在 return 前完成
                    if self.llm is None:
                        self.llm = LLMPredictor(logger=self.logger)
                        self.logger.info("LLM初始化完成")

                    # 初始化查询重写器（依赖 LLM）- 必须在 return 前完成
                    if self.query_rewriter is None:
                        self.query_rewriter = create_query_rewriter(self.llm)
                        self.logger.info("查询重写器初始化完成")
                    
                    # 初始化模板（如果是首次初始化）- 必须在 return 前完成
                    if self.intent_template is None:
                        await self._initialize_price_templates()
                        self.logger.info("价格推荐模板初始化完成")
                    
                    # 关键：如果retriever未初始化，需要加载它（不重建，只加载）
                    if self.retriever is None:
                        self.logger.info("语料库未变化，但Retriever未初始化，正在加载...")
                        self._rebuild_progress = {"stage": "loading", "message": "正在加载检索器...", "percent": 30}
                        
                        # 加载嵌入模型（复用现有向量库）
                        if EMBEDDING_TYPE == 'modelscope':
                            from app.retriever import TextEmbedding
                            emb_model = TextEmbedding(
                                emb_model_name_or_path=EMBEDDING_MODEL_PATH,
                                device=EMBEDDING_DEVICE
                            )
                        elif EMBEDDING_TYPE == 'sentence_transformer':
                            from app.retriever import SentenceTransformerEmbedding
                            emb_model = SentenceTransformerEmbedding(
                                emb_model_name_or_path=EMBEDDING_MODEL_PATH,
                                device=EMBEDDING_DEVICE
                            )
                        else:
                            from app.qwen3_embedding import Qwen3Embedding
                            from app.embedding_config import EMBEDDING_DIMENSION, QWEN3_EMBEDDING_DIM
                            use_full_dim = EMBEDDING_DIMENSION >= 2048
                            emb_model = Qwen3Embedding(
                                model_path=EMBEDDING_MODEL_PATH,
                                device=EMBEDDING_DEVICE,
                                embedding_dim=QWEN3_EMBEDDING_DIM if use_full_dim else EMBEDDING_DIMENSION,
                                use_fp16=True,
                                batch_size=8 if '0.6b' in EMBEDDING_MODEL_PATH.lower() else 4
                            )
                        
                        # 连接Milvus（不重建）
                        from langchain_community.vectorstores import Milvus
                        if "lite" in str(self.__dict__.get('milvus_mode', 'standalone')):
                            connection_args = {"uri": ROOT_PATH+"/milvus_rag.db"}
                        else:
                            connection_args = {"host": "127.0.0.1", "port": "19530"}
                        
                        self.retriever = IncrementalRetriever(
                            emb_model_name_or_path=EMBEDDING_MODEL_PATH,
                            corpus=self.corpus,
                            tokenized_path="tokenized_docs.pkl",
                            device=EMBEDDING_DEVICE,
                            lan="zh",
                            collection_name="easy_rag_milvus",
                            milvus_mode="standalone",
                            embedding_type=EMBEDDING_TYPE,
                            force_rebuild=False  # 关键：不重建，只加载
                        )
                        self.logger.info("Retriever加载完成")
                    
                    # 关键：如果reranker未初始化且启用重排，需要加载它
                    if self.reranker is None and USE_RERANKER:
                        self.logger.info("语料库未变化，但Reranker未初始化，正在加载...")
                        self._rebuild_progress = {"stage": "loading", "message": "正在加载重排器...", "percent": 35}
                        self.reranker = Reranker(
                            rerank_model_name_or_path=RERANKER_MODEL_PATH,
                            device=RERANKER_DEVICE
                        )
                        self.logger.info("Reranker加载完成")

                    kb_sync_stats = self.kb_manager.sync_with_corpus(self.corpus)
                    self.logger.info(
                        f"知识库索引同步完成: old_files={kb_sync_stats['old_files']}, "
                        f"new_files={kb_sync_stats['new_files']}, "
                        f"skipped_files={len(kb_sync_stats['skipped_files'])}"
                    )
                    
                    self._is_rebuilding = False
                    self._rebuild_progress = {"stage": "idle", "message": "就绪", "percent": 100}
                    return
                
                # 检查显存是否充足
                if EMBEDDING_DEVICE.startswith("cuda"):
                    import torch
                    if torch.cuda.is_available():
                        gpu_id = int(EMBEDDING_DEVICE.split(":")[1]) if ":" in EMBEDDING_DEVICE else 0
                        free_mem = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
                        free_mem_gb = free_mem / 1024**3
                        self.logger.info(f"   GPU {gpu_id} 可用显存: {free_mem_gb:.2f} GB")

                # 3. 释放旧模型显存（避免双缓冲导致显存翻倍）
                if self.retriever is not None or self.reranker is not None:
                    self.logger.info("正在释放旧模型显存...")
                    self.retriever = None
                    self.reranker = None
                    import gc
                    gc.collect()
                    if EMBEDDING_DEVICE.startswith("cuda"):
                        import torch
                        torch.cuda.empty_cache()
                        self.logger.info("已释放 GPU 缓存")
                        # 再次检查显存
                        gpu_id = int(EMBEDDING_DEVICE.split(":")[1]) if ":" in EMBEDDING_DEVICE else 0
                        free_mem = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
                        free_mem_gb = free_mem / 1024**3
                        self.logger.info(f"   释放后 GPU {gpu_id} 可用显存: {free_mem_gb:.2f} GB")

                # 5. 构建新的 Retriever（使用增量检索器）
                self._rebuild_progress = {"stage": "building", "message": "正在构建向量库...", "percent": 40}
                new_retriever = IncrementalRetriever(
                    emb_model_name_or_path=EMBEDDING_MODEL_PATH,
                    corpus=new_corpus,
                    tokenized_path="tokenized_docs.pkl",
                    device=EMBEDDING_DEVICE,
                    lan="zh",
                    collection_name="easy_rag_milvus",
                    milvus_mode="standalone",
                    embedding_type=EMBEDDING_TYPE,
                    force_rebuild=force_rebuild
                )
                self.logger.info("检索器初始化完成")
                self._rebuild_progress = {"stage": "building", "message": "正在加载重排模型...", "percent": 70}

                # 6. 构建新的 Reranker（在新变量上）
                new_reranker = None
                if USE_RERANKER:
                    self.logger.info(f"📦 加载 Reranker 模型: {RERANKER_MODEL_PATH}")
                    self.logger.info(f"   设备: {RERANKER_DEVICE}")
                    new_reranker = Reranker(
                        rerank_model_name_or_path=RERANKER_MODEL_PATH,
                        device=RERANKER_DEVICE
                    )
                self.logger.info("重排器初始化完成")
                self._rebuild_progress = {"stage": "building", "message": "正在初始化LLM...", "percent": 85}

                # 7. 初始化 LLM（如果是首次初始化）
                if self.llm is None:
                    self.llm = LLMPredictor(logger=self.logger)
                    self.logger.info("LLM初始化完成")
                
                # 7.5 初始化查询重写器（依赖LLM）
                if self.query_rewriter is None:
                    self.query_rewriter = create_query_rewriter(self.llm)
                    self.logger.info("查询重写器初始化完成")

                # 8. 初始化模板（如果是首次初始化）
                if self.intent_template is None:
                    await self._initialize_price_templates()
                    self.logger.info("价格推荐模板初始化完成")

                # 9. 原子性替换：所有新组件就绪后才替换旧组件
                self._rebuild_progress = {"stage": "switching", "message": "正在切换知识库...", "percent": 95}
                self.reader = new_reader
                self.corpus = new_corpus
                self.retriever = new_retriever
                self.reranker = new_reranker
                kb_sync_stats = self.kb_manager.sync_with_corpus(self.corpus)
                self.logger.info(
                    f"知识库索引同步完成: old_files={kb_sync_stats['old_files']}, "
                    f"new_files={kb_sync_stats['new_files']}, "
                    f"skipped_files={len(kb_sync_stats['skipped_files'])}"
                )
                
                # 8. 延迟清理旧资源（给正在进行的请求一些时间完成）
                # 注意：这里不立即清理，因为 Python 的垃圾回收机制会在之后处理
                
                self._is_rebuilding = False
                self._rebuild_progress = {"stage": "idle", "message": "就绪", "percent": 100}
                self.logger.info("RAG服务初始化完成！（双缓冲切换成功）")
                
            except Exception as e:
                self._is_rebuilding = False
                self._rebuild_progress = {"stage": "error", "message": f"重建失败: {str(e)}", "percent": 0}
                self.logger.error(f"RAG服务初始化失败: {e}")
                raise
    
    # 初始化价格推荐相关的模板
    async def _initialize_price_templates(self):
        """初始化价格推荐相关的模板"""
        try:
            self.price_channel_template = Price_Channel_TEMPLATE
            self.entity_extract_template = Entity_Extract_TEMPLATE
            self.better_template = better_template
            self.logger.info("价格推荐模板加载成功")
        except ImportError as e:
            self.logger.warning(f"价格推荐模板加载失败: {e}")
            self.logger.warning("将使用默认模板或跳过价格推荐功能")

    
    # 渠道意图识别入口（增强版）
    async def identify_channel(self, question: str, parsed_entities: Optional[dict] = None) -> Tuple[ChannelType, ChannelInferenceResult]:
        """
        渠道意图识别（增强版）
        
        Returns:
            Tuple[ChannelType, ChannelInferenceResult]: 
                - 渠道类型（用于后续处理）
                - 推断详情（用于展示给用户）
        """
        try:
            if not self.price_channel_template:
                self.logger.warning("价格渠道模板未初始化")
                return ChannelType.UNKNOWN, ChannelInferenceResult(
                    channel=InferencerChannelType.UNKNOWN,
                    confidence=0.0,
                    reason="模板未初始化",
                    matched_keywords=[]
                )
            
            # 1. 先尝试 LLM 精确识别
            channel_str = await asyncio.to_thread(
                self.llm.predict_prompt, 
                self.price_channel_template,
                {"user_question": question},
            )
            
            self.logger.info(f"LLM 渠道识别原始结果: {channel_str}")
            
            # 2. 解析 LLM 结果
            llm_channel = self._parse_channel_string(channel_str.strip().lower() if channel_str else "")
            
            # 3. 如果 LLM 识别成功且明确，直接返回
            if llm_channel and llm_channel != ChannelType.UNKNOWN:
                result = ChannelInferenceResult(
                    channel=InferencerChannelType(llm_channel.value),
                    confidence=1.0,
                    reason=f"LLM识别：根据明确关键词识别为{llm_channel.value}",
                    matched_keywords=[channel_str.strip()] if channel_str else [],
                    suggestion=None
                )
                return llm_channel, result
            
            # 4. LLM 识别为 unknown，启用智能推断
            self.logger.info("LLM 识别为 unknown，启用智能推断...")
            
            inference_result = self.channel_inferencer.infer(question, parsed_entities)
            
            # 将推断器的 ChannelType 转换为服务的 ChannelType
            service_channel = ChannelType(inference_result.channel.value)
            
            self.logger.info(
                f"智能推断结果: channel={inference_result.channel.value}, "
                f"confidence={inference_result.confidence:.2f}, "
                f"reason={inference_result.reason}"
            )
            
            return service_channel, inference_result
            
        except Exception as e:
            self.logger.error(f"渠道意图识别失败: {e}")
            # 异常时默认返回信息价
            return ChannelType.INFORMATION_PRICE, ChannelInferenceResult(
                channel=InferencerChannelType.INFORMATION_PRICE,
                confidence=0.5,
                reason="识别过程异常，使用默认渠道",
                matched_keywords=[],
                suggestion="如需其他渠道请明确说明"
            )

    def _parse_channel_string(self, channel_str: str) -> Optional[ChannelType]:
        """解析渠道字符串"""
        if not channel_str:
            return None
        channel_map = {
            "information_price": ChannelType.INFORMATION_PRICE,
            "manufacturer_price": ChannelType.MANUFACTURER_PRICE,
            "zc_price": ChannelType.ZC_PRICE,
            "unknown": ChannelType.UNKNOWN,
        }
        return channel_map.get(channel_str)

    # ========== 渐进式实体补全相关方法 ==========
    
    async def process_dialogue_query(
        self,
        session_id: Optional[str],
        user_input: str
    ) -> Dict[str, Any]:
        """
        处理对话式价格查询（渐进式实体补全）
        
        Args:
            session_id: 会话ID（可选，首次为空）
            user_input: 用户输入
        
        Returns:
            包含会话状态、实体收集进度、下一步操作等的字典
        """
        try:
            # 1. 提取实体
            extracted_entities = await self.extract_entities(user_input)
            self.logger.info(f"从输入提取的实体: {extracted_entities}")
            
            # 2. 渠道识别（传入实体辅助推断）
            channel_result, inference_detail = await self.identify_channel(
                user_input, 
                extracted_entities
            )
            
            # 3. 使用对话管理器处理
            dialogue_result = self.dialogue_manager.process_user_input(
                session_id=session_id,
                user_input=user_input,
                extracted_entities=extracted_entities,
                channel=channel_result.value,
                channel_confidence=inference_detail.confidence
            )
            
            # 4. 添加渠道推断信息
            dialogue_result["channel_info"] = {
                "channel": channel_result.value,
                "confidence": inference_detail.confidence,
                "reason": inference_detail.reason,
                "inferred": inference_detail.confidence < 1.0
            }
            
            # 5. 如果可以查询，执行价格查询
            if dialogue_result.get("can_query") and dialogue_result.get("status") == "ready":
                # 执行实际查询
                price_result = await self.process_price_recommendation(
                    channel_result.value,
                    dialogue_result["entities"]
                )
                dialogue_result["price_result"] = price_result
                
                # 标记会话完成
                session = self.dialogue_manager.get_session(dialogue_result["session_id"])
                if session:
                    session.status = session.status.COMPLETED
            
            return dialogue_result
            
        except Exception as e:
            self.logger.error(f"对话式查询处理失败: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "session_id": session_id
            }
    
    async def execute_dialogue_query(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        执行对话会话中已收集条件的查询
        
        Args:
            session_id: 会话ID
        
        Returns:
            查询结果
        """
        try:
            session = self.dialogue_manager.get_session(session_id)
            if not session:
                return {
                    "success": False,
                    "error_message": "会话已过期或不存在，请重新开始"
                }
            
            if not session.entities.get("materialName"):
                return {
                    "success": False,
                    "error_message": "缺少材料名称，无法查询"
                }
            
            # 执行查询
            channel = session.channel or ChannelType.INFORMATION_PRICE.value
            price_result = await self.process_price_recommendation(
                channel,
                session.entities
            )
            
            # 标记会话完成
            session.status = session.status.COMPLETED
            
            return {
                "success": True,
                "session_id": session_id,
                "entities": session.entities,
                "price_result": price_result
            }
            
        except Exception as e:
            self.logger.error(f"执行对话查询失败: {e}")
            return {
                "success": False,
                "error_message": str(e)
            }
    
    def get_dialogue_session(self, session_id: str) -> Optional[Dict]:
        """获取对话会话状态"""
        session = self.dialogue_manager.get_session(session_id)
        if session:
            return session.to_dict()
        return None
    
    def clear_dialogue_session(self, session_id: str):
        """清除对话会话"""
        self.dialogue_manager.clear_session(session_id)

    # 实体抽取入口
    async def extract_entities(self, question: str) -> dict:
        """实体抽取"""
        try:
            if not self.entity_extract_template:
                # 如果模板未初始化，返回空字典
                return {}
                
            entities = await asyncio.to_thread(
                self.llm.predict_prompt, 
                self.entity_extract_template,
                {"user_question": question},
            )
            # 解析实体提取结果
            parsed_entities = self._parse_llm_output(entities)
            parsed_entities = self._normalize_price_query_entities(parsed_entities)
            if not parsed_entities:
                raise Exception("实体解析失败")
            #  检查是否包含时间字段： # 如果没有时间字段，则传递默认时间参数：近一年
            # if "startReleaseDate" not in parsed_entities or "endReleaseDate" not in parsed_entities or parsed_entities["startReleaseDate"] is None or parsed_entities["endReleaseDate"] is None :
            #     current_year = datetime.now().year
            #     # 获取当前时间
            #     now = datetime.now()
            #     # 格式化为 "YYYY-MM-DD"
            #     formatted_date = now.strftime("%Y-%m-%d")
            #     parsed_entities["startReleaseDate"] = f"{current_year-3}-01-01"
            #     # parsed_entities["endReleaseDate"] = f"{formatted_date}"
            # self.logger.info(f"填充时间后的实体结果: {parsed_entities}")

            return parsed_entities if parsed_entities else {}
        except Exception as e:
            self.logger.error(f"实体抽取失败: {e}")
            return {}       

    def _build_price_query_description(self, parsed_entities: Dict[str, Any], channel: str) -> str:
        """构造用于报告生成的查询描述"""
        channel_labels = {
            ChannelType.INFORMATION_PRICE.value: "信息价",
            ChannelType.MANUFACTURER_PRICE.value: "厂商报价",
            ChannelType.ZC_PRICE.value: "智诚价",
        }
        parts = [f"查询{channel_labels.get(channel, channel)}"]
        if parsed_entities.get("province"):
            parts.append(str(parsed_entities["province"]))
        if parsed_entities.get("city"):
            parts.append(str(parsed_entities["city"]))
        if parsed_entities.get("brand"):
            parts.append(f"{parsed_entities['brand']}品牌")
        if parsed_entities.get("materialName"):
            parts.append(str(parsed_entities["materialName"]))
        if parsed_entities.get("materialModelSpec"):
            parts.append(f"规格{parsed_entities['materialModelSpec']}")
        return "".join(parts)

    def _build_summary_metrics(
        self,
        analysis_results: Dict[str, Any],
        most_common_unit: str,
        total_count: int
    ) -> Dict[str, Any]:
        """提取前端直接可用的摘要指标"""
        unit_metrics = analysis_results.get(most_common_unit, {}) if most_common_unit else {}
        recommend = unit_metrics.get("recommend_kmeans", {})
        return {
            "total_count": total_count,
            "most_common_unit": most_common_unit,
            "valid_count": unit_metrics.get("valid_count", 0),
            "min_price": self._safe_number_from_range(unit_metrics.get("price_range"), 0),
            "max_price": self._safe_number_from_range(unit_metrics.get("price_range"), 1),
            "mean_price": unit_metrics.get("mean_price"),
            "median_price": unit_metrics.get("median_price"),
            "recommend_mode": recommend.get("mode"),
            "recommended_prices": recommend.get("prices", []),
            "recommend_reason": recommend.get("reason", ""),
        }

    def _safe_number_from_range(self, value_range: Any, index: int) -> Optional[float]:
        if isinstance(value_range, (list, tuple)) and len(value_range) > index:
            try:
                return float(value_range[index])
            except (TypeError, ValueError):
                return None
        return None

    def _build_chart_data(
        self,
        records: List[Dict[str, Any]],
        analysis_results: Dict[str, Any],
        most_common_unit: str
    ) -> Dict[str, Any]:
        """构造前端可直接消费的图表数据"""
        chart_data: Dict[str, Any] = {
            "summary": {"labels": [], "values": [], "unit": most_common_unit},
            "histogram": {"bins": [], "counts": [], "unit": most_common_unit},
            "unit_distribution": {"labels": [], "values": []},
            "time_trend": {"labels": [], "values": [], "unit": most_common_unit},
        }

        unit_metrics = analysis_results.get(most_common_unit, {}) if most_common_unit else {}
        summary_pairs = [
            ("最低价", self._safe_number_from_range(unit_metrics.get("price_range"), 0)),
            ("平均价", unit_metrics.get("mean_price")),
            ("中位价", unit_metrics.get("median_price")),
            ("最高价", self._safe_number_from_range(unit_metrics.get("price_range"), 1)),
        ]
        chart_data["summary"] = {
            "labels": [label for label, _ in summary_pairs],
            "values": [value for _, value in summary_pairs],
            "unit": most_common_unit,
        }

        unit_counts: Dict[str, int] = {}
        prices: List[float] = []
        trend_values: Dict[str, List[float]] = {}
        time_field_candidates = ["releaseDate", "releaseTime", "publishTime", "date"]

        for record in records:
            unit = str(record.get("unit") or "UNKNOWN")
            unit_counts[unit] = unit_counts.get(unit, 0) + 1

            try:
                prices.append(float(record.get("price")))
            except (TypeError, ValueError):
                pass

            time_value = None
            for field_name in time_field_candidates:
                if record.get(field_name):
                    time_value = str(record.get(field_name))
                    break
            if time_value:
                time_key = time_value[:7]
                try:
                    trend_price = float(record.get("price"))
                    trend_values.setdefault(time_key, []).append(trend_price)
                except (TypeError, ValueError):
                    pass

        chart_data["unit_distribution"] = {
            "labels": list(unit_counts.keys()),
            "values": list(unit_counts.values()),
        }

        if prices:
            counts, bin_edges = np.histogram(prices, bins=min(8, max(1, len(set(prices)))))
            chart_data["histogram"] = {
                "bins": [round(float(edge), 2) for edge in bin_edges.tolist()],
                "counts": [int(count) for count in counts.tolist()],
                "unit": most_common_unit,
            }

        if trend_values:
            sorted_keys = sorted(trend_values.keys())
            chart_data["time_trend"] = {
                "labels": sorted_keys,
                "values": [
                    round(sum(trend_values[key]) / len(trend_values[key]), 2)
                    for key in sorted_keys
                ],
                "unit": most_common_unit,
            }

        return chart_data

    def _build_table_data(self, records: List[Dict[str, Any]], limit: int = 20) -> Dict[str, Any]:
        """构造前端表格数据"""
        rows = records[:limit]
        columns = [
            "materialName",
            "materialModelSpec",
            "brand",
            "price",
            "unit",
            "province",
            "city",
            "releaseDate",
        ]
        return {
            "columns": columns,
            "rows": rows,
            "total_rows": len(records),
            "display_rows": len(rows),
        }

    def _build_price_report_fallback(
        self,
        parsed_entities: Dict[str, Any],
        channel_result: str,
        summary_metrics: Dict[str, Any],
        total_count: int
    ) -> str:
        """LLM 失败时使用的回退报告"""
        channel_labels = {
            ChannelType.INFORMATION_PRICE.value: "信息价",
            ChannelType.MANUFACTURER_PRICE.value: "厂商报价",
            ChannelType.ZC_PRICE.value: "智诚价",
        }
        material_name = parsed_entities.get("materialName", "该材料")
        location = "".join([
            str(parsed_entities.get("province", "")),
            str(parsed_entities.get("city", "")),
        ]) or "指定范围"
        unit = summary_metrics.get("most_common_unit") or "-"
        mean_price = summary_metrics.get("mean_price")
        median_price = summary_metrics.get("median_price")
        min_price = summary_metrics.get("min_price")
        max_price = summary_metrics.get("max_price")
        recommend_mode = summary_metrics.get("recommend_mode")
        recommended_prices = summary_metrics.get("recommended_prices", [])

        lines = [
            f"{channel_labels.get(channel_result, channel_result)}价格分析报告",
            f"查询对象：{location}{material_name}",
            f"样本数量：共筛选出 {total_count} 条有效记录，主单位为 {unit}。",
        ]
        if None not in (mean_price, median_price, min_price, max_price):
            lines.append(
                f"核心指标：平均价 {mean_price:.2f}，中位价 {median_price:.2f}，"
                f"价格区间 {min_price:.2f} 至 {max_price:.2f}。"
            )
        if recommend_mode == "two-tier" and len(recommended_prices) >= 2:
            lines.append(
                f"推荐结论：当前价格呈现双峰分层，可参考两个代表价位 "
                f"{recommended_prices[0]:.2f} 和 {recommended_prices[1]:.2f} 元/{unit}。"
            )
        elif recommended_prices:
            lines.append(f"推荐结论：当前可参考代表价 {recommended_prices[0]:.2f} 元/{unit}。")
        recommend_reason = summary_metrics.get("recommend_reason")
        if recommend_reason:
            lines.append(f"判定依据：{recommend_reason}")
        return "\n".join(lines)

    async def _generate_price_report(
        self,
        user_question: Optional[str],
        parsed_entities: Dict[str, Any],
        detail_answer_json: str,
        summary_metrics: Dict[str, Any],
        channel_result: str,
        total_count: int,
        trace_id: Optional[str] = None,
    ) -> str:
        """基于统计结果生成价格报告"""
        if not self.llm:
            return self._build_price_report_fallback(
                parsed_entities, channel_result, summary_metrics, total_count
            )

        prompt_question = user_question or self._build_price_query_description(parsed_entities, channel_result)
        parsed_entities_str = json.dumps(parsed_entities, ensure_ascii=False, indent=2)
        started_at = time.time()
        self._price_trace_log(
            trace_id,
            "report.generate.start",
            channel=channel_result,
            total_count=total_count,
            entities=self._summarize_price_entities(parsed_entities),
        )
        try:
            report = await asyncio.to_thread(
                self.llm.predict_prompt,
                Price_Answer_TEMPLATE,
                {
                    "user_question": prompt_question,
                    "parsed_entities": parsed_entities_str,
                    "detail_answer": detail_answer_json,
                },
                False,
                None,
                1000,
                8.0,
                0
            )
            if report:
                self._price_trace_log(
                    trace_id,
                    "report.generate.success",
                    duration_ms=int((time.time() - started_at) * 1000),
                    report_length=len(report),
                )
                return report
        except Exception as e:
            self.logger.error(f"[Price Trace][{trace_id or 'no-trace'}] report.generate.error | error={self._safe_json(str(e))}")

        self._price_trace_log(
            trace_id,
            "report.generate.fallback",
            duration_ms=int((time.time() - started_at) * 1000),
        )
        return self._build_price_report_fallback(
            parsed_entities, channel_result, summary_metrics, total_count
        )

    # 价格推荐处理入口
    async def process_price_recommendation(
        self,
        channel_result: str,
        parsed_entities: dict,
        user_question: Optional[str] = None,
        generate_report: bool = True,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """处理价格推荐查询"""
        start_time = time.time()
        parsed_entities = self._normalize_price_query_entities(parsed_entities)
        self._price_trace_log(
            trace_id,
            "request.start",
            channel=channel_result,
            generate_report=generate_report,
            entities=self._summarize_price_entities(parsed_entities),
            user_question=(user_question[:120] + "...") if user_question and len(user_question) > 120 else user_question,
        )
        try:
            # ✅ 阶段1：先快速检查数据总数（不获取具体数据，节省时间和带宽）
            phase_started_at = time.time()
            self._price_trace_log(trace_id, "phase.quick_check.start", channel=channel_result)
            total_count = await self._quick_check_data_count(channel_result, parsed_entities, trace_id=trace_id)
            self._price_trace_log(
                trace_id,
                "phase.quick_check.end",
                total_count=total_count,
                duration_ms=int((time.time() - phase_started_at) * 1000),
            )
            
            # ✅ 数据量检查和引导
            volume_check = check_data_volume_and_guide(
                record_count=total_count,
                current_entities=parsed_entities,
                channel=channel_result
            )
            
            self._price_trace_log(
                trace_id,
                "phase.volume_check",
                total_count=total_count,
                estimated_tokens=volume_check["estimated_tokens"],
                can_process=volume_check["can_process"],
            )
            
            # 如果数据量过大，返回引导消息，不进行后续处理
            # if not volume_check["can_process"]:
            #     total_time = time.time() - start_time
                
            #     # ✅ 打印数据量过大的警告信息
            #     self.logger.warning("=" * 80)
            #     self.logger.warning("⚠️  数据量过大警告")
            #     self.logger.warning("=" * 80)
            #     self.logger.warning(f"🔍 查询条件: {parsed_entities}")
            #     self.logger.warning(f"📈 数据量: {total_count:,} 条 (超过阈值 1000 条)")
            #     self.logger.warning(f"🎫 预计 Token: {volume_check['estimated_tokens']:,} tokens")
            #     self.logger.warning(f"💡 建议补充筛选条件: {volume_check.get('suggested_filters', [])}")
            #     self.logger.warning(f"⏱️  响应时间: {total_time:.3f}s")
            #     self.logger.warning("=" * 80)
                
            #     return {
            #         "answer": volume_check["guidance_message"],
            #         "contexts": [],
            #         "metadata": {
            #             "total_time": total_time,
            #             "volume_check": volume_check,
            #             "need_refinement": True,
            #             "performance": {
            #                 "total_time": total_time,
            #                 "original_count": total_count,
            #                 "estimated_tokens": volume_check['estimated_tokens'],
            #                 "suggested_filters": volume_check.get('suggested_filters', [])
            #             }
            #         },
            #         "success": False,
            #         "total_count": total_count,
            #         "error_message": "数据量过大，需要补充筛选条件",
            #         "intent": "price_recommendation_refinement",  # 特殊意图标记
            #         "suggested_filters": volume_check.get("suggested_filters", [])
            #     }
            
            # ✅ 阶段2：数据量可控，获取具体数据
            phase_started_at = time.time()
            self._price_trace_log(trace_id, "phase.query_data.start", channel=channel_result)
            price_data = await self._query_price_data(channel_result, parsed_entities, trace_id=trace_id)
            self._price_trace_log(
                trace_id,
                "phase.query_data.end",
                success=price_data.get("success"),
                total_count=price_data.get("total_count"),
                error=price_data.get("error"),
                duration_ms=int((time.time() - phase_started_at) * 1000),
            )
            # self.logger.info(f"返回的查询结果: {price_data['totalCount']}")
            if not price_data or not price_data.get("success"):
                raise Exception(f"价格查询失败: {price_data.get('error', '未知错误')}")

            # 对于模糊查询的结果进行匹配过滤
            phase_started_at = time.time()
            filtered_items =self.fillter_items(
                query=parsed_entities.get("materialName", ""),
                items=price_data.get("data", []),
                alpha=0.4,
                T_keep=0.75,
                T_drop=0.50
            )
            price_data["data"] = filtered_items
            price_data["total_count"] = len(filtered_items)
            self._price_trace_log(
                trace_id,
                "phase.filter.end",
                original_count=total_count,
                filtered_count=price_data["total_count"],
                duration_ms=int((time.time() - phase_started_at) * 1000),
            )

            # # 过滤模糊匹配的结果，只保留完全匹配的
            # filtered_records = []
            # for item in price_data.get("data", []):
            #     material_name = str(item.get("materialName", "")).strip()
            #     target_name = str(parsed_entities.get("materialName", "")).strip()

            #     # 只有完全匹配才保留
            #     if target_name and material_name == target_name:
            #         filtered_records.append(item)

            # price_data["data"] = filtered_records
            # price_data["total_count"] = len(filtered_records)

            if price_data['total_count'] ==0:
                total_time = time.time() - start_time
                
                # ✅ 打印查询结果为空的信息
                self.logger.warning("=" * 80)
                self.logger.warning("⚠️  查询结果为空")
                self.logger.warning("=" * 80)
                self.logger.warning(f"🔍 查询条件: {parsed_entities}")
                self.logger.warning(f"📈 数据量: 0 条 (过滤后无符合条件的数据)")
                self.logger.warning(f"⏱️  响应时间: {total_time:.3f}s")
                self.logger.warning("=" * 80)
                
                return {
                    "answer": f"抱歉，价格查询失败,查询条件没有符合的样本",
                    "contexts": [],
                    "metadata": {
                        "total_time": total_time,
                        "error": "查询条件没有符合的样本",
                        "performance": {
                            "total_time": total_time,
                            "filtered_count": 0
                        }
                    },
                    "success": False,
                    "total_count": price_data['total_count'],
                    "error_message":"查询条件没有符合的样本",
                    "intent": "price_recommendation"
                }

            # 价格分析和推荐
            analysis_start = time.time()
            try:
                unit_result = self.analyze_by_unit(price_data["data"], unit_col="unit")
                self._price_trace_log(
                    trace_id,
                    "phase.analyze.end",
                    most_common_unit=unit_result.get("most_common_unit"),
                    result_unit_count=len(unit_result.get("results", {})),
                    duration_ms=int((time.time() - analysis_start) * 1000),
                )
            except Exception as e:
                self.logger.error(f"[Price Trace][{trace_id or 'no-trace'}] phase.analyze.error | error={self._safe_json(str(e))}")
                raise

            # 获得可视化内容：区间频数分布图 and md 原始数据展示
            if price_data and price_data['total_count'] >0:
                md_table = self.parse_material_response(channel_result, unit_result['most_common_records'])
            
            # 获得展示的数据
            show_data = unit_result['most_common_records']
            
            # ✅ 计算最终的性能指标
            total_time = time.time() - start_time
            check_time = time.time() - start_time  # 数据检查耗时
            analysis_time = time.time() - analysis_start  # 分析耗时
            
            # 计算实际处理的数据的 token 数
            actual_tokens = TokenEstimator.estimate_tokens(price_data['total_count'])
            
            # ✅ 计算传给 LLM 的实际 Token（用于统计）
            parsed_entities_json = json.dumps(parsed_entities, ensure_ascii=False, indent=2)
            detail_answer_json = json.dumps(unit_result['results'], ensure_ascii=False, indent=2)
            llm_input_text = f"{parsed_entities_json}\n{detail_answer_json}"
            llm_input_tokens = len(llm_input_text) // 2  # 粗略估算

            summary_metrics = self._build_summary_metrics(
                unit_result["results"],
                unit_result["most_common_unit"],
                price_data["total_count"]
            )
            chart_data = self._build_chart_data(
                show_data,
                unit_result["results"],
                unit_result["most_common_unit"]
            )
            table_data = self._build_table_data(show_data)
            report = None
            if generate_report:
                report = await self._generate_price_report(
                    user_question=user_question,
                    parsed_entities=parsed_entities,
                    detail_answer_json=detail_answer_json,
                    summary_metrics=summary_metrics,
                    channel_result=channel_result,
                    total_count=price_data["total_count"],
                    trace_id=trace_id,
                )
            
            self._price_trace_log(
                trace_id,
                "request.success",
                original_count=total_count,
                filtered_count=price_data["total_count"],
                returned_count=len(show_data),
                estimated_tokens=volume_check["estimated_tokens"],
                filtered_data_tokens=actual_tokens,
                llm_input_tokens=llm_input_tokens,
                token_optimization_rate=f"{(1 - llm_input_tokens/volume_check['estimated_tokens'])*100:.1f}%",
                check_time_ms=int(check_time * 1000),
                analysis_time_ms=int(analysis_time * 1000),
                total_time_ms=int(total_time * 1000),
                summary_metrics=summary_metrics,
            )
            
            return {
                "answer": report,
                "report": report,
                "parsed_entities": json.dumps(parsed_entities, ensure_ascii=False, indent=2),
                "detail_answer": json.dumps(unit_result['results'], ensure_ascii=False, indent=2),
                "contexts": [],  # 价格推荐不需要检索上下文
                "summary_metrics": summary_metrics,
                "chart_data": chart_data,
                "table_data": table_data,
                "metadata": {
                    "channel": channel_result,
                    "entities": parsed_entities,
                    "price_analysis": unit_result['results'],
                    "md_table": md_table if 'md_table' in locals() else None,
                    "summary_metrics": summary_metrics,
                    "chart_data": chart_data,
                    "table_data": table_data,
                    "performance": {
                        "data_processing": {
                            "total_time": total_time,
                            "check_time": check_time,
                            "analysis_time": analysis_time,
                            "original_count": total_count,
                            "filtered_count": price_data['total_count'],
                            "returned_count": len(show_data)
                        },
                        "token_analysis": {
                            "original_data_tokens": volume_check['estimated_tokens'],
                            "filtered_data_tokens": actual_tokens,
                            "llm_input_tokens": llm_input_tokens,
                            "token_optimization_rate": f"{(1 - llm_input_tokens/volume_check['estimated_tokens'])*100:.1f}%"
                        }
                    }
                },
                "success": True,
                "intent": "price_recommendation",
                "total_count": price_data['total_count'],
                "price_data": show_data,
                "most_common_unit": unit_result['most_common_unit']
            }
            
 
        except Exception as e:
            total_time = time.time() - start_time
            
            self.logger.error(
                f"[Price Trace][{trace_id or 'no-trace'}] request.error | "
                f"channel={self._safe_json(channel_result)}, "
                f"entities={self._safe_json(self._summarize_price_entities(parsed_entities))}, "
                f"duration_ms={int(total_time * 1000)}, "
                f"error={self._safe_json(str(e))}"
            )
            
            # 安全获取 total_count
            total_count_safe = 0
            try:
                if 'price_data' in locals() and price_data:
                    total_count_safe = price_data.get('total_count', 0)
            except:
                pass
            
            # 检查是否是因为数据为空导致的错误
            error_msg = str(e)
            if "Found array with 0 sample(s)" in error_msg or total_count_safe == 0:
                # 返回友好的提示信息
                return {
                    "answer": "抱歉，未查询到符合条件的价格数据",
                    "contexts": [],
                    "metadata": {
                        "total_time": total_time,
                        "error": "未查询到符合条件的价格数据",
                        "performance": {
                            "total_time": total_time,
                            "filtered_count": 0
                        }
                    },
                    "success": False,
                    "error_message": "未查询到符合条件的价格数据",
                    "total_count": total_count_safe,
                    "intent": "price_recommendation"
                }
            else:
                # 返回原始错误信息
                return {
                    "answer": f"抱歉，价格查询失败：{error_msg}",
                    "contexts": [],
                    "metadata": {
                        "total_time": total_time,
                        "error": error_msg,
                        "performance": {
                            "total_time": total_time,
                            "filtered_count": 0
                        }
                    },
                    "success": False,
                    "error_message": error_msg,
                    "total_count": total_count_safe,
                    "intent": "price_recommendation"
                }

    # 价格推荐流式处理入口
    async def stream_price_recommendation(self, channel_result:str,parsed_entities:dict):
        """价格推荐的流式处理"""
        try:
            # 先获取完整结果
            result = await self.process_price_recommendation(channel_result,parsed_entities)
            
            # 存储metadata到实例变量中
            self._last_price_metadata = result.get("metadata", {})
            self._last_price_metadata["success"] = result.get("success", False)
            if self._last_price_metadata["success"] == False:
                self._last_price_metadata["error_message"] = result.get("error_message", "unknown error")
            self._last_price_metadata["price_data"] = result.get("price_data", [])
            self._last_price_metadata["total_count"] = result.get("total_count", 0)
            return {
                "parsed_entities": result.get("parsed_entities", {}),
                "answer": result.get("answer", ""),
                "detail_answer": result.get("detail_answer", ""),
                "metadata": self._last_price_metadata,
                "success": result.get("success", False),
                "error_message": result.get("error_message", ""),
                "intent": "price_recommendation"
            }
                
        except Exception as e:
            # 错误时也要存储metadata
            self._last_price_metadata = {
                "success": False,
                "error": str(e)
            }
            return{
                "answer": f"价格推荐流式处理失败：{str(e)}",
                "contexts": [],
                "metadata": self._last_price_metadata,
                "success": False,
                "error_message": str(e),
                "intent": "price_recommendation"
            }
            # yield f"价格推荐流式处理失败：{str(e)}"

    # 获取最后一次价格推荐的元数据
    def get_last_price_metadata(self) -> dict:
        """获取最后一次价格推荐的元数据"""
        return self._last_price_metadata

    # 解析材料响应为Markdown表格
    def parse_material_response(self, 
                            channel: ChannelType,
                            response_json: list, 
                            top_n: Optional[int] = None  # 默认 None
                            ) -> str:
        """解析材料响应为Markdown表格"""
        try:
            # 如果 top_n 有传值才截取，否则保留全部
            items = response_json if top_n is None else response_json[:top_n]

            if not items:
                return "⚠️ 没有查询到材料信息"
            
            # 根据渠道类型设置表头
            if channel == ChannelType.INFORMATION_PRICE:
                tabs = "| 编号 | 材料名称 | 规格型号 | 价格(元) | 单位 | 省份 | 城市 | 发布时间 |"
                table_lines = [
                    tabs,
                    "|------|-----------|-----------|------|------|------|------|-----------|"
                ]
            elif channel == ChannelType.MANUFACTURER_PRICE:
                tabs = "| 编号 | 材料名称 | 规格型号 | 品牌 | 价格(元) | 单位 | 省份 | 城市 | 发布时间 |"
                table_lines = [
                    tabs,
                    "|------|-----------|-----------|------|------|------|------|-----------|------|"
                ]
            else:
                tabs = "| 编号 | 材料名称 | 规格型号 | 价格(元) | 单位 | 省份 | 城市 | 发布时间 |"
                table_lines = [
                    tabs,
                    "|------|-----------|-----------|------|------|------|------|-----------|"
                ]

            # 构造表格内容
            for i, item in enumerate(items):
                item = {k: (v.replace('\\', '/').replace('\n', ' / ').replace('\r', ' ')
                            if isinstance(v, str) else v) for k, v in item.items()}

                if channel == ChannelType.INFORMATION_PRICE:
                    table_lines.append(
                        f"| {i+1} | {item.get('materialName', '')} | "
                        f"{item.get('materialModelSpec', '')} | "
                        f"{item.get('price', '')} | {item.get('unit', '')} | "
                        f"{item.get('province', '')} | {item.get('city', '')} | "
                        f"{item.get('releaseDate', '')} |"
                    )
                elif channel == ChannelType.MANUFACTURER_PRICE:
                    table_lines.append(
                        f"| {i+1} | {item.get('materialName', '')} | "
                        f"{item.get('materialModelSpec', '')} | "
                        f"{item.get('brand', '')} | "
                        f"{item.get('price', '')} | {item.get('unit', '')} | "
                        f"{item.get('province', '')} | {item.get('city', '')} | "
                        f"{item.get('releaseDate', '')} |"
                    )
                else:
                    table_lines.append(
                        f"| {i+1} | {item.get('materialName', '')} | "
                        f"{item.get('materialModelSpec', '')} | "
                        f"{item.get('price', '')} | {item.get('unit', '')} | "
                        f"{item.get('province', '')} | {item.get('city', '')} | "
                        f"{item.get('releaseDate', '')} |"
                    )
            return "\n".join(table_lines)

        except Exception as e:
            return {"error": f"响应解析失败: {str(e)}"}
    
    # 解析LLM输出的JSON
    def _parse_llm_output(self, llm_output: str) -> dict:
        """解析LLM输出的JSON"""
        try:
            clean_output = llm_output.strip().removeprefix("```json").removesuffix("```").strip()
            return json.loads(clean_output)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            return {}

    # 快速检查数据总数（不返回具体数据，仅用于数据量判断）
    async def _quick_check_data_count(self, channel: str, entities: dict, trace_id: Optional[str] = None) -> int:
        """快速检查数据总数，不返回具体数据"""
        try:
            # 创建副本，设置只返回总数不返回具体数据
            check_entities = entities.copy()
            
            if channel == ChannelType.INFORMATION_PRICE:
                result = await asyncio.to_thread(self._get_infor_material, check_entities)
            elif channel == ChannelType.MANUFACTURER_PRICE:
                result = await asyncio.to_thread(self._get_factory_material, check_entities)
            elif channel == ChannelType.ZC_PRICE:
                result = await asyncio.to_thread(self._get_zhicheng_info_material, check_entities)
            else:
                result = await asyncio.to_thread(self._get_infor_material, check_entities)
            
            if result.get("code") == 200:
                data = result.get("data", {})
                total_count = data.get("totalCount", 0)
                self._price_trace_log(
                    trace_id,
                    "api.quick_check.success",
                    channel=channel,
                    total_count=total_count,
                )
                return total_count
            else:
                self.logger.warning(
                    f"[Price Trace][{trace_id or 'no-trace'}] api.quick_check.non_200 | "
                    f"channel={self._safe_json(channel)}, code={self._safe_json(result.get('code'))}, "
                    f"message={self._safe_json(result.get('message'))}"
                )
                return 0
        except Exception as e:
            self.logger.error(f"[Price Trace][{trace_id or 'no-trace'}] api.quick_check.error | error={self._safe_json(str(e))}")
            return 0
    
    # 根据渠道查询价格数据
    async def _query_price_data(self, channel: str, entities: dict, trace_id: Optional[str] = None) -> dict:
        """根据渠道查询价格数据"""
        started_at = time.time()
        try:
            self._price_trace_log(
                trace_id,
                "api.query.start",
                channel=channel,
                entities=self._summarize_price_entities(entities),
            )
            # 渠道意图识别
            if channel == ChannelType.INFORMATION_PRICE:
                result = await asyncio.to_thread(self._get_infor_material, entities)
            elif channel == ChannelType.MANUFACTURER_PRICE:
                result = await asyncio.to_thread(self._get_factory_material, entities)
            elif channel == ChannelType.ZC_PRICE:
                result = await asyncio.to_thread(self._get_zhicheng_info_material, entities)
            else:
                # 未知渠道:使用默认的信息价渠道
                result = await asyncio.to_thread(self._get_infor_material, entities)
                # ，把已有的渠道列出出来，让用户重新输入
                # return {"success": False, "error": f"未知渠道: {channel}"}

            if result.get("code") == 200:
                data = result.get("data", {})
                self._price_trace_log(
                    trace_id,
                    "api.query.success",
                    channel=channel,
                    total_count=data.get("totalCount", 0),
                    duration_ms=int((time.time() - started_at) * 1000),
                )
                return {
                    "success": True,
                    "data": data.get("list", []),
                    "total_count": data.get("totalCount", 0)
                }
            else:
                self.logger.warning(
                    f"[Price Trace][{trace_id or 'no-trace'}] api.query.non_200 | "
                    f"channel={self._safe_json(channel)}, code={self._safe_json(result.get('code'))}, "
                    f"message={self._safe_json(result.get('message'))}, "
                    f"duration_ms={int((time.time() - started_at) * 1000)}"
                )
                return {
                    "success": False, 
                    "error": result.get("message", "API调用失败")
                }
                
        except Exception as e:
            self.logger.error(
                f"[Price Trace][{trace_id or 'no-trace'}] api.query.error | "
                f"channel={self._safe_json(channel)}, duration_ms={int((time.time() - started_at) * 1000)}, "
                f"error={self._safe_json(str(e))}"
            )
            return {"success": False, "error": str(e)}

    #   信息价查询接口
    def _get_infor_material(self, data: dict) -> dict:
        """信息价查询接口"""
        # 根据配置选择使用 Mock 或真实 API
        if USE_MOCK_API:
            url = f"{MOCK_API_BASE_URL}/zhichengInfo/getZhichengInfoMaterial"
            self.logger.info(f"使用 Mock API: {url}")
            # Mock模式使用旧参数
            payload = {
                "accountId": PRICE_API_ACCOUNT_ID,
                "matchMethod": 1,
                "categoryOneLevelName": "",
                "categoryTwoLevelName": "",
                "categoryThreeLevelName": "",
                "minPrice": data.get("minPrice", ""),
                "maxPrice": data.get("maxPrice", ""),
                "releaseDepartment": data.get("releaseDepartment", ""),
                "startReleaseDate": data.get("startReleaseDate", ""),
                "endReleaseDate": data.get("endReleaseDate", ""),
                "province": data.get("province", ""),
                "city": data.get("city", ""),
                "materialModelSpec": data.get("materialModelSpec", ""),
                "materialName": data.get("materialName", ""),
                "returnNumber": data.get("returnNumber", 10000),
                "returnTotalCount": 1
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            return response.json()
        else:
            # 真实 API 使用新接口，分页拉取数据
            return self._fetch_price_data_paginated(data, channel_type="information")
    
    def _fetch_price_data_paginated(self, data: dict, channel_type: str = "information") -> dict:
        """
        分页获取价格数据，每页最多2000条
        
        Args:
            data: 查询参数
            channel_type: 渠道类型 (information/factory/zhicheng)
        
        Returns:
            合并后的结果字典
        """
        channel_configs = {
            "information": {
                "path": "/material/infoRelatedMaterials/getRelatedMaterial",
                "service": "infoRelatedMaterials",
            },
            "factory": {
                "path": "/material/factoryRelatedMaterials/getRelatedMaterial",
                "service": "factoryRelatedMaterials",
            },
        }
        config = channel_configs.get(channel_type, channel_configs["factory"])
        url = f"{REAL_API_BASE_URL}{config['path']}"
        self.logger.info(f"使用真实 API (分页拉取): {url}")
        
        page_size = 2000  # 每页最多2000条
        all_records = []
        total_count_from_api = 0
        page = 1
        max_pages = 50  # 最多拉取50页，防止无限循环
        
        headers = {"Content-Type": "application/json"}
        
        while page <= max_pages:
            payload = {
                "accountId": PRICE_API_ACCOUNT_ID,
                "matchMethod": 2,  # 精确查询
                "excludeExactMatch": False,
                "checkState": 1,
                "belongDataPool": 2,
                "categoryOneLevelId": None,
                "categoryTwoLevelId": None,
                "categoryThreeLevelId": None,
                "categoryOneLevelName": None,
                "categoryTwoLevelName": None,
                "categoryThreeLevelName": None,
                "materialName": data.get("materialName", None),
                "materialCode": None,
                "materialModelSpec": data.get("materialModelSpec", None) if data.get("materialModelSpec") else None,
                "domain": None,
                "releaseDepartment": data.get("releaseDepartment", None) if data.get("releaseDepartment") else None,
                "startReleaseDate": data.get("startReleaseDate", None) if data.get("startReleaseDate") else None,
                "endReleaseDate": data.get("endReleaseDate", None) if data.get("endReleaseDate") else None,
                "minPrice": data.get("minPrice", None) if data.get("minPrice") else None,
                "maxPrice": data.get("maxPrice", None) if data.get("maxPrice") else None,
                "province": data.get("province", None) if data.get("province") else None,
                "city": data.get("city", None) if data.get("city") else None,
                "provinceId": None,
                "cityId": None,
                "enterpriseId": None,
                "enterpriseName": None,
                "queryAccountId": None,
                "accountName": None,
                "page": page,
                "pageSize": page_size
            }

            if channel_type == "factory":
                payload["brand"] = data.get("brand", None) if data.get("brand") else None
                payload["supplyName"] = data.get("supplyName", None) if data.get("supplyName") else None
            
            # 生成签名
            signature = generate_signature(
                secret_key=API_SECRET_KEY,
                params=payload,
                module="material",
                service=config["service"],
                operator="getRelatedMaterial",
                debug=False
            )
            headers["signature"] = signature
            
            try:
                self.logger.info(f"请求第 {page} 页，每页 {page_size} 条...")
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                if result.get("code") != 200:
                    self.logger.error(f"API返回错误: {result.get('message', '未知错误')}")
                    return {
                        "code": result.get("code", 500),
                        "message": result.get("message", "API调用失败"),
                        "data": {"list": all_records, "totalCount": total_count_from_api}
                    }
                
                response_data = result.get("data", {})
                records = response_data.get("list", [])
                total_count_from_api = response_data.get("totalCount", 0)
                
                if not records:
                    break
                
                all_records.extend(records)
                self.logger.info(f"第 {page} 页获取 {len(records)} 条数据，累计 {len(all_records)} 条")
                
                # 如果获取的记录数少于page_size，说明已经获取完所有数据
                if len(records) < page_size:
                    break
                
                # 如果已经达到totalCount，也停止
                if len(all_records) >= total_count_from_api:
                    break
                
                page += 1
                
            except Exception as e:
                self.logger.error(f"分页获取数据失败 (第{page}页): {e}")
                return {
                    "code": 500,
                    "message": str(e),
                    "data": {"list": all_records, "totalCount": len(all_records)}
                }
        
        self.logger.info(f"分页拉取完成，共 {len(all_records)} 条数据")
        return {
            "code": 200,
            "message": "success",
            "data": {"list": all_records, "totalCount": len(all_records)}
        }

    #  厂商报价查询接口
    def _get_factory_material(self, data: dict) -> dict:
        """厂商报价查询接口"""
        # 根据配置选择使用 Mock 或真实 API
        if USE_MOCK_API:
            url = f"{MOCK_API_BASE_URL}/factory/getFactoryMaterial"
            self.logger.info(f"使用 Mock API: {url}")
            # Mock模式使用旧参数
            payload = {
                "accountId": PRICE_API_ACCOUNT_ID,
                "matchMethod": 1,
                "minPrice": data.get("minPrice", 0),
                "maxPrice": data.get("maxPrice", 0),
                "releaseDepartment": data.get("releaseDepartment", ""),
                "startReleaseDate": data.get("startReleaseDate", ""),
                "endReleaseDate": data.get("endReleaseDate", ""),
                "province": data.get("province", ""),
                "city": data.get("city", ""),
                "materialModelSpec": data.get("materialModelSpec", ""),
                "materialName": data.get("materialName", ""),
                "brand": data.get("brand", ""),
                "supplyName": data.get("supplyName", ""),
                "returnNumber": data.get("returnNumber", 10000),
                "returnTotalCount": 1
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            return response.json()
        else:
            # 真实 API 使用新接口，分页拉取数据
            return self._fetch_price_data_paginated(data, channel_type="factory")

    #  智诚信息价查询接口
    def _get_zhicheng_info_material(self, data: dict) -> dict:
        """智诚信息价查询接口"""
        # 根据配置选择使用 Mock 或真实 API
        if USE_MOCK_API:
            url = f"{MOCK_API_BASE_URL}/zhichengInfo/getZhichengInfoMaterial"
            self.logger.info(f"使用 Mock API: {url}")
            # Mock模式使用旧参数
            payload = {
                "accountId": PRICE_API_ACCOUNT_ID,
                "matchMethod": 1,
                "categoryOneLevelName": "",
                "categoryTwoLevelName": "",
                "categoryThreeLevelName": "",
                "minPrice": data.get("minPrice", 0),
                "maxPrice": data.get("maxPrice", 0),
                "releaseDepartment": data.get("releaseDepartment", ""),
                "startReleaseDate": data.get("startReleaseDate", ""),
                "endReleaseDate": data.get("endReleaseDate", ""),
                "province": data.get("province", ""),
                "city": data.get("city", ""),
                "materialModelSpec": data.get("materialModelSpec", ""),
                "materialName": data.get("materialName", ""),
                "grade": data.get("grade", ""),
                "returnNumber": data.get("returnNumber", 10000),
                "returnTotalCount": 1
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            return response.json()
        else:
            # 真实 API 使用新接口，分页拉取数据
            return self._fetch_price_data_paginated(data, channel_type="zhicheng")

    # RAG知识问答处理入口
    async def process_single_query(self, question: str, num_docs: int = 10) -> Dict[str, Any]:
        # 检查是否正在重建知识库
        if self._is_rebuilding:
            self.logger.warning(f"知识库正在重建中，无法回答知识问答: {question[:50]}...")
            return {
                "success": False,
                "error_code": "KNOWLEDGE_BASE_UPDATING",
                "error_message": "知识库正在更新中，请稍后再试",
                "rebuild_progress": self._rebuild_progress,
                "contexts": [],
                "answer": ""
            }
        
        # 检查 retriever 是否就绪
        if self.retriever is None:
            self.logger.error("Retriever 未初始化，无法进行知识问答")
            return {
                "success": False,
                "error_code": "RETRIEVER_NOT_READY",
                "error_message": "知识库尚未准备就绪，请稍后重试",
                "contexts": [],
                "answer": ""
            }
        
        async with self.semaphore:
            start_time = time.time()
            try:
                retrieval_start = time.time()
                # 使用 run_in_executor 替代 to_thread 以避免潜在的线程问题
                loop = asyncio.get_event_loop()
                retrieval_methods = ["bm25", "emb"]
                retrieval_res = await loop.run_in_executor(
                    None,
                    self.retriever.retrieval,
                    question,
                    retrieval_methods
                )
                retrieval_time = time.time() - retrieval_start
                self.logger.info(f"混合检索完成，methods={retrieval_methods}，返回 {len(retrieval_res)} 个文档")
                if not retrieval_res:
                    raise Exception("检索结果为空")

                rerank_start = time.time()
                # 直接在主线程调用 rerank，避免线程间 CUDA 同步问题
                # Qwen3Reranker.rerank 不是 IO 密集型操作，不需要放到线程池
                if hasattr(self.reranker, '_use_qwen3') and self.reranker._use_qwen3:
                    # Qwen3Reranker 使用 PyTorch，直接在主线程调用以避免 CUDA 线程问题
                    rerank_res = self.reranker.rerank(retrieval_res, question, num_docs)
                else:
                    # BGE Reranker 使用线程池
                    rerank_res = await loop.run_in_executor(None, self.reranker.rerank, retrieval_res, question, num_docs)
                rerank_time = time.time() - rerank_start
                self.logger.info(f"rerank 完成，返回 {len(rerank_res)} 个文档")
                if not rerank_res:
                    raise Exception("重排结果为空")

                llm_start = time.time()
                self.logger.info(f"开始构建上下文，rerank_res 类型: {type(rerank_res)}")
                try:
                    context_str = '\n'.join(item["page_content"] for item in rerank_res)
                    self.logger.info(f"上下文构建成功，长度: {len(context_str)}")
                except Exception as e:
                    self.logger.error(f"构建上下文失败: {e}, rerank_res={rerank_res}")
                    raise
                # answer = await asyncio.to_thread(self.llm.predict, context_str, question)
                # llm_time = time.time() - llm_start
                # if not answer or not answer.strip():
                #     raise Exception("LLM生成答案失败")

                # answer = answer.strip()
                total_time = time.time() - start_time
                self.logger.info(f"process_single_query 即将返回结果，contexts数量: {len(rerank_res)}")
                return {
                    "contexts": rerank_res,
                    "metadata": {
                        "total_time": total_time,
                        "retrieval_time": retrieval_time,
                        "rerank_time": rerank_time,
                        "num_docs": len(rerank_res)
                    },
                    "success": True
                }
                # return {
                #     "answer": answer,
                #     "contexts": rerank_res,
                #     "metadata": {
                #         "total_time": total_time,
                #         "retrieval_time": retrieval_time,
                #         "rerank_time": rerank_time,
                #         "llm_time": llm_time,
                #         "num_docs": len(rerank_res)
                #     },
                #     "success": True
                # }
            except Exception as e:
                self.logger.error(f"查询处理失败: {e}")
                return {
                    "answer": "",
                    "contexts": [],
                    "metadata": {
                        "total_time": time.time() - start_time,
                        "error": str(e)
                    },
                    "success": False,
                    "error_message": str(e)
                }

    # ========== 多轮对话知识问答处理入口（支持查询重写）==========
    async def process_multi_turn_query(
        self, 
        question: str, 
        session, 
        num_docs: int = 10
    ) -> Dict[str, Any]:
        """
        多轮对话知识问答处理入口（支持查询重写）
        
        核心流程：
        1. 查询重写：将依赖上下文的省略/指代问题重写为独立完整的问题
        2. RAG检索：基于重写后的问题进行文档检索
        3. 结果返回：返回检索结果，供后续LLM生成使用
        
        Args:
            question: 用户当前问题
            session: 对话会话对象（包含历史消息）
            num_docs: 返回的文档数量
            
        Returns:
            Dict: 包含检索结果和重写信息的字典
        """
        # 检查知识库状态
        if self._is_rebuilding:
            self.logger.warning(f"知识库正在重建中: {question[:50]}...")
            return {
                "success": False,
                "error_code": "KNOWLEDGE_BASE_UPDATING",
                "error_message": "知识库正在更新中，请稍后再试",
                "contexts": [],
                "rewritten": False
            }
        
        if self.retriever is None:
            self.logger.error("Retriever 未初始化")
            return {
                "success": False,
                "error_code": "RETRIEVER_NOT_READY",
                "error_message": "知识库尚未准备就绪",
                "contexts": [],
                "rewritten": False
            }
        
        if self.query_rewriter is None:
            self.logger.warning("查询重写器未初始化，退回到单轮查询")
            return await self.process_single_query(question, num_docs)
        
        async with self.semaphore:
            start_time = time.time()
            rewrite_result = None
            
            try:
                # ========== 步骤1: 查询重写 ==========
                rewrite_start = time.time()
                
                # 从session中获取消息历史
                messages = session.get_context_for_llm(max_turns=5) if hasattr(session, 'get_context_for_llm') else []
                
                # 执行查询重写
                rewrite_result = await self.query_rewriter.rewrite(
                    current_question=question,
                    messages=messages
                )
                
                rewrite_time = time.time() - rewrite_start
                
                # 使用重写后的问题进行检索
                search_question = rewrite_result.rewritten_question if rewrite_result.was_rewritten else question
                
                if rewrite_result.was_rewritten:
                    self.logger.info(
                        f"查询已重写: '{question[:50]}...' → '{search_question[:50]}...' "
                        f"({rewrite_result.reason}, 耗时: {rewrite_time:.3f}s)"
                    )
                
                # ========== 步骤2: RAG检索（使用重写后的问题）==========
                retrieval_start = time.time()
                loop = asyncio.get_event_loop()
                
                # 使用重写后的问题进行检索
                retrieval_methods = ["bm25", "emb"]
                retrieval_res = await loop.run_in_executor(
                    None,
                    self.retriever.retrieval,
                    search_question,
                    retrieval_methods
                )
                retrieval_time = time.time() - retrieval_start
                
                self.logger.info(
                    f"混合检索完成（基于重写问题），methods={retrieval_methods}，返回 {len(retrieval_res)} 个文档"
                )
                
                if not retrieval_res:
                    self.logger.warning("检索结果为空")
                    # 如果重写后检索为空，尝试用原问题再检索一次
                    if rewrite_result.was_rewritten:
                        self.logger.info("尝试使用原问题重新检索...")
                        retrieval_res = await loop.run_in_executor(
                            None,
                            self.retriever.retrieval,
                            question,
                            retrieval_methods
                        )
                        self.logger.info(
                            f"原问题混合检索返回 {len(retrieval_res)} 个文档，methods={retrieval_methods}"
                        )
                
                # ========== 步骤3: 重排序 ==========
                rerank_start = time.time()
                
                if hasattr(self.reranker, '_use_qwen3') and self.reranker._use_qwen3:
                    rerank_res = self.reranker.rerank(retrieval_res, search_question, num_docs)
                else:
                    rerank_res = await loop.run_in_executor(
                        None, 
                        self.reranker.rerank, 
                        retrieval_res, 
                        search_question, 
                        num_docs
                    )
                rerank_time = time.time() - rerank_start
                
                self.logger.info(f"rerank 完成，返回 {len(rerank_res)} 个文档")
                
                # ========== 步骤4: 构建上下文 ==========
                try:
                    context_str = '\n'.join(item["page_content"] for item in rerank_res)
                except Exception as e:
                    self.logger.error(f"构建上下文失败: {e}")
                    context_str = ""
                
                total_time = time.time() - start_time
                
                # 构建返回结果
                result = {
                    "contexts": rerank_res,
                    "context_text": context_str,
                    "success": True,
                    "metadata": {
                        "total_time": total_time,
                        "rewrite_time": rewrite_time if rewrite_result else 0,
                        "retrieval_time": retrieval_time,
                        "rerank_time": rerank_time,
                        "num_docs": len(rerank_res)
                    }
                }
                
                # 添加重写信息
                if rewrite_result:
                    result["rewrite_info"] = {
                        "original_question": rewrite_result.original_question,
                        "rewritten_question": rewrite_result.rewritten_question,
                        "was_rewritten": rewrite_result.was_rewritten,
                        "reason": rewrite_result.reason
                    }
                
                return result
                
            except Exception as e:
                self.logger.error(f"多轮查询处理失败: {e}", exc_info=True)
                return {
                    "success": False,
                    "error_message": str(e),
                    "contexts": [],
                    "rewritten": False,
                    "metadata": {
                        "total_time": time.time() - start_time,
                        "error": str(e)
                    }
                }

    # 直接价格查询处理入口
    async def process_direct_price_query(self, datatype: str, material_item, question: str = None) -> Dict[str, Any]:
        """处理直接价格查询"""
        start_time = time.time()
        try:
            # 根据 datatype 确定渠道类型
            channel_map = {
                "informaterial": ChannelType.INFORMATION_PRICE,
                "factory": ChannelType.MANUFACTURER_PRICE,
                "zhicheng": ChannelType.ZC_PRICE
            }
            
            channel = channel_map.get(datatype, ChannelType.INFORMATION_PRICE)
            
            # 构造基础查询实体，只包含指定的字段（不包含materialModelSpec、releaseDepartment和unit）
            parsed_entities = {
                "materialName": getattr(material_item, "materialName", ""),
                "province": getattr(material_item, "province", ""),
                "city": getattr(material_item, "city", "")
            }
            
            # 清理空值
            parsed_entities = {k: v for k, v in parsed_entities.items() if v is not None and v != ""}
            
            # 如果提供了问题，使用大模型从问题中提取额外的查询条件
            if question:
                self.logger.info(f"用户提供了问题，将从中提取额外查询条件: {question}")
                
                # 调用大模型提取额外条件
                try:
                    # 直接调用 predict_prompt，传递提示词和查询
                    # 注意：predict_prompt 方法内部会使用 user_question 作为格式化参数
                    # 为JSON响应设置更大的max_tokens值
                    extra_conditions_str = await asyncio.to_thread(
                        self.llm.predict_prompt, 
                        DIRECT_QUERY_EXTRACTION_TEMPLATE,
                        question,
                        None,  # guided_decoding
                        None,  # allowed_tokens
                        1000    # max_tokens，为JSON响应设置更大的值
                    )
                    
                    # 解析大模型返回的JSON
                    if extra_conditions_str:
                        self.logger.info(f"大模型返回的原始响应: {extra_conditions_str}")
                        
                        # 清理可能的代码标记和其他干扰字符
                        clean_output = extra_conditions_str.strip()
                        
                        # 移除可能的```
                        if clean_output.startswith("```"):
                            # 找到最后一个```
                            last_code_block_end = clean_output.rfind("```")
                            if last_code_block_end > 3:
                                clean_output = clean_output[3:last_code_block_end]
                            else:
                                clean_output = clean_output[3:]
                        
                        # 再次清理首尾空白
                        clean_output = clean_output.strip()
                        
                        # 如果以json开头，移除它
                        if clean_output.startswith("json"):
                            clean_output = clean_output[4:].strip()
                        
                        # 移除可能的注释行（以//开头的行）
                        lines = clean_output.split('\n')
                        cleaned_lines = [line for line in lines if not line.strip().startswith('//')]
                        clean_output = '\n'.join(cleaned_lines)
                        
                        self.logger.info(f"清理后的响应: {clean_output}")
                        
                        try:
                            extra_conditions = json.loads(clean_output)
                            
                            # 将额外条件合并到查询实体中，只处理指定的字段
                            allowed_fields = [
                                "materialName", "province", "city"
                            ]
                            
                            for key, value in extra_conditions.items():
                                # 只处理允许的字段，且值不为null且不为空
                                if key in allowed_fields and value is not None and value != "":
                                    parsed_entities[key] = value
                            
                            self.logger.info(f"从问题中提取的额外查询条件: {extra_conditions}")
                        except json.JSONDecodeError as json_error:
                            self.logger.error(f"JSON解析失败: {json_error}")
                            self.logger.error(f"尝试解析的字符串: {clean_output}")
                            
                            # 尝试手动解析简单的键值对
                            try:
                                manual_parsed = self._manual_parse_response(clean_output)
                                if manual_parsed:
                                    # 将手动解析的结果合并到查询实体中
                                    allowed_fields = [
                                        "materialName", "province", "city"
                                    ]
                                    
                                    for key, value in manual_parsed.items():
                                        if key in allowed_fields and value is not None and value != "":
                                            parsed_entities[key] = value
                                    
                                    self.logger.info(f"手动解析的额外查询条件: {manual_parsed}")
                            except Exception as manual_error:
                                self.logger.error(f"手动解析也失败了: {manual_error}")
                    else:
                        self.logger.warning("大模型未能提取到额外的查询条件")
                except json.JSONDecodeError as e:
                    self.logger.error(f"解析大模型返回的JSON失败: {e}")
                except Exception as e:
                    self.logger.error(f"调用大模型提取额外查询条件失败: {e}")
            
            self.logger.info(f"直接价格查询参数: channel={channel.value}, entities={parsed_entities}")
            
            # 调用价格推荐处理逻辑
            result = await self.process_price_recommendation(
                channel.value,
                parsed_entities,
                user_question=question
            )

            answer = result.get("report") or result.get("answer", "")

            # 兼容旧流程：如果底层还没有生成报告，再补做一次自然语言生成
            if result.get("success") and result.get("total_count", 0) > 0 and not answer:
                try:
                    parsed_entities_str = json.dumps(parsed_entities, ensure_ascii=False, indent=2)
                    detail_answer_str = result.get("detail_answer", "")

                    answer = await asyncio.to_thread(
                        self.llm.predict_prompt,
                        self.llm.price_answer_template,
                        {
                            "user_question": question or "",
                            "parsed_entities": parsed_entities_str,
                            "detail_answer": detail_answer_str
                        },
                        False,
                        None,
                        1000
                    )

                    if not answer:
                        answer = "已查询到相关价格数据。"

                except Exception as e:
                    self.logger.error(f"direct 接口调用 LLM 生成回答失败: {e}")
                    answer = "已查询到相关价格数据，但生成自然语言失败。"

            
            total_time = time.time() - start_time
            
            return {
                "answer": answer,
                "parsed_entities": result.get("parsed_entities", {}),
                "detail_answer": result.get("detail_answer", ""),
                "metadata": result.get("metadata", {}),
                "success": result.get("success", False),
                "intent": "price_recommendation",
                "channel": channel.value,
                "total_count": result.get("total_count", 0),
                "price_data": result.get("price_data", []),
                "processing_time": total_time
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"直接价格查询处理失败: {e}")
            return {
                "answer": result.get("answer", ""),
                "parsed_entities": {},
                "detail_answer": "",
                "metadata": {
                    "total_time": total_time,
                    "error": str(e)
                },
                "success": False,
                "intent": "price_recommendation",
                "channel": None,
                "total_count": 0,
                "price_data": [],
                "processing_time": total_time,
                "error_message": str(e)
            }
    
    def _manual_parse_response(self, response_str: str) -> dict:
        """
        手动解析大模型响应的简单键值对格式
        """
        result = {}
        try:
            # 如果响应不完整，尝试修复
            response_str = response_str.strip()
            if not response_str.endswith('}'):
                # 尝试找到最后一个未闭合的引号并修复
                last_quote_pos = response_str.rfind('"')
                if last_quote_pos != -1 and response_str.count('"') % 2 == 1:
                    # 如果有奇数个引号，说明最后一个引号未闭合
                    response_str = response_str[:last_quote_pos+1] + '"\n}'
                else:
                    # 否则简单地添加闭合括号
                    response_str = response_str + '\n}'
            
            # 移除大括号
            if response_str.startswith('{'):
                response_str = response_str[1:]
            if response_str.endswith('}'):
                response_str = response_str[:-1]
            
            content = response_str.strip()
            
            # 按逗号分割
            pairs = content.split(',')
            
            for pair in pairs:
                pair = pair.strip()
                # 按冒号分割键值对
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip().strip('"\'')
                    value = value.strip().strip('"\'')
                    
                    # 处理null值
                    if value.lower() == 'null':
                        result[key] = None
                    else:
                        # 清理值中的特殊字符
                        value = value.replace('\n', '').replace('\r', '').strip()
                        result[key] = value
                        
            return result
        except Exception as e:
            self.logger.warning(f"手动解析响应失败: {e}")
            return {}
