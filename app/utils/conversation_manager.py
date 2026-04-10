"""
统一对话会话管理器
支持知识问答和价格查询的统一会话管理
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import uuid

logger = logging.getLogger("easy_rag_api")


class ConversationStatus(str, Enum):
    """对话状态"""
    ACTIVE = "active"          # 活跃中
    PAUSED = "paused"          # 暂停（等待用户输入）
    COMPLETED = "completed"    # 已完成
    EXPIRED = "expired"        # 已过期


class ConversationType(str, Enum):
    """对话类型"""
    KNOWLEDGE_QA = "knowledge_qa"      # 知识问答
    PRICE_QUERY = "price_query"        # 价格查询


@dataclass
class Message:
    """单条消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """统一对话会话"""
    session_id: str
    conversation_type: ConversationType  # 知识问答 or 价格查询
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    status: ConversationStatus = ConversationStatus.ACTIVE
    
    # 对话历史（完整的用户-助手交互）
    messages: List[Message] = field(default_factory=list)
    
    # 实体状态（价格查询用）
    entities: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    
    # 会话摘要（用于长对话）
    summary: str = ""
    
    # 话题切换检测配置
    TOPIC_SWITCH_SIGNALS = [
        "换个话题", "另外", "还有", "顺便问一下",
        "我想问", "请问", "查一下", "告诉我",
        "是什么", "什么是", "介绍一下", "多少钱"
    ]
    
    # 价格查询话题切换信号
    PRICE_TOPIC_SWITCH_SIGNALS = [
        "换个材料", "另外查", "再查一下", "还有",
        "另外", "其他材料", "换一个", "查别的",
        "还有别的", "别的材料"
    ]
    
    # 配置
    SESSION_TTL: int = 3600  # 1小时过期
    MAX_MESSAGES: int = 20   # 最大保存消息数
    
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return time.time() - self.last_updated > self.SESSION_TTL
    
    def update_timestamp(self):
        """更新最后访问时间"""
        self.last_updated = time.time()
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """添加消息到历史"""
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        
        # 限制消息数量，保留最近的
        if len(self.messages) > self.MAX_MESSAGES:
            self._compress_history()
        
        self.update_timestamp()
    
    def _compress_history(self):
        """压缩历史：生成摘要并删除旧消息"""
        # 保留前2轮和最后3轮，中间生成摘要
        if len(self.messages) > 10:
            old_messages = self.messages[:-6]  # 取较老的消息
            # 生成简单摘要
            user_msgs = [m for m in old_messages if m.role == "user"]
            self.summary = f"...前期对话包含 {len(old_messages)} 条消息，用户提问: {user_msgs[-1].content if user_msgs else '...'}..."
            self.messages = self.messages[-6:]
    
    def get_context_for_llm(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """获取给LLM的上下文（格式化为OpenAI messages格式）"""
        # 返回最近 N 轮对话
        recent = self.messages[-max_turns*2:]
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def detect_topic_switch_by_keywords(self, new_question: str) -> bool:
        """
        基于关键词的话题切换检测（快速检测）
        
        Args:
            new_question: 新的用户问题
        
        Returns:
            bool: 是否检测到话题切换
        """
        if not self.messages or len(self.messages) < 2:
            return False
        
        for signal in self.TOPIC_SWITCH_SIGNALS:
            if signal in new_question:
                logger.info(f"检测到话题切换信号词: {signal}")
                return True
        
        last_user_msg = None
        for msg in reversed(self.messages):
            if msg.role == "user":
                last_user_msg = msg
                break
        
        if not last_user_msg:
            return False
        
        pronouns = ["它", "这个", "那个", "这些", "那些", "其", "此"]
        has_pronoun = any(p in new_question for p in pronouns)
        
        if not has_pronoun:
            len_diff = abs(len(new_question) - len(last_user_msg.content))
            if len_diff > 30:
                if new_question.endswith("？") or new_question.endswith("?"):
                    logger.info(f"检测到可能的话题切换（无代词且长度差异大）: {len_diff}")
                    return True
        
        return False
    
    async def detect_topic_switch_by_llm(
        self, 
        new_question: str, 
        llm_predictor
    ) -> bool:
        """
        基于LLM的话题切换检测（兜底检测）
        
        Args:
            new_question: 新的用户问题
            llm_predictor: LLM预测器实例
        
        Returns:
            bool: 是否检测到话题切换
        """
        if not self.messages or len(self.messages) < 2:
            return False
        
        recent_messages = self.messages[-4:]
        history_text = "\n".join([
            f"{'用户' if m.role == 'user' else '助手'}: {m.content[:100]}"
            for m in recent_messages
        ])
        
        prompt = f"""请判断新问题是否与对话历史属于同一话题。

对话历史：
{history_text}

新问题：{new_question}

判断标准：
1. 如果新问题是继续追问、深入探讨历史话题，回答"NO"（同一话题）
2. 如果新问题是全新的、不相关的问题，回答"YES"（话题切换）
3. 如果不确定，倾向于回答"NO"（保守策略）

请只回答 YES 或 NO，不要解释。"""

        try:
            result = await llm_predictor.predict_prompt(
                prompt,
                {},
                guided_decoding=False,
                max_tokens=10
            )
            
            is_switch = result.strip().upper() == "YES"
            logger.info(f"LLM话题检测结果: {result.strip()} -> {is_switch}")
            return is_switch
            
        except Exception as e:
            logger.error(f"LLM话题检测失败: {e}")
            return False

    def detect_price_topic_switch_by_material(
        self, 
        new_question: str,
        extracted_entities: Dict[str, Any]
    ) -> bool:
        """
        基于材料名称变化的价格查询话题切换检测
        
        策略：
        1. 如果新问题包含明确的材料名称切换信号词 → 切换
        2. 如果历史中有材料A，新问题提到材料B（不同名称）→ 切换
        3. 如果是补充信息（规格、省份等）→ 不切换
        
        Args:
            new_question: 新的用户问题
            extracted_entities: 新提取的实体
        
        Returns:
            bool: 是否检测到话题切换
        """
        if not self.messages or len(self.messages) < 2:
            return False
        
        # 1. 检测明确的切换信号词
        for signal in self.PRICE_TOPIC_SWITCH_SIGNALS:
            if signal in new_question:
                logger.info(f"检测到价格查询话题切换信号词: {signal}")
                return True
        
        # 2. 获取历史材料名称
        historical_materials = set()
        for msg in self.messages:
            if msg.role == "user":
                # 从metadata中获取历史实体（如果有）
                material = msg.metadata.get("materialName") if msg.metadata else None
                if material:
                    historical_materials.add(material.lower())
        
        # 如果没有历史材料，无法判断是否切换
        if not historical_materials:
            return False
        
        # 3. 获取当前材料名称
        current_material = extracted_entities.get("materialName", "")
        if not current_material:
            # 新请求没有提取到材料名称，可能是补充其他信息
            return False
        
        current_material_lower = current_material.lower()
        
        # 4. 判断材料是否变化
        # 如果当前材料与所有历史材料都不匹配，认为是话题切换
        for hist_material in historical_materials:
            # 完全匹配或包含关系视为同一材料
            if (current_material_lower == hist_material or
                current_material_lower in hist_material or
                hist_material in current_material_lower):
                logger.debug(f"材料 '{current_material}' 与历史材料 '{hist_material}' 匹配，不切换")
                return False
        
        # 材料名称不同，检测到话题切换
        logger.info(f"检测到材料变化: {historical_materials} -> {current_material}，触发话题切换")
        return True

    async def detect_price_topic_switch_by_llm(
        self, 
        new_question: str,
        extracted_entities: Dict[str, Any],
        llm_predictor
    ) -> bool:
        """
        基于LLM的价格查询话题切换检测（兜底）
        
        专门用于判断是否是新材料的查询
        
        Args:
            new_question: 新的用户问题
            extracted_entities: 新提取的实体
            llm_predictor: LLM预测器实例
        
        Returns:
            bool: 是否检测到话题切换
        """
        if not self.messages or len(self.messages) < 2:
            return False
        
        # 获取历史材料名称
        historical_materials = []
        for msg in self.messages:
            if msg.role == "user" and msg.metadata:
                material = msg.metadata.get("materialName")
                if material and material not in historical_materials:
                    historical_materials.append(material)
        
        if not historical_materials:
            return False
        
        current_material = extracted_entities.get("materialName", "未知材料")
        
        # 构建历史对话文本
        recent_messages = self.messages[-4:]
        history_text = "\n".join([
            f"{'用户' if m.role == 'user' else '助手'}: {m.content[:80]}"
            for m in recent_messages
        ])
        
        prompt = f"""判断用户是否在查询新的材料价格。

对话历史：
{history_text}

历史查询材料：{', '.join(historical_materials)}
新问题：{new_question}
新提到的材料：{current_material}

判断标准：
1. 如果用户是在补充/修改历史材料的查询条件（规格、省份等），回答"NO"
2. 如果用户是在询问一个全新的、不同的材料价格，回答"YES"
3. 如果不确定，倾向于回答"NO"

请只回答 YES 或 NO，不要解释。"""

        try:
            result = await llm_predictor.predict_prompt(
                prompt,
                {},
                guided_decoding=False,
                max_tokens=10
            )
            
            is_switch = result.strip().upper() == "YES"
            logger.info(f"LLM价格话题检测结果: {result.strip()} -> {is_switch}")
            return is_switch
            
        except Exception as e:
            logger.error(f"LLM价格话题检测失败: {e}")
            return False
    
    def update_entities(self, new_entities: Dict[str, Any]):
        """更新实体（合并而非替换）"""
        for key, value in new_entities.items():
            if value is not None and value != "":
                self.entities[key] = value
        self.update_timestamp()
    
    def set_missing_fields(self, fields: List[str]):
        """设置缺失字段"""
        self.missing_fields = fields
        self.update_timestamp()
    
    def to_dict(self) -> Dict:
        """转换为字典（用于API响应）"""
        return {
            "session_id": self.session_id,
            "conversation_type": self.conversation_type.value,
            "status": self.status.value,
            "entities": self.entities,
            "missing_fields": self.missing_fields,
            "message_count": len(self.messages),
            "expires_in": self.SESSION_TTL - (time.time() - self.last_updated),
            "summary": self.summary
        }


class ConversationManager:
    """对话管理器 - 统一管理知识问答和价格查询的会话"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.logger = logging.getLogger("easy_rag_api")
        
        self.TOPIC_SWITCH_THRESHOLD = {
            "enable_keyword_detection": True,
            "enable_llm_detection": True,
            "llm_detection_min_messages": 2,
            "llm_detection_probability": 0.3
        }
        
        self.stats = {
            "total_checks": 0,
            "keyword_switches": 0,
            "llm_switches": 0,
            "llm_calls": 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取话题切换检测统计信息"""
        return {
            **self.stats,
            "keyword_switch_rate": (
                self.stats["keyword_switches"] / self.stats["total_checks"] 
                if self.stats["total_checks"] > 0 else 0
            ),
            "llm_switch_rate": (
                self.stats["llm_switches"] / self.stats["llm_calls"] 
                if self.stats["llm_calls"] > 0 else 0
            )
        }
    
    async def detect_topic_switch(
        self,
        session: ConversationSession,
        new_question: str,
        llm_predictor=None
    ) -> bool:
        """
        混合话题切换检测（关键词 + LLM兜底）
        
        Args:
            session: 会话对象
            new_question: 新的用户问题
            llm_predictor: LLM预测器实例（可选）
        
        Returns:
            bool: 是否检测到话题切换
        """
        self.stats["total_checks"] += 1
        
        if not self.TOPIC_SWITCH_THRESHOLD["enable_keyword_detection"]:
            return False
        
        if session.detect_topic_switch_by_keywords(new_question):
            self.logger.info("✓ 关键词检测到话题切换")
            self.stats["keyword_switches"] += 1
            return True
        
        if (self.TOPIC_SWITCH_THRESHOLD["enable_llm_detection"] and 
            llm_predictor and 
            len(session.messages) >= self.TOPIC_SWITCH_THRESHOLD["llm_detection_min_messages"]):
            
            import random
            if random.random() < self.TOPIC_SWITCH_THRESHOLD["llm_detection_probability"]:
                self.logger.info("启用LLM兜底检测...")
                self.stats["llm_calls"] += 1
                is_switch = await session.detect_topic_switch_by_llm(new_question, llm_predictor)
                if is_switch:
                    self.logger.info("✓ LLM检测到话题切换")
                    self.stats["llm_switches"] += 1
                return is_switch
        
        return False
    
    def create_session(self, conversation_type: ConversationType) -> ConversationSession:
        """创建新会话"""
        session_id = str(uuid.uuid4())[:12]
        session = ConversationSession(
            session_id=session_id,
            conversation_type=conversation_type
        )
        self.sessions[session_id] = session
        self.logger.info(f"创建新会话: {session_id} (类型: {conversation_type.value})")
        return session
    
    def get_session(self, session_id: Optional[str]) -> Optional[ConversationSession]:
        """获取会话"""
        if not session_id:
            return None
        
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        if session.is_expired():
            self.logger.info(f"会话 {session_id} 已过期，清理中...")
            del self.sessions[session_id]
            return None
        
        session.update_timestamp()
        return session
    
    async def get_or_create_session(
        self, 
        session_id: Optional[str], 
        conversation_type: ConversationType,
        force_new: bool = False,
        new_question: str = None,
        llm_predictor = None,
        extracted_entities: Dict[str, Any] = None
    ) -> ConversationSession:
        """
        获取或创建会话（支持话题切换检测）
        
        Args:
            session_id: 会话ID（可选）
            conversation_type: 会话类型（知识问答/价格查询）
            force_new: 强制创建新会话，清空历史对话记录
            new_question: 新的用户问题（用于话题切换检测）
            llm_predictor: LLM预测器实例（用于LLM话题检测）
            extracted_entities: 新提取的实体（价格查询话题切换检测用）
        
        Returns:
            ConversationSession: 会话对象
        """
        if force_new:
            self.logger.info(f"用户请求强制创建新会话（force_new=True）")
            return self.create_session(conversation_type)
        
        session = self.get_session(session_id)
        if session:
            if session.conversation_type != conversation_type:
                self.logger.warning(
                    f"会话 {session_id} 类型为 {session.conversation_type.value}，"
                    f"但请求类型为 {conversation_type.value}，将创建新会话"
                )
                return self.create_session(conversation_type)
            
            if conversation_type == ConversationType.KNOWLEDGE_QA and new_question:
                is_topic_switch = await self.detect_topic_switch(
                    session, new_question, llm_predictor
                )
                if is_topic_switch:
                    self.logger.info(f"检测到话题切换，创建新会话")
                    return self.create_session(conversation_type)
            
            # 价格查询话题切换检测
            if conversation_type == ConversationType.PRICE_QUERY and new_question:
                is_price_topic_switch = await self.detect_price_topic_switch(
                    session, new_question, extracted_entities or {}, llm_predictor
                )
                if is_price_topic_switch:
                    self.logger.info(f"检测到价格查询话题切换（材料变化），创建新会话")
                    return self.create_session(conversation_type)
            
            return session
        return self.create_session(conversation_type)
    
    async def detect_price_topic_switch(
        self,
        session: ConversationSession,
        new_question: str,
        extracted_entities: Dict[str, Any],
        llm_predictor=None
    ) -> bool:
        """
        检测价格查询是否需要话题切换（材料变化）
        
        Args:
            session: 当前会话
            new_question: 新的用户问题
            extracted_entities: 提取的实体
            llm_predictor: LLM预测器（可选）
        
        Returns:
            bool: 是否需要创建新会话
        """
        self.stats["total_checks"] += 1
        
        # 1. 基于材料名称变化的检测
        if session.detect_price_topic_switch_by_material(new_question, extracted_entities):
            self.stats["keyword_switches"] += 1
            return True
        
        # 2. LLM兜底检测（概率触发，节省成本）
        if (llm_predictor and 
            len(session.messages) >= self.TOPIC_SWITCH_THRESHOLD["llm_detection_min_messages"]):
            
            import random
            if random.random() < self.TOPIC_SWITCH_THRESHOLD["llm_detection_probability"]:
                self.logger.info("启用LLM兜底检测（价格查询）...")
                self.stats["llm_calls"] += 1
                is_switch = await session.detect_price_topic_switch_by_llm(
                    new_question, extracted_entities, llm_predictor
                )
                if is_switch:
                    self.stats["llm_switches"] += 1
                return is_switch
        
        return False
    
    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"会话 {session_id} 已清除")
    
    def clean_expired_sessions(self):
        """清理过期会话"""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired()
        ]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            self.logger.info(f"清理了 {len(expired)} 个过期会话")


# 全局单例
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """获取对话管理器单例"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
