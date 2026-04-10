# RAG 系统对话功能改进方案

> 文档版本: v1.0  
> 创建时间: 2026-04-09  
> 适用项目: easy-rag

---

## 目录

1. [现状分析](#一现状分析)
2. [核心问题](#二核心问题)
3. [改进方案](#三改进方案)
4. [实施路线图](#四实施路线图)
5. [附录：代码示例](#五附录代码示例)

---

## 一、现状分析

### 1.1 系统架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                         前端 (Streamlit)                      │
├─────────────────────────────────────────────────────────────┤
│  web_app.py          │  web_app_dialogue.py                  │
│  (传统模式)           │  (对话式查询)                          │
├─────────────────────────────────────────────────────────────┤
│                         API 层 (FastAPI)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ /query/stream│  │ /query/price│  │ /query/dialogue     │  │
│  │ 知识问答     │  │ 价格查询     │  │ 对话式价格查询       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      服务层 (RAGService)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 全局messages │  │ 全局messages │  │ DialogueManager     │  │
│  │ (无会话隔离) │  │ (无会话隔离) │  │ (仅实体收集)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 功能现状对比

| 功能 | 知识问答 | 价格查询(传统) | 价格查询(对话式) |
|------|---------|--------------|----------------|
| **多轮对话** | ⚠️ 有限支持 | ⚠️ 有限支持 | ✅ 支持 |
| **会话隔离** | ❌ 不支持 | ❌ 不支持 | ✅ 支持 |
| **意图跟踪** | ❌ 不支持 | ❌ 不支持 | ⚠️ 部分支持 |
| **对话记忆** | ❌ 不支持 | ❌ 不支持 | ⚠️ 有限支持 |
| **跨功能切换** | ❌ 不支持 | ❌ 不支持 | ❌ 不支持 |

---

## 二、核心问题

### 2.1 🔴 严重问题（P0）

#### 问题1：全局共享的对话历史

**位置**: `app/api/endpoints.py` 第 85 行

```python
# ✅ 全局对话历史（存储最近 6 条消息：user/assistant）
messages: list[dict] = []
```

**影响**:
- 所有用户共享同一对话历史，存在数据安全隐患
- 并发场景下对话内容可能相互污染
- 无法支持多用户同时使用

#### 问题2：无会话管理机制

**影响**:
- 无法区分不同用户的对话上下文
- 页面刷新后对话历史丢失
- 无法跨请求保持对话状态

### 2.2 🟡 中等问题（P1）

#### 问题3：意图漂移

每轮对话都重新进行意图识别，没有锁定机制：

```
用户: 查一下钢筋的价格
AI: 请问是哪个省份？
用户: 广东省深圳市的
AI: [识别为新意图，而非继续价格查询]
```

#### 问题4：对话与LLM分离

`DialogueManager` 只管理实体收集，不将对话历史传递给LLM：

```python
# dialogue_manager.py 中只保存简单的历史记录
history: List[Dict[str, str]] = field(default_factory=list)

# 但 LLM 调用时并不使用这些历史
async for chunk in rag_service.llm.stream_price_predict(
    parsed_entities_str, detail_answer, request.question, messages  # 使用的是全局messages
):
```

### 2.3 🟢 优化项（P2）

#### 问题5：无法跨功能对话

知识问答和价格查询之间无法自然切换：

```
用户: 什么是工程造价？                    → 知识问答
AI: 工程造价是指...
用户: 那钢筋的价格是多少？               → 价格查询
AI: [查询价格]
用户: 刚才说的工程造价包括哪些费用？      → 无法关联到第一轮对话
```

#### 问题6：缺乏长期记忆

系统无法记住：
- 用户的常用查询偏好
- 历史查询的上下文
- 用户确认过的信息

---

## 三、改进方案

### 3.1 方案一：统一会话管理系统（推荐）

#### 3.1.1 新增 ConversationManager 模块

创建文件: `app/utils/conversation_manager.py`

```python
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


class IntentType(str, Enum):
    """意图类型"""
    KNOWLEDGE_QA = "knowledge_qa"
    PRICE_RECOMMENDATION = "price_recommendation"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class Message:
    """单条消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    intent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """统一对话会话"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    status: ConversationStatus = ConversationStatus.ACTIVE
    
    # 对话历史（完整的用户-助手交互）
    messages: List[Message] = field(default_factory=list)
    
    # 意图状态
    current_intent: Optional[str] = None
    intent_confidence: float = 0.0
    intent_locked: bool = False  # 是否锁定意图
    
    # 实体状态（价格查询用）
    entities: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    
    # 会话摘要（用于长对话）
    summary: str = ""
    
    # 用户偏好
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # 配置
    SESSION_TTL: int = 3600  # 1小时过期
    MAX_MESSAGES: int = 20   # 最大保存消息数
    
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return time.time() - self.last_updated > self.SESSION_TTL
    
    def update_timestamp(self):
        """更新最后访问时间"""
        self.last_updated = time.time()
    
    def add_message(self, role: str, content: str, intent: str = None, metadata: Dict = None):
        """添加消息到历史"""
        self.messages.append(Message(
            role=role,
            content=content,
            intent=intent,
            metadata=metadata or {}
        ))
        
        # 限制消息数量，保留最近的
        if len(self.messages) > self.MAX_MESSAGES:
            # 保留系统消息和最近的消息
            self._compress_history()
        
        self.update_timestamp()
    
    def _compress_history(self):
        """压缩历史：生成摘要并删除旧消息"""
        # 保留前2轮和最后3轮，中间生成摘要
        if len(self.messages) > 10:
            old_messages = self.messages[:-6]  # 取较老的消息
            # TODO: 使用LLM生成摘要
            self.summary = f"...前期对话包含 {len(old_messages)} 条消息..."
            self.messages = self.messages[-6:]
    
    def get_context_for_llm(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """获取给LLM的上下文"""
        # 返回最近 N 轮对话
        recent = self.messages[-max_turns*2:]
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def update_intent(self, intent: str, confidence: float, force: bool = False):
        """更新意图（带锁定机制）"""
        # 如果已锁定且不强制，则不改变
        if self.intent_locked and not force:
            if self.current_intent and confidence < 0.9:
                logger.info(f"意图已锁定为 {self.current_intent}，忽略新意图 {intent}")
                return
        
        self.current_intent = intent
        self.intent_confidence = confidence
        
        # 高置信度时锁定意图
        if confidence > 0.8:
            self.intent_locked = True
            logger.info(f"意图已锁定: {intent} (置信度: {confidence})")
    
    def unlock_intent(self):
        """解锁意图（用户明确切换话题时调用）"""
        self.intent_locked = False
        self.intent_confidence = 0.0
        logger.info("意图已解锁")
    
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
            "status": self.status.value,
            "current_intent": self.current_intent,
            "intent_locked": self.intent_locked,
            "entities": self.entities,
            "missing_fields": self.missing_fields,
            "message_count": len(self.messages),
            "expires_in": self.SESSION_TTL - (time.time() - self.last_updated),
            "summary": self.summary
        }


class ConversationManager:
    """对话管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.logger = logging.getLogger("easy_rag_api")
    
    def create_session(self) -> ConversationSession:
        """创建新会话"""
        session_id = str(uuid.uuid4())[:12]
        session = ConversationSession(session_id=session_id)
        self.sessions[session_id] = session
        self.logger.info(f"创建新会话: {session_id}")
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
    
    def get_or_create_session(self, session_id: Optional[str]) -> ConversationSession:
        """获取或创建会话"""
        session = self.get_session(session_id)
        if session:
            return session
        return self.create_session()
    
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
```

#### 3.1.2 改造 API 层

修改 `app/api/endpoints.py`:

```python
# ===== 移除全局变量 =====
# ❌ 删除以下代码
# messages: list[dict] = []

# ✅ 引入会话管理器
from app.utils.conversation_manager import get_conversation_manager, IntentType

conversation_manager = get_conversation_manager()

# ===== 改造请求模型 =====
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    num_docs: int = Field(10, ge=1, le=20)

class PriceRequest(BaseModel):
    question: str
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")


# ===== 改造知识问答接口 =====
@router.post("/api/v1/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    # 获取或创建会话
    session = conversation_manager.get_or_create_session(request.session_id)
    
    async def answer_generator() -> AsyncGenerator[str, None]:
        try:
            # 添加用户消息到会话
            session.add_message("user", request.question)
            
            # 检索
            result = await rag_service.process_single_query(
                question=request.question,
                num_docs=request.num_docs
            )
            contexts = result.get("contexts", [])
            
            # 构建上下文
            context_text = '\n'.join([
                c.get('page_content', str(c)) for c in contexts
            ])
            
            # 获取会话历史作为上下文
            messages = session.get_context_for_llm(max_turns=5)
            
            full_text = ""
            async for chunk in rag_service.llm.stream_predict(
                context_text, request.question, messages
            ):
                full_text += chunk
                yield chunk + "\n"
            
            yield "\n[END]\n"
            
            # 保存助手回复到会话
            session.add_message("assistant", full_text, intent="knowledge_qa")
            
            # 返回元数据（包含 session_id）
            meta = {
                "text": full_text,
                "contexts": contexts,
                "session_id": session.session_id,
                "intent": "knowledge_qa"
            }
            yield json.dumps(meta, ensure_ascii=False) + "\n"
            
        except Exception as e:
            logging.error(f"流式查询异常: {e}", exc_info=True)
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
    
    return StreamingResponse(answer_generator(), media_type="text/plain")


# ===== 改造价格查询接口 =====
@router.post("/api/v1/query/price")
async def query_price(request: PriceRequest) -> StreamingResponse:
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    # 获取或创建会话
    session = conversation_manager.get_or_create_session(request.session_id)
    
    async def price_answer_generator() -> AsyncGenerator[str, None]:
        try:
            # 添加用户消息
            session.add_message("user", request.question)
            
            # 意图识别（带上下文）
            current_intent = session.current_intent
            if not current_intent or not session.intent_locked:
                intent = await rag_service.classify_intent(request.question)
                session.update_intent(intent, confidence=0.9)
            else:
                intent = current_intent
            
            # 渠道识别
            parsed_entities = await rag_service.extract_entities(request.question)
            channel_result, inference_detail = await rag_service.identify_channel(
                request.question, parsed_entities
            )
            
            # 合并实体（保留历史实体）
            session.update_entities(parsed_entities)
            
            # 检查缺失字段
            required_fields = ["materialName", "province", "city"]
            missing = [f for f in required_fields if not session.entities.get(f)]
            
            if missing:
                # 需要继续收集实体
                session.set_missing_fields(missing)
                next_field = missing[0]
                
                response_text = f"需要补充以下信息：{next_field}"
                session.add_message("assistant", response_text, intent="price_recommendation")
                
                yield json.dumps({
                    "session_id": session.session_id,
                    "status": "collecting",
                    "intent": "price_recommendation",
                    "entities": session.entities,
                    "missing_fields": missing,
                    "response_text": response_text,
                    "can_query": False
                }, ensure_ascii=False) + "\n"
                return
            
            # 执行价格查询
            result = await rag_service.process_price_recommendation(
                channel_result.value, 
                session.entities
            )
            
            # 流式输出
            full_text = ""
            if result.get("success"):
                messages = session.get_context_for_llm(max_turns=3)
                async for chunk in rag_service.llm.stream_price_predict(
                    json.dumps(session.entities, ensure_ascii=False),
                    result.get("detail_answer", ""),
                    request.question,
                    messages
                ):
                    full_text += chunk
                    yield chunk + "\n"
                
                yield "\n[END]\n"
                session.add_message("assistant", full_text, intent="price_recommendation")
            
            # 返回元数据
            meta = {
                "text": full_text,
                "session_id": session.session_id,
                "intent": "price_recommendation",
                "channel": channel_result.value,
                "entities": session.entities,
                "metadata": result.get("metadata", {}),
                "success": result.get("success", False)
            }
            yield json.dumps(meta, ensure_ascii=False) + "\n"
            
        except Exception as e:
            yield json.dumps({
                "error": str(e),
                "session_id": session.session_id
            }, ensure_ascii=False) + "\n"
    
    return StreamingResponse(price_answer_generator(), media_type="text/plain")
```

#### 3.1.3 新增意图上下文识别

在 `app/services/rag_service.py` 中添加:

```python
async def classify_intent_with_context(
    self, 
    question: str, 
    session: ConversationSession
) -> str:
    """
    带上下文的意图识别
    
    策略：
    1. 如果已有锁定意图，优先保持
    2. 检测意图切换信号词
    3. 结合对话历史进行识别
    """
    # 意图切换信号词
    switch_signals = [
        "换个话题", "我想问", "另外", "还有", "顺便问一下",
        "查一下价格", "多少钱", "什么价格",
        "是什么", "什么是", "介绍一下"
    ]
    
    # 检测是否有切换信号
    has_switch_signal = any(signal in question for signal in switch_signals)
    
    # 如果已有锁定意图且没有切换信号，保持原意图
    if session.intent_locked and not has_switch_signal:
        self.logger.info(f"保持锁定意图: {session.current_intent}")
        return session.current_intent
    
    # 检测价格相关关键词
    price_keywords = ["价格", "多少钱", "报价", "成本", "费用", "元", "块"]
    is_price_query = any(kw in question for kw in price_keywords)
    
    # 检测知识问答关键词
    knowledge_keywords = ["什么是", "是什么", "介绍一下", "说明", "概念", "定义"]
    is_knowledge_query = any(kw in question for kw in knowledge_keywords)
    
    # 结合历史上下文判断
    last_intent = session.current_intent
    history = session.get_context_for_llm(max_turns=2)
    
    # 如果上一轮是价格查询，且当前没有明确的知识问答信号，保持价格查询
    if last_intent == "price_recommendation" and not is_knowledge_query:
        if is_price_query or "补充" in question or "改" in question:
            return "price_recommendation"
    
    # 否则调用LLM进行标准意图识别
    intent = await self.classify_intent(question)
    
    # 更新会话意图
    session.update_intent(intent, confidence=0.9)
    
    return intent
```

---

### 3.2 方案二：分层记忆架构

```
┌─────────────────────────────────────────┐
│           工作记忆 (Working Memory)        │  ← 当前对话轮次，给LLM用
│  最近 5-10 轮对话                        │
├─────────────────────────────────────────┤
│           短期记忆 (Short-term Memory)     │  ← 当前会话，实体、意图等
│  会话级别：实体收集状态、锁定意图          │
│  存储：内存中，1小时过期                  │
├─────────────────────────────────────────┤
│           长期记忆 (Long-term Memory)      │  ← 跨会话，用户偏好
│  用户级别：常用查询、偏好设置、历史摘要    │
│  存储：数据库/文件，永久保存              │
└─────────────────────────────────────────┘
```

#### 3.2.1 长期记忆存储

创建文件: `app/utils/long_term_memory.py`

```python
"""长期记忆管理 - 持久化存储用户偏好和历史"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import hashlib

class LongTermMemory:
    """长期记忆管理器"""
    
    def __init__(self, storage_dir: str = "./memory"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_user_file(self, user_id: str) -> str:
        """获取用户记忆文件路径"""
        # 使用哈希避免特殊字符
        hashed = hashlib.md5(user_id.encode()).hexdigest()[:16]
        return os.path.join(self.storage_dir, f"{hashed}.json")
    
    def load_user_memory(self, user_id: str) -> Dict:
        """加载用户记忆"""
        file_path = self._get_user_file(user_id)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "preferences": {},
            "common_queries": [],
            "conversation_summaries": []
        }
    
    def save_user_memory(self, user_id: str, memory: Dict):
        """保存用户记忆"""
        file_path = self._get_user_file(user_id)
        memory["updated_at"] = datetime.now().isoformat()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    
    def add_conversation_summary(self, user_id: str, summary: str, entities: Dict):
        """添加对话摘要"""
        memory = self.load_user_memory(user_id)
        memory["conversation_summaries"].append({
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "entities": entities
        })
        # 只保留最近10条
        memory["conversation_summaries"] = memory["conversation_summaries"][-10:]
        self.save_user_memory(user_id, memory)
    
    def update_preferences(self, user_id: str, preferences: Dict):
        """更新用户偏好"""
        memory = self.load_user_memory(user_id)
        memory["preferences"].update(preferences)
        self.save_user_memory(user_id, memory)
```

---

### 3.3 方案三：混合对话流程

支持在知识问答和价格查询之间自然切换：

```
用户: 什么是工程造价？                    → 意图: knowledge_qa
AI: 工程造价是指...
用户: 那钢筋的价格是多少？               → 意图: price_recommendation
AI: 请问是哪个省份？
用户: 广东省
AI: [查询并返回价格]
用户: 刚才说的工程造价包括哪些费用？      → 意图: knowledge_qa（根据上下文推断）
AI: 工程造价包括人工费、材料费...（关联到第一轮对话）
```

实现要点：

1. **上下文感知意图识别**: 结合历史对话判断真实意图
2. **对话摘要**: 长对话时生成摘要，保留关键信息
3. **实体继承**: 同一会话中，实体信息可跨意图复用

---

## 四、实施路线图

### 4.1 阶段划分

```
Phase 1: 基础改造（2周）
├── 移除全局 messages 变量
├── 引入 ConversationManager
├── API 层改造支持 session_id
└── 前端适配传递 session_id

Phase 2: 意图管理（1周）
├── 实现意图锁定机制
├── 上下文感知意图识别
└── 意图切换检测

Phase 3: 混合对话（1周）
├── 统一对话入口
├── 跨功能上下文传递
└── 对话摘要生成

Phase 4: 高级功能（2周）
├── 长期记忆存储
├── 用户偏好学习
├── 会话恢复机制
└── 多用户并发优化
```

### 4.2 优先级矩阵

| 优先级 | 改进项 | 工作量 | 影响 | 风险 |
|-------|--------|--------|------|------|
| P0 | 移除全局 messages | 中等 | 高 | 低 |
| P0 | API 层会话隔离 | 中等 | 高 | 低 |
| P1 | 意图锁定机制 | 中等 | 高 | 中 |
| P1 | 对话历史传递 | 中等 | 高 | 低 |
| P2 | 对话摘要生成 | 较大 | 中 | 中 |
| P2 | 长期记忆存储 | 较大 | 中 | 低 |
| P3 | 混合对话流程 | 较大 | 中 | 高 |
| P3 | 用户偏好学习 | 大 | 低 | 高 |

### 4.3 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 会话数据丢失 | 中 | 高 | 增加持久化备份 |
| 意图锁定过强 | 中 | 中 | 提供手动切换入口 |
| 内存占用增加 | 中 | 中 | 设置会话过期时间 |
| 并发性能下降 | 低 | 高 | 使用连接池和缓存 |

---

## 五、附录：代码示例

### 5.1 前端适配示例

```python
# front/server/web_app.py

class ConversationState:
    """前端对话状态管理"""
    
    def __init__(self):
        self.session_id: Optional[str] = None
        self.current_intent: Optional[str] = None
        self.dialogue_history: List[Dict] = []
        self.entities: Dict = {}
        self.is_collecting: bool = False
        
    def reset(self):
        """重置对话状态"""
        self.__init__()

# 在 Streamlit 中初始化
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationState()

# 发送请求时携带 session_id
def send_query(question: str):
    payload = {
        "question": question,
        "session_id": st.session_state.conversation.session_id
    }
    
    response = requests.post(
        f"{api.base_url}/api/v1/query/stream",
        json=payload,
        stream=True
    )
    
    # 从响应中提取 session_id
    for chunk in response.iter_lines():
        if chunk.startswith(b'{'):
            meta = json.loads(chunk)
            if "session_id" in meta:
                st.session_state.conversation.session_id = meta["session_id"]
```

### 5.2 数据库查询复用示例

```python
# 在同一会话中复用已查询的数据

class QueryCache:
    """查询结果缓存"""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
    
    def cache_result(self, session_id: str, query_hash: str, result: Any):
        """缓存查询结果"""
        key = f"{session_id}:{query_hash}"
        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def get_cached_result(self, session_id: str, query_hash: str) -> Optional[Any]:
        """获取缓存结果"""
        key = f"{session_id}:{query_hash}"
        cached = self.cache.get(key)
        if cached and time.time() - cached["timestamp"] < 300:  # 5分钟过期
            return cached["result"]
        return None
```

### 5.3 会话状态机

```python
class DialogueStateMachine:
    """对话状态机"""
    
    STATES = {
        "idle": ["collecting", "querying"],
        "collecting": ["ready", "idle"],
        "ready": ["querying", "collecting", "idle"],
        "querying": ["completed", "error"],
        "completed": ["idle", "collecting"],
        "error": ["idle", "retrying"]
    }
    
    def __init__(self, session: ConversationSession):
        self.session = session
        self.state = "idle"
    
    def transition(self, new_state: str):
        """状态转换"""
        if new_state in self.STATES.get(self.state, []):
            old_state = self.state
            self.state = new_state
            self.session.status = new_state
            logger.info(f"状态转换: {old_state} -> {new_state}")
            return True
        else:
            logger.warning(f"非法状态转换: {self.state} -> {new_state}")
            return False
```

---

## 六、总结

### 核心改进点

1. **会话隔离**: 从全局共享改为每个会话独立管理
2. **意图锁定**: 防止多轮对话中的意图漂移
3. **统一入口**: 知识问答和价格查询使用统一的管理机制
4. **分层记忆**: 工作记忆、短期记忆、长期记忆三层架构

### 预期收益

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 多轮对话准确率 | ~60% | ~90% |
| 会话数据隔离 | ❌ | ✅ |
| 跨功能对话 | ❌ | ✅ |
| 用户体验 | 一般 | 优秀 |

---

**文档维护**: 请根据实际开发进度更新此文档  
**相关文件**: 
- `app/utils/dialogue_manager.py`
- `app/utils/conversation_manager.py` (新增)
- `app/utils/long_term_memory.py` (新增)
- `app/api/endpoints.py`
- `app/services/rag_service.py`
