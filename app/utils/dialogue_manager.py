"""
对话状态管理器
用于管理多轮对话中的实体补全
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger("easy_rag_api")


class DialogueStatus(str, Enum):
    """对话状态"""
    COLLECTING = "collecting"      # 收集中
    READY = "ready"                # 准备就绪，可执行查询
    COMPLETED = "completed"        # 已完成
    EXPIRED = "expired"            # 已过期


@dataclass
class DialogueSession:
    """对话会话"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    status: DialogueStatus = DialogueStatus.COLLECTING
    
    # 已收集的实体
    entities: Dict[str, Any] = field(default_factory=dict)
    
    # 缺失的关键字段（按优先级排序）
    missing_fields: List[Tuple[str, str, str]] = field(default_factory=list)
    
    # 对话历史
    history: List[Dict[str, str]] = field(default_factory=list)
    
    # 推断的渠道信息
    channel: Optional[str] = None
    channel_confidence: float = 0.0
    
    # 最大会话有效期（秒）
    SESSION_TTL: int = 600  # 10分钟
    
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return time.time() - self.last_updated > self.SESSION_TTL
    
    def update_timestamp(self):
        """更新最后访问时间"""
        self.last_updated = time.time()
    
    def add_user_input(self, user_input: str):
        """添加用户输入到历史"""
        self.history.append({"role": "user", "content": user_input, "time": datetime.now().isoformat()})
        self.update_timestamp()
    
    def add_system_response(self, response: str):
        """添加系统响应到历史"""
        self.history.append({"role": "assistant", "content": response, "time": datetime.now().isoformat()})
        self.update_timestamp()
    
    def update_entities(self, new_entities: Dict[str, Any]):
        """更新实体"""
        self.entities.update({k: v for k, v in new_entities.items() if v is not None and v != ""})
        self.update_timestamp()
    
    def set_missing_fields(self, fields: List[Tuple[str, str, str]]):
        """设置缺失字段"""
        self.missing_fields = fields
        self.update_timestamp()
    
    def get_next_question(self) -> Optional[Tuple[str, str, str]]:
        """获取下一个需要询问的字段"""
        if self.missing_fields:
            return self.missing_fields[0]
        return None
    
    def mark_field_collected(self, field_name: str):
        """标记字段已收集"""
        self.missing_fields = [f for f in self.missing_fields if f[0] != field_name]
        if not self.missing_fields:
            self.status = DialogueStatus.READY
        self.update_timestamp()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "entities": self.entities,
            "missing_fields": [
                {"field": f[0], "label": f[1], "prompt": f[2]} 
                for f in self.missing_fields
            ],
            "channel": self.channel,
            "channel_confidence": self.channel_confidence,
            "history_count": len(self.history),
            "expires_in": self.SESSION_TTL - (time.time() - self.last_updated)
        }


class DialogueManager:
    """对话管理器"""
    
    # 关键字段优先级配置
    REQUIRED_FIELDS_PRIORITY = [
        ("materialName", "材料名称", "您想查询什么材料的价格？（如：钢筋、水泥、铝合金等）"),
        ("province", "省份", "请问是哪个省份的材料？（如：广东省、江苏省）"),
        ("city", "城市", "具体是哪个城市？（如：深圳市、南京市）"),
        ("materialModelSpec", "规格型号", "请问需要什么规格型号？（如：HRB400、P.O 42.5，不知道可直接跳过）"),
        ("brand", "品牌", "有指定的品牌吗？（如：宝钢、三棵树，没有可直接跳过）"),
    ]
    
    # 可选字段（不强制要求）
    OPTIONAL_FIELDS = [
        ("startReleaseDate", "开始时间", "查询从什么时间开始？（格式：2024-01-01，可直接跳过）"),
        ("endReleaseDate", "结束时间", "查询到什么时间结束？（格式：2024-12-31，可直接跳过）"),
    ]
    
    def __init__(self):
        self.sessions: Dict[str, DialogueSession] = {}
        self.logger = logging.getLogger("easy_rag_api")
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Tuple[str, DialogueSession]:
        """获取或创建会话"""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            if not session.is_expired():
                session.update_timestamp()
                return session_id, session
            else:
                # 过期，删除旧会话
                del self.sessions[session_id]
        
        # 创建新会话
        new_session_id = session_id or self._generate_session_id()
        new_session = DialogueSession(session_id=new_session_id)
        self.sessions[new_session_id] = new_session
        return new_session_id, new_session
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def check_missing_fields(self, entities: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """检查缺失的关键字段"""
        missing = []
        for field, label, prompt in self.REQUIRED_FIELDS_PRIORITY:
            if not entities.get(field):
                missing.append((field, label, prompt))
        return missing
    
    def process_user_input(
        self, 
        session_id: Optional[str], 
        user_input: str,
        extracted_entities: Dict[str, Any],
        channel: str,
        channel_confidence: float
    ) -> Dict[str, Any]:
        """
        处理用户输入
        
        Returns:
            {
                "session_id": str,
                "status": "collecting" | "ready" | "completed",
                "entities": dict,  # 当前已收集的实体
                "next_question": tuple | None,  # 下一个问题
                "is_complete": bool,  # 是否已收集完整
                "response_text": str,  # 给用户的回复文本
                "quick_options": list,  # 快捷选项
            }
        """
        session_id, session = self.get_or_create_session(session_id)
        
        # 添加用户输入到历史
        session.add_user_input(user_input)
        
        # 如果是新会话，初始化渠道信息
        if not session.channel:
            session.channel = channel
            session.channel_confidence = channel_confidence
        
        # 更新实体
        session.update_entities(extracted_entities)
        
        # 检查缺失字段
        missing = self.check_missing_fields(session.entities)
        session.set_missing_fields(missing)
        
        # 生成响应
        result = self._generate_response(session, user_input)
        
        # 添加系统响应到历史
        session.add_system_response(result["response_text"])
        
        return result
    
    def _generate_response(self, session: DialogueSession, user_input: str) -> Dict[str, Any]:
        """生成响应"""
        result = {
            "session_id": session.session_id,
            "status": session.status.value,
            "entities": session.entities,
            "channel": session.channel,
        }
        
        # 检查是否已就绪（有材料名称即可查询）
        has_material = bool(session.entities.get("materialName"))
        
        if not has_material:
            # 必须收集材料名称
            next_q = session.get_next_question()
            if next_q:
                result["next_question"] = {
                    "field": next_q[0],
                    "label": next_q[1],
                    "prompt": next_q[2]
                }
                result["response_text"] = next_q[2]
                result["is_complete"] = False
                result["quick_options"] = self._get_quick_options(next_q[0])
            else:
                result["response_text"] = "请告诉我您想查询什么材料的价格？"
                result["is_complete"] = False
                result["quick_options"] = []
        
        elif session.missing_fields:
            # 有材料名称，但还有其他缺失字段
            # 询问是否继续补充，或直接查询
            next_q = session.get_next_question()
            
            if next_q and session.channel_confidence < 0.8:
                # 渠道置信度低，先确认渠道
                result["response_text"] = (
                    f"收到，查询「{session.entities.get('materialName')}」的价格。\n"
                    f"当前渠道：{self._get_channel_display_name(session.channel)}\n"
                    f"{next_q[2] if next_q else ''}"
                )
            else:
                # 提供选择：继续补充或直接查询
                result["response_text"] = (
                    f"收到，查询「{session.entities.get('materialName')}」在"
                    f"{session.entities.get('province', '全国')}"
                    f"{session.entities.get('city', '')}的价格。\n\n"
                    f"当前查询条件：\n"
                    f"{self._format_current_entities(session.entities)}\n"
                    f"您可以选择：\n"
                    f"1. 直接查询（使用当前条件）\n"
                    f"2. 补充更多信息"
                )
                result["quick_options"] = [
                    {"text": "🔍 直接查询", "action": "query_now"},
                    {"text": "➕ 补充条件", "action": "continue_collect"}
                ]
            
            result["is_complete"] = True  # 已有材料名称，可以查询
            result["can_query"] = True
        
        else:
            # 所有必需字段已收集
            session.status = DialogueStatus.READY
            result["status"] = DialogueStatus.READY.value
            result["is_complete"] = True
            result["can_query"] = True
            result["response_text"] = (
                f"信息已收集完整！\n"
                f"查询条件：\n{self._format_current_entities(session.entities)}\n"
                f"正在为您查询..."
            )
            result["quick_options"] = [
                {"text": "✅ 确认查询", "action": "query_now"},
                {"text": "🔄 修改条件", "action": "modify"}
            ]
        
        return result
    
    def _get_quick_options(self, field_name: str) -> List[Dict]:
        """获取快捷选项"""
        options_map = {
            "province": [
                {"text": "广东省", "value": "广东省"},
                {"text": "江苏省", "value": "江苏省"},
                {"text": "浙江省", "value": "浙江省"},
                {"text": "北京市", "value": "北京市"},
                {"text": "上海市", "value": "上海市"},
            ],
            "city": [
                {"text": "深圳市", "value": "深圳市"},
                {"text": "广州市", "value": "广州市"},
                {"text": "南京市", "value": "南京市"},
                {"text": "杭州市", "value": "杭州市"},
            ],
        }
        return options_map.get(field_name, [])
    
    def _get_channel_display_name(self, channel: str) -> str:
        """获取渠道显示名"""
        names = {
            "information_price": "信息价",
            "manufacturer_price": "厂商报价",
            "zc_price": "智诚信息价",
        }
        return names.get(channel, "未知渠道")
    
    def _format_current_entities(self, entities: Dict) -> str:
        """格式化当前实体"""
        lines = []
        field_names = {
            "materialName": "材料名称",
            "province": "省份",
            "city": "城市",
            "materialModelSpec": "规格型号",
            "brand": "品牌",
        }
        for field, label in field_names.items():
            value = entities.get(field)
            if value:
                lines.append(f"  ✅ {label}：{value}")
            else:
                lines.append(f"  ⭕ {label}：未指定")
        return "\n".join(lines)
    
    def get_session(self, session_id: str) -> Optional[DialogueSession]:
        """获取会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if not session.is_expired():
                return session
            else:
                del self.sessions[session_id]
        return None
    
    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
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
_dialogue_manager: Optional[DialogueManager] = None


def get_dialogue_manager() -> DialogueManager:
    """获取对话管理器单例"""
    global _dialogue_manager
    if _dialogue_manager is None:
        _dialogue_manager = DialogueManager()
    return _dialogue_manager
