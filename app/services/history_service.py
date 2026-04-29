# 对话历史管理
import logging
from typing import Optional

from app.services.historyHelper.history_helper_base import HistoryHelperBase


class HistoryService:
    def __init__(self, history_helper:HistoryHelperBase):
        self.history_helper = history_helper

    def create_history(self, username:str, conversation_id:str) -> None:
        return self.history_helper.create_history(username, conversation_id)

    def append_message_user(self, username:str, conversation_id:str, message:str):
        return self.history_helper.append_message_user(username, conversation_id, message)

    def append_message_assistant(self, username:str, conversation_id:str, message:str):
        return self.history_helper.append_message_assistant(username, conversation_id, message)

    def get_history(self, username:str, conversation_id:str) -> dict:
        return self.history_helper.get_history(username, conversation_id)

    def delete_history(self, username:str, conversation_id:str) -> bool:
        return self.history_helper.delete_history(username, conversation_id)

    def get_all_history(self, username:str) -> list:
        return self.history_helper.get_all_history(username)

    def delete_conversation_if_empty(self, username: str, conversation_id: str) -> bool:
        """若会话文件中尚无对话轮次，则删除文件（用于客户端中断且未写入历史时清理空会话）。"""
        try:
            data = self.get_history(username, conversation_id)
        except Exception:
            return False
        cl = data.get("conversation_list") or []
        if isinstance(cl, list) and len(cl) == 0:
            return self.delete_history(username, conversation_id)
        return False


history_service: Optional[HistoryService] = None  # 全局服务实例


def init_history_service(history_helper: HistoryHelperBase) -> HistoryService:
    """初始化全局 history_service（进程启动时调用一次）。"""
    global history_service
    history_service = HistoryService(history_helper=history_helper)
    return history_service


def get_history_service() -> HistoryService:
    """获取全局 history_service，未初始化则抛出异常。"""
    if history_service is None:
        raise RuntimeError("history_service 未初始化，请先调用 init_history_service")
    return history_service