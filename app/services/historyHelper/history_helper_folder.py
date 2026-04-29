from app.services.historyHelper.history_helper_base import HistoryHelperBase
from datetime import datetime
from pathlib import Path
import uuid
import logging

import json

# 写入文件时保留中文（非 \\uXXXX），并缩进便于阅读
_JSON_DUMP_KWARGS = {"ensure_ascii": False, "indent": 2}

_TITLE_MAX_LEN = 80


def _now_iso() -> str:
    return datetime.now().isoformat()


def _updated_at_for_list(data: dict, file_mtime: float) -> tuple[str, float]:
    """
    返回 (展示用 updated_at 字符串, 用于排序的时间戳)。
    无字段或解析失败时回退为文件修改时间（兼容旧数据）。
    """
    s = data.get("updated_at")
    if isinstance(s, str) and s.strip():
        try:
            ts = datetime.fromisoformat(s.strip()).timestamp()
            return s.strip(), ts
        except ValueError:
            pass
    iso = datetime.fromtimestamp(file_mtime).isoformat()
    return iso, file_mtime


def _derive_conversation_title(history_data: dict) -> str:
    """用首条用户文本作为标题；过长截断。"""
    for msg in history_data.get("conversation_list", []):
        if msg.get("role") != "user":
            continue
        c = msg.get("content", "")
        if isinstance(c, str) and c.strip():
            s = c.strip()
            if len(s) <= _TITLE_MAX_LEN:
                return s
            return s[: _TITLE_MAX_LEN - 3] + "..."
        if isinstance(c, dict) and c:
            s = str(c.get("text", c))[:200].strip()
            if s:
                return s[:_TITLE_MAX_LEN] if len(s) <= _TITLE_MAX_LEN else s[: _TITLE_MAX_LEN - 3] + "..."
    return "新对话"


class HistoryHelperFolder(HistoryHelperBase):
    def __init__(self, history_folder:str):
        self.history_folder = Path(history_folder)
        self.history_folder.mkdir(parents=True, exist_ok=True)

    def create_history(self, username:str, conversation_id:str) -> None:
        history_file = self.history_folder / f"{username}/{conversation_id}.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file.touch()
        history_file.write_text(
            json.dumps(
                {
                    "id": conversation_id,
                    "username": username,
                    "conversation_list": [],
                    "updated_at": _now_iso(),
                },
                **_JSON_DUMP_KWARGS,
            ),
            encoding="utf-8",
        )

    def append_message_user(self, username:str, conversation_id:str, message:str):
        try:
            history_file = self.history_folder / f"{username}/{conversation_id}.json"
            history_data = json.loads(history_file.read_text(encoding="utf-8"))
            history_data["conversation_list"].append({
                "role": "user",
                "content": message
            })
            history_data["updated_at"] = _now_iso()
            history_file.write_text(
                json.dumps(history_data, **_JSON_DUMP_KWARGS),
                encoding="utf-8",
            )
        except Exception as e:
            logging.error(f"添加用户消息失败: {str(e)}")

    def append_message_assistant(self, username:str, conversation_id:str, message:dict):
        try:
            history_file = self.history_folder / f"{username}/{conversation_id}.json"
            history_data = json.loads(history_file.read_text(encoding="utf-8"))
            history_data["conversation_list"].append({
                "role": "assistant",
                "content": message
            })
            history_data["updated_at"] = _now_iso()
            history_file.write_text(
                json.dumps(history_data, **_JSON_DUMP_KWARGS),
                encoding="utf-8",
            )
        except Exception as e:
            logging.error(f"添加助手消息失败: {str(e)}")

    def get_history(self, username:str, conversation_id:str) -> dict:
        try:
            history_file = self.history_folder / f"{username}/{conversation_id}.json"
            history_data = json.loads(history_file.read_text(encoding="utf-8"))
            return history_data
        except Exception as e:
            return None
            raise HTTPException(status_code=500, detail=f"获取历史失败: {str(e)}")

    def delete_history(self, username:str, conversation_id:str) -> bool:
        history_file = self.history_folder / f"{username}/{conversation_id}.json"
        if not history_file.is_file():
            return False
        history_file.unlink()
        return True

    def get_all_history(self, username:str) -> list:
        user_dir = self.history_folder / username
        if not user_dir.is_dir():
            return []
        files = list(user_dir.glob("*.json"))
        rows: list[tuple[float, dict]] = []
        for history_file in files:
            conv_id = history_file.stem
            try:
                raw = history_file.read_text(encoding="utf-8")
                data = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                continue
            mtime = history_file.stat().st_mtime
            updated_at_str, sort_ts = _updated_at_for_list(data, mtime)
            rows.append(
                (
                    sort_ts,
                    {
                        "conversation_id": conv_id,
                        "title": _derive_conversation_title(data),
                        "updated_at": updated_at_str,
                    },
                )
            )
        rows.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in rows]