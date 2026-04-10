from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PriceRequest(BaseModel):
    question: str
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    force_new: bool = Field(False, description="强制创建新会话，清空历史对话记录")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "从信息价查铝合金幕墙型材的价格",
                "session_id": "abc123",
                "force_new": False
            }
        }

class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    num_docs: int = Field(10, description="检索文档数量", ge=1, le=50)
    session_id: Optional[str] = Field(None, description="会话ID，用于多轮对话")
    force_new: bool = Field(False, description="强制创建新会话，清空历史对话记录")

class QueryResponse(BaseModel):
    answer: str
    contexts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
