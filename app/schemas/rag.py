from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class IntentRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "从信息价查铝合金幕墙型材的价格"
            }
        }

class IntentResponse(BaseModel):
    intent: str
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "intent": "price_recommendation",
                "question": "从信息价查铝合金幕墙型材的价格"
            }
        }

class PriceRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "从信息价查铝合金幕墙型材的价格"
            }
        }

class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    num_docs: int = Field(10, description="检索文档数量", ge=1, le=50)

class QueryResponse(BaseModel):
    answer: str
    contexts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
