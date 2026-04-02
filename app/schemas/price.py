from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.material import MaterialItem

class DirectPriceQueryRequest(BaseModel):
    datatype: str = Field(..., description="数据类型，如 informaterial 表示信息价")
    list: List[MaterialItem] = Field(..., description="材料信息列表")
    question: Optional[str] = Field(None, description="用户问题")

class DirectPriceQueryResponse(BaseModel):
    text: Optional[str] = None
    intent: str = "price_recommendation"
    channel: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
    price_data: Optional[List[Dict[str, Any]]] = None
