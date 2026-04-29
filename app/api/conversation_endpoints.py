from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.db.base import get_db
from app.db.models import Conversation
from app.services.conversation_service import ConversationService

router = APIRouter()


def get_current_user_id() -> int:
    """当前用户ID（临时方案：固定返回管理员账号ID）"""
    return 1


@router.get("/api/v1/conversations")
async def list_conversations(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取当前用户的所有会话列表"""
    service = ConversationService(db)
    conversations = await service.list_conversations(user_id)
    
    return [
        {
            "session_id": conv.session_id,
            "title": conv.title or "新会话",
            "type": conv.type,
            "message_count": conv.message_count,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        }
        for conv in conversations
    ]


@router.get("/api/v1/conversations/{session_id}/messages")
async def get_conversation_messages(
    session_id: str,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """根据session_id获取对应的历史所有消息"""
    service = ConversationService(db)
    conversation = await service.get_conversation(user_id, session_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    import json
    messages = []
    for msg in conversation.messages:
        metadata = None
        if msg.metadata_json:
            try:
                metadata = json.loads(msg.metadata_json)
            except Exception:
                metadata = None
        messages.append({
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
            "metadata": metadata,
        })
    
    return {
        "session_id": session_id,
        "title": conversation.title or "新会话",
        "type": conversation.type,
        "messages": messages
    }


@router.delete("/api/v1/conversations/{session_id}")
async def delete_conversation(
    session_id: str,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """删除（归档）指定的历史会话"""
    service = ConversationService(db)
    success = await service.archive_conversation(user_id, session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return {"success": True, "message": "会话已删除"}
