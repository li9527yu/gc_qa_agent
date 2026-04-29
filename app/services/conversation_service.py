from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, update
from sqlalchemy.orm import selectinload
from typing import List, Optional
import json
from datetime import datetime

from app.db.models import Conversation, Message, User


class ConversationService:
    """会话服务"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_conversation(
        self,
        user_id: int,
        session_id: str,
        conv_type: str,
        title: str = "新会话"
    ) -> Conversation:
        """创建会话"""
        conversation = Conversation(
            user_id=user_id,
            session_id=session_id,
            type=conv_type,
            title=title
        )
        self.db.add(conversation)
        await self.db.commit()
        await self.db.refresh(conversation)
        return conversation

    async def get_conversation(
        self,
        user_id: int,
        session_id: str
    ) -> Optional[Conversation]:
        """获取会话"""
        result = await self.db.execute(
            select(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.session_id == session_id,
                Conversation.status == "active"
            )
            .options(selectinload(Conversation.messages))
        )
        return result.scalar_one_or_none()

    async def list_conversations(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Conversation]:
        """获取用户会话列表"""
        result = await self.db.execute(
            select(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.status == "active"
            )
            .order_by(desc(Conversation.updated_at))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def save_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: dict = None,
    ) -> Message:
        """保存消息"""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata_json=json.dumps(metadata, ensure_ascii=False) if metadata else None
        )
        self.db.add(message)

        # 更新会话的消息数和最后消息时间
        await self.db.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                message_count=Conversation.message_count + 1,
                last_message_at=datetime.utcnow()
            )
        )

        await self.db.commit()
        await self.db.refresh(message)
        return message

    async def update_conversation_title(
        self,
        user_id: int,
        session_id: str,
        title: str
    ) -> bool:
        """更新会话标题"""
        result = await self.db.execute(
            update(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.session_id == session_id
            )
            .values(title=title)
        )
        await self.db.commit()
        return result.rowcount > 0

    async def archive_conversation(
        self,
        user_id: int,
        session_id: str
    ) -> bool:
        """归档（软删除）会话"""
        result = await self.db.execute(
            update(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.session_id == session_id
            )
            .values(status="archived")
        )
        await self.db.commit()
        return result.rowcount > 0

    async def cleanup_inactive_conversations(self, days: int = 7) -> int:
        """自动归档超过指定天数未活跃的会话"""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.db.execute(
            update(Conversation)
            .where(
                Conversation.status == "active",
                Conversation.last_message_at < cutoff
            )
            .values(status="archived")
        )
        await self.db.commit()
        return result.rowcount

    async def get_or_create_user(self, username: str) -> User:
        """获取或创建用户（用于初始化管理员）"""
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        user = result.scalar_one_or_none()
        if user:
            return user
        user = User(username=username, is_active=True)
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user
