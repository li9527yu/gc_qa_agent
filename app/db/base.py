import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./rag_agent.db"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    poolclass=NullPool if "sqlite" in DATABASE_URL else None,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()


async def get_db() -> AsyncSession:
    """获取数据库会话（用于FastAPI依赖注入）"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
