import os
import asyncio
import uvicorn
from fastapi import FastAPI

from app.core.logger import setup_logger
from app.api.endpoints import router as api_router
from app.api.mcp_endpoints import router as mcp_router, init_mcp_endpoints
from app.api.conversation_endpoints import router as conversation_router
from app.services.rag_service import RAGService
from app.services.history_service import init_history_service, get_history_service
from app.services.historyHelper.history_helper_folder import HistoryHelperFolder
from app.config import HISTORY_ROOT_PATH
from app.db.base import engine, Base, AsyncSessionLocal
from app.services.conversation_service import ConversationService
from app.utils.conversation_manager import get_conversation_manager

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

app = FastAPI(
    title="Easy-RAG API",
    description="基于RAG的智能问答系统API，支持MCP工具调用和Agent多轮对话。",
    version="1.1.0"
)

# 挂载API路由
app.include_router(api_router)
app.include_router(mcp_router, prefix="/mcp", tags=["MCP & Agent"])
app.include_router(conversation_router, tags=["Conversation"])

logger = setup_logger()
rag_service = RAGService()

@app.on_event("startup")
async def startup_event():
    global rag_service, logger
    logger.info("启动Easy-RAG API服务...")
    
    # 初始化数据库
    logger.info("初始化数据库...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 初始化管理员账号、清理过期会话并注入数据库会话到ConversationManager
    async with AsyncSessionLocal() as db:
        service = ConversationService(db)
        admin = await service.get_or_create_user("admin")
        logger.info(f"管理员账号初始化完成: id={admin.id}, username={admin.username}")
        
        cleaned = await service.cleanup_inactive_conversations(days=7)
        if cleaned > 0:
            logger.info(f"自动清理了 {cleaned} 个超过7天未活跃的会话")
    
    conversation_manager = get_conversation_manager()
    conversation_manager.set_db_session_maker(AsyncSessionLocal)
    logger.info("ConversationManager数据库会话注入完成")
    
    await rag_service.initialize()

    # 初始化全局 history_service（全局单例）
    history_helper = HistoryHelperFolder(history_folder=HISTORY_ROOT_PATH)
    init_history_service(history_helper)
    hs = get_history_service()

    # 将rag_service注入api模块，方便访问
    from app.api import endpoints, mcp_endpoints
    endpoints.rag_service = rag_service
    endpoints.history_service = hs
    mcp_endpoints.history_service = hs
    
    # 初始化MCP端点
    init_mcp_endpoints(rag_service)
    logger.info("MCP和Agent服务初始化完成")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("关闭Easy-RAG API服务...")

if __name__ == "__main__":
    uvicorn.run("app.api_server:app", host="0.0.0.0", port=8001, reload=False)
