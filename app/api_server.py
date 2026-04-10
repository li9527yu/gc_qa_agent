import os
import asyncio
import uvicorn
from fastapi import FastAPI

from app.core.logger import setup_logger
from app.api.endpoints import router as api_router
from app.api.mcp_endpoints import router as mcp_router, init_mcp_endpoints
from app.services.rag_service import RAGService

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

app = FastAPI(
    title="Easy-RAG API",
    description="基于RAG的智能问答系统API，支持MCP工具调用和Agent多轮对话。",
    version="1.1.0"
)

# 挂载API路由
app.include_router(api_router)
app.include_router(mcp_router, prefix="/mcp", tags=["MCP & Agent"])

logger = setup_logger()
rag_service = RAGService()

@app.on_event("startup")
async def startup_event():
    global rag_service, logger
    logger.info("启动Easy-RAG API服务...")
    await rag_service.initialize()
    # 将rag_service注入api模块，方便访问
    from app.api import endpoints
    endpoints.rag_service = rag_service
    
    # 初始化MCP端点
    init_mcp_endpoints(rag_service)
    logger.info("MCP和Agent服务初始化完成")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("关闭Easy-RAG API服务...")

if __name__ == "__main__":
    uvicorn.run("app.api_server:app", host="0.0.0.0", port=8001, reload=False)
