import os
import asyncio
import uvicorn
from fastapi import FastAPI

from app.core.logger import setup_logger
from app.api.endpoints import router as api_router
from app.services.rag_service import RAGService

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

app = FastAPI(
    title="Easy-RAG API",
    description="基于RAG的智能问答系统API。大致流程step1:通过/api/v1/query/intent接口判断用户提问是知识问答或价格推荐；step2:根据拿到结果，知识问答走接口/api/v1/query/stream，价格推荐走接口/api/v1/query/price。另外由于上传和删除文件两个操作需要更新知识库所以耗时较久，可通过/api/v1/task_status接口查询任务状态。",
    version="1.0.0"
)

# 挂载API路由
app.include_router(api_router)

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

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("关闭Easy-RAG API服务...")

if __name__ == "__main__":
    uvicorn.run("app.api_server:app", host="0.0.0.0", port=8001, reload=False)
