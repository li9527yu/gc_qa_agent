from fastapi import APIRouter, Body, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import AsyncGenerator, List, Optional
import os
from pathlib import Path
import shutil
import json
import uuid
import base64
from concurrent.futures import ThreadPoolExecutor
import asyncio
import functools
import logging
import threading

from app.schemas.files import DeleteResponse, DeleteRequest, UploadResponse
from app.schemas.rag import QueryRequest, QueryResponse, IntentRequest, PriceRequest
from app.schemas.price import DirectPriceQueryRequest, DirectPriceQueryResponse
from app.services.rag_service import RAGService
from app.services.file_processor import process_uploaded_files
from app.services.rag_service import ChannelType
from pydantic import BaseModel, Field
from app.config import DATA_PATH,RELATED_DATA_PATH

router = APIRouter()
rag_service: RAGService = None  # 全局服务实例，由 main.py 注入
task_status_dict = {}  # 可替换为 Redis 缓存更稳


# ================================================================================
# 性能统计辅助函数
# ================================================================================

def log_llm_generation_performance(logger, input_tokens: int, output_tokens: int, 
                                   ttft: float, total_time: float):
    """
    打印 LLM 生成性能统计
    
    Args:
        logger: 日志记录器
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        ttft: Time To First Token (首个token响应时间)
        total_time: 总生成时间
    """
    logger.info("=" * 80)
    logger.info("🤖 LLM 生成性能统计")
    logger.info("=" * 80)
    logger.info(f"🎫 Token 统计:")
    logger.info(f"   - 输入 Token (估算): {input_tokens:,} tokens")
    logger.info(f"   - 输出 Token (估算): {output_tokens:,} tokens")
    logger.info(f"   - 总 Token: {input_tokens + output_tokens:,} tokens")
    logger.info(f"⏱️  生成时间:")
    logger.info(f"   - TTFT (首个Token): {ttft:.3f}s")
    logger.info(f"   - 总生成时间: {total_time:.3f}s")
    logger.info(f"   - 生成速度: {output_tokens/total_time:.1f} tokens/s")
    logger.info("=" * 80)


def log_total_request_performance(logger, total_time: float, llm_time: float, ttft: float):
    """
    打印完整请求性能统计
    
    Args:
        logger: 日志记录器
        total_time: 总耗时
        llm_time: LLM 生成耗时
        ttft: Time To First Token
    """
    logger.info("=" * 80)
    logger.info("📊 完整请求性能统计")
    logger.info("=" * 80)
    logger.info(f"⏱️  总耗时: {total_time:.3f}s")
    logger.info(f"   - LLM 生成: {llm_time:.3f}s")
    logger.info(f"   - TTFT: {ttft:.3f}s")
    logger.info("=" * 80)


# ✅ 全局对话历史（存储最近 6 条消息：user/assistant）
messages: list[dict] = []

# 全局线程池（进程启动时初始化，避免反复创建）
executor = ThreadPoolExecutor(max_workers=2)

# 初始化锁，防止并发重建知识库导致状态不一致
initialize_lock = threading.Lock()

@router.post(
    "/api/v1/upload",
    summary="批量上传文件",
    description="""
    批量上传文件接口，支持异步触发知识库更新任务。
    
    - files: 上传的文件列表
    - 支持的上传格式包含 TXT, PDF, MD
    - 成功上传后会触发后台 RAG 服务更新
    - 可通过 `/api/v1/task_status` 查询后台任务状态
    """,
    responses={
        200: {
            "description": "成功响应",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "original_file": "/home/tpc/suda/rag/easy-rag/app/dataset/uploads/test2.txt",
                                "markdown_file": "/home/tpc/suda/rag/easy-rag/app/dataset/data/test2_258350b6.md",
                                "status": "success"
                            },
                            {
                                "original_file": "/home/tpc/suda/rag/easy-rag/app/dataset/uploads/test1.pdf",
                                "markdown_file": "/home/tpc/suda/rag/easy-rag/app/dataset/data/test1_9965f086.md",
                                "status": "success"
                            }
                        ],
                        "task_id": "733a0362-9db9-4ba7-8cab-6cc59cecce5c"
                    }
                }
            }
        }
    }
)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    # 将同步的文件处理放到线程池执行，避免阻塞 FastAPI 事件循环
    results = await asyncio.to_thread(process_uploaded_files, files)
    success = [r for r in results if r.get("status") == "success"]

    if success and rag_service:
        task_id = str(uuid.uuid4())
        task_status_dict[task_id] = "pending"
        background_tasks.add_task(executor.submit, _sync_initialize, task_id)
        return {"results": results, "task_id": task_id}

    return {"results": results}

def _sync_initialize(task_id: str):
    """
    真正耗时的初始化逻辑，跑在独立线程里。
    使用全局锁保证同时只有一个初始化任务在执行，防止并发冲突。
    """
    with initialize_lock:
        try:
            # 如果你的 rag_service.initialize 是 async，需要启动一次事件循环
            asyncio.run(rag_service.initialize())
            task_status_dict[task_id] = "success"
        except Exception as e:
            task_status_dict[task_id] = f"failed: {e}"

@router.get(
    "/api/v1/files",
    summary="获取文件列表",
    description="""
        获取语料库目录下的所有文件列表，按修改时间倒序排列
    """,
    responses={
        200: {
            "description": "成功响应",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "corpus_path": "app/dataset/data/",
                        "total_files": 2,
                        "total_size_mb": 0.26,
                        "files": [
                            {
                                "filename": "test2_f0e57789.md",
                                "size": 110167,
                                "size_mb": 0.11,
                                "modified_time": "2025-12-19T08:34:49.028561",
                                "path": "app/dataset/data/test2_f0e57789.md"
                            },
                            {
                                "filename": "output_piclegend.md",
                                "size": 165112,
                                "size_mb": 0.16,
                                "modified_time": "2025-11-17T05:01:19.252978",
                                "path": "app/dataset/data/output_piclegend.md"
                            }
                        ],
                        "timestamp": "2025-12-19T08:36:47.929076"
                    }
                }
            }
        }
    }
)
async def list_files():
    """
    获取语料库目录下的所有文件列表
    
    Returns:
        dict: 文件列表信息
    """
    try:
        # 定义语料库路径
        corpus_path = RELATED_DATA_PATH
        
        # 检查目录是否存在
        if not os.path.exists(corpus_path):
            raise HTTPException(status_code=404, detail="语料库目录不存在")
        
        files = []
        total_size = 0
        
        # 遍历目录获取文件信息
        for filename in os.listdir(corpus_path):
            file_path = os.path.join(corpus_path, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                file_mtime = os.path.getmtime(file_path)
                
                files.append({
                    "filename": filename,
                    "size": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "modified_time": datetime.fromtimestamp(file_mtime).isoformat(),
                    "path": file_path
                })
                total_size += file_size
        
        # 按修改时间排序（最新的在前）
        files.sort(key=lambda x: x["modified_time"], reverse=True)
        
        return {
            "status": "success",
            "corpus_path": corpus_path,
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": files,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

@router.delete(
    "/api/v1/delete/batch",
    response_model=DeleteResponse,
    summary="批量删除文件",
    description="""
    批量删除指定文件。

    - 仅支持删除语料库目录下的文件
    - 仅需传文件名，不包含路径。
        例如：
        path = "app/dataset/data/output_piclegend.md"
        则传入：output_piclegend.md
    - 成功删除后会异步触发知识库更新
    - 可通过 `/api/v1/task_status` 查询后台任务状态
    """,
    responses={
        200: {
            "description": "成功响应",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "批量删除完成，成功删除 1 个文件，失败 0 个文件，知识库正在后台更新",
                        "deleted_files": [
                            {
                                "filename": "test2_1f972e43.md",
                                "size_mb": 0.11,
                                "modified_time": "2025-12-19T08:23:56.443916"
                            }
                        ],
                        "failed_files": [],
                        "total_deleted_size_mb": 0.11,
                        "timestamp": "2025-12-19T08:32:13.077454",
                        "task_id": "8e1daab8-3789-49ab-b91c-9f266f961c4f"
                    }
                }
            }
        }
    })
async def delete_files(
    req: DeleteRequest = Body(
        ...,
        example={
            "filenames": ["text1.md", "test2_1f972e43.md"]
        }
    ),
    background_tasks: BackgroundTasks = None
):
    filenames = req.filenames
    try:
        # 定义语料库路径
        corpus_path = DATA_PATH
        
        deleted_files = []
        failed_files = []
        total_deleted_size = 0
        
        for filename in filenames:
            try:
                file_path = os.path.join(corpus_path, filename)
                
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    failed_files.append({
                        "filename": filename,
                        "error": "文件不存在"
                    })
                    continue
                
                # 检查文件是否在允许的目录内（安全检查）
                real_file_path = os.path.realpath(file_path)
                real_corpus_path = os.path.realpath(corpus_path)
                
                if not real_file_path.startswith(real_corpus_path):
                    failed_files.append({
                        "filename": filename,
                        "error": "不允许删除该目录外的文件"
                    })
                    continue
                
                # 获取文件信息
                file_size = os.path.getsize(file_path)
                file_mtime = os.path.getmtime(file_path)
                
                # 删除文件
                os.remove(file_path)
                
                deleted_files.append({
                    "filename": filename,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "modified_time": datetime.fromtimestamp(file_mtime).isoformat()
                })
                total_deleted_size += file_size
                
            except Exception as e:
                failed_files.append({
                    "filename": filename,
                    "error": str(e)
                })
        
        # 如果有文件被成功删除，则更新RAG服务
        if deleted_files and rag_service:
            task_id = str(uuid.uuid4())
            task_status_dict[task_id] = "pending"
            background_tasks.add_task(executor.submit, _sync_initialize, task_id)
        else:
            task_id = None
        
        return {
            "status": "success",
            "message": f"批量删除完成，成功删除 {len(deleted_files)} 个文件，失败 {len(failed_files)} 个文件，知识库正在后台更新",
            "deleted_files": deleted_files,
            "failed_files": failed_files,
            "total_deleted_size_mb": round(total_deleted_size / (1024 * 1024), 2),
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量删除文件失败: {str(e)}")

@router.post(
    "/api/v1/query/stream",
    summary="流式查询接口",
    description="""
    流式查询接口，支持按 token 流式返回回答内容。
    
    - request: 查询请求对象 QueryRequest
    - 成功返回 StreamingResponse，按 chunk 流式输出文本
    - 每次回答结束后会返回 JSON 格式的 meta 信息，包括完整文本和上下文
    - 参数介绍：
        {
            "question": 用户问题
            "num_docs": 返回的参考文档个数, 最多 10 个
        }
    - 响应类型：text/plain
    """,
    responses={
        200: {
            "description": "成功响应（StreamingResponse）",
            "content": {
                "text/plain": {
                    "example": (
                        "工程\n造价\n是指\n在\n工程\n项目的\n整个\n生命周期\n中\n，\n"
                        "为\n完成\n工程\n项目的\n建设\n、\n运营\n和\n维护\n等活动\n所\n发生的\n全部\n费用\n。\n"
                        "它\n包括\n了\n从\n项目\n立项\n、\n设计\n、\n施工\n、\n直至\n竣工验收\n、\n交付\n使用\n等\n各个\n阶段\n的\n费用\n总\n和\n。\n"
                        "工程\n造价\n的\n计算\n通常\n涉及到\n材料\n费\n、\n人工\n费\n、\n机械\n费\n、\n管理\n费\n、\n利润\n、\n税\n金\n等\n各项\n费用\n的\n汇总\n。\n"
                        "[END]\n"
                        "{\"text\": \"工程造价是指在工程项目的整个生命周期中，为完成工程项目的建设、运营和维护等活动所发生的全部费用。它包括了从项目立项、设计、施工、直至竣工验收、交付使用等各个阶段的费用总和。工程造价的计算通常涉及到材料费、人工费、机械费、管理费、利润、税金等各项费用的汇总。\", "
                        "\"contexts\": ["
                        "{\"page_content\": \"﻿\\n1  # <a name=\\\"_toc8194\\\"></a><a name=\\\"_toc31236\\\"></a>**系统介绍**\\n   **工程造价大数据系统**是以大数据、人工智能、云计算等先进IT技术为依托的工程造价数据分析系统,公司放眼未来，加快企业数字化转型，进一步增强公司咨询服务能力,突破传统算量算价的传统业务，迈向以工程价值、专业化、数字化的工程咨询新型服务。\\n\\n   大数据系统应用于工程前期决策、概算优化与比选、招投标复核、中标价分析、结算的差异化分析以及全过程中进行动态的成本监控与管理,核心功能主要包括以下：\", \"metadata\": {\"type\": \"text\", \"chunk_id\": 1, \"source\": \"output_piclegend.md\"}}, "
                        "{\"page_content\": \"- 根据模板设定生成报告\\n\\n这张图片显示的是一个数据报告的截图。报告显示了几个不同项目的工程造价信息。  在表格的第一列是项目名称或编号，包括“桩基、盖板及护坡”、“景观给排水工...”。第二列是每个项目的限价（即最高允许的价格），第三列则是投标价格，第四列计算出的差值和第五列单个工程项目单价增减率百分比。  具体来说：  - “桩基、盖板及护坡”的限价为18,163,109.03元人民币。 - 投标价格为15,705,538.59元人民币，与上限相比降低了2,457,570.44元，下降幅度达到了13.53%。 - 其他列出的项目也都有类似的报价对比情况，并且都给出了相应的降价比例。  这个表可能用于评估各个承包商对这些项目的竞标结果以及他们的成本控制能力。通过比较实际投标价格和预设的限额，可以了解各公司的竞争力及其对于预算管理的能力。同时也可以看出哪些公司能够有效地降低成本并保持质量标准不变。\", \"metadata\": {\"type\": \"text\", \"chunk_id\": 166, \"source\": \"output_piclegend.md\"}}"
                        "]}"
                    )
                }
            }
        }
    }
)
async def query_stream(
    request: QueryRequest = Body(
        ...,
        example={
            "question": "什么是工程造价",
            "num_docs": 2
        }
    )
) -> StreamingResponse:
    """
    流式查询接口

    Args:
        request (QueryRequest): 查询请求

    Returns:
        StreamingResponse: 流式响应

    example: 
        {"question": "什么是工程造价", "num_docs": 10}
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")

    async def answer_generator() -> AsyncGenerator[str, None]:
        try:
            logging.info(f"=== 开始流式查询，问题: {request.question}")
            
            result = await rag_service.process_single_query(
                question=request.question,
                num_docs=request.num_docs
            )
            logging.info(f"process_single_query 返回结果: success={result.get('success')}, num_contexts={len(result.get('contexts', []))}")
            
            contexts = result.get("contexts", [])
            question = request.question
            full_text = ""

            def get_page_content(c):
                if isinstance(c, dict):
                    return c.get('page_content', str(c))
                return str(c)

            # 构建上下文文本
            context_text = '\n'.join([get_page_content(c) for c in contexts])
            logging.info(f"上下文文本长度: {len(context_text)} 字符")
            
            logging.info("开始调用 LLM stream_predict...")
            chunk_count = 0
            async for chunk in rag_service.llm.stream_predict(
                context_text, question, messages
            ):
                full_text += chunk
                chunk_count += 1
                yield chunk + "\n"
            
            logging.info(f"LLM stream_predict 完成，共收到 {chunk_count} 个 chunk")

            yield "\n[END]\n"
            messages.append({"role": "assistant", "content": full_text})

            # ✅ 限制最多 10 条（即最近 5 轮对话）
            if len(messages) > 10:
                del messages[:len(messages) - 6]

            meta = {
                "text": full_text,
                "contexts": contexts
            }
            yield json.dumps(meta, ensure_ascii=False) + "\n"
            logging.info("流式查询完成")

        except Exception as e:
            logging.error(f"流式查询异常: {e}", exc_info=True)
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(answer_generator(), media_type="text/plain")

@router.post(
    "/api/v1/query/intent",
    summary="意图识别接口",
    description="""
    对用户输入的问题进行意图分类，判断问题属于哪种意图类别。
    
    - `knowledge_qa`：知识问答类型，使用 `/api/v1/query/stream` 接口
    - `price_recommendation`：从数据库查询，使用 `/api/v1/query/price` 接口
    """,
    responses={
        200: {
            "description": "意图识别成功",
            "content": {
                "application/json": {
                    "example": {
                        "intent": "knowledge_qa",
                        "question": "什么是工程造价"
                    }
                }
            }
        }
    }
)
async def classify_intent(
    request: IntentRequest = Body(
        ...,
        example={
            "question": "什么是工程造价"
        }
    )
):
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    try:
        intent = await rag_service.classify_intent(request.question)
        logging.info(f"识别到类型: {intent}")
        return {"intent": intent, "question": request.question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/api/v1/query/price",
    summary="价格推荐接口（流式）",
    description="""
    根据用户问题提供价格推荐，采用流式响应方式逐步返回推荐结果。
    
    - 仅需要 `question` 字段，注意这里提问时需要明确从什么渠道查询
    - 流式返回包括渠道识别、实体解析、价格推荐结果和元数据
    - 响应类型：text/plain
    """,
    responses={
        200: {
            "description": "价格推荐流式响应",
            "content": {
                "text/plain": {
                    "example": (
                        "识别到渠道类型: information_price\n"
                        "\n[END]\n"
                        "解析到实体结果: {\"materialName\": \"铝合金幕墙型材\"} \n"
                        "\n[END]\n"
                        "{\"text\": \"\", "
                        "\"intent\": \"price_recommendation\", "
                        "\"channel\": \"information_price\", "
                        "\"metadata\": {"
                            "\"channel\": \"information_price\", "
                            "\"entities\": {\"materialName\": \"铝合金幕墙型材\"}, "
                            "\"price_analysis\": {"
                                "\"kg\": {"
                                    "\"total_count\": 46, "
                                    "\"valid_count\": 46, "
                                    "\"price_range\": [23.0, 25.0], "
                                    "\"mean_price\": 24.02173913043478, "
                                    "\"median_price\": 24.0, "
                                    "\"recommend_kmeans\": {"
                                        "\"mode\": \"single\", "
                                        "\"prices\": [24.0], "
                                        "\"reason\": \"条件不足: silhouette=0.75, gap_ratio=0.06, 小簇=False\""
                                    "}"
                                "}"
                            "}, "
                            "\"md_table\": \"\""
                        "}, "
                        "\"success\": true}\n"
                    )
                }
            }
        }
    }
)
async def query_price(
    request: PriceRequest = Body(
        ...,
        example={
            "question": "从信息价查铝合金幕墙型材的价格"
        }
    )
) -> StreamingResponse:
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")

    async def price_answer_generator() -> AsyncGenerator[str, None]:
        import time
        try:
            # ✅ 初始化所有时间变量
            total_start_time = time.time()
            llm_start_time = None
            first_token_time = None
            llm_end_time = None
            ttft = 0.0  # ← 初始化 ttft
            input_tokens = 0  # ← 初始化 input_tokens
            token_count = 0  # ← 初始化 token_count
            llm_total_time = 0.0  # ← 初始化 llm_total_time
            
            full_text = ""
            
            # ========== 修改开始：智能渠道推断 ==========
            # 先进行实体抽取（用于辅助渠道推断）
            parsed_entities = await rag_service.extract_entities(request.question)
            
            # 渠道识别（传入已解析的实体辅助推断）
            channel_result, inference_detail = await rag_service.identify_channel(
                request.question, 
                parsed_entities
            )
            
            # 构建渠道识别响应（包含推断信息）
            channel_info = {
                "channel": channel_result.value,
                "confidence": inference_detail.confidence,
                "reason": inference_detail.reason,
                "matched_keywords": inference_detail.matched_keywords
            }
            
            # 如果是推断的（非明确指定），添加提示信息
            if inference_detail.confidence < 1.0:
                channel_info["inferred"] = True
                channel_info["suggestion"] = inference_detail.suggestion
            
            # 如果是 UNKNOWN 且置信度为0，才真正报错
            if channel_result == ChannelType.UNKNOWN and inference_detail.confidence == 0:
                yield f"识别到渠道类型: {json.dumps(channel_info, ensure_ascii=False)}\n"
                yield "\n[END]\n"
                
                error_meta = {
                    "text": "",
                    "intent": "price_recommendation",
                    "channel": "unknown",
                    "success": False,
                    "error_message": inference_detail.suggestion or "无法识别查询渠道，请提供渠道关键词"
                }
                yield json.dumps(error_meta, ensure_ascii=False) + "\n"
                return  # 直接返回，不继续后续处理
            
            print(f"识别到渠道类型: {channel_result.value}, 置信度: {inference_detail.confidence:.2f}")
            yield f"识别到渠道类型: {json.dumps(channel_info, ensure_ascii=False)}\n"
            yield "\n[END]\n"
            # ========== 修改结束 ==========
            
            # 实体解析部分（已在前面执行）

            yield f"解析到实体结果: {json.dumps(parsed_entities,ensure_ascii=False,indent=2)} "+" \\n"
            yield "\n[END]\n"

            # 再调用价格推荐的流式接口
            result = await rag_service.stream_price_recommendation(channel_result.value, parsed_entities)
            parsed_entities_str = result.get("parsed_entities", {})
            detail_answer = result.get("detail_answer", "")

            total_count = result.get("metadata", {}).get("total_count", 0)
            if total_count > 0 and result.get("success") == True:
                # 记录 LLM 生成开始时间
                llm_start_time = time.time()
                
                # 计算传给 LLM 的实际 token 数
                input_text = f"{parsed_entities_str}\n{detail_answer}\n{request.question}"
                # 粗略估算：中文约 1.5-2 字符/token，英文约 4 字符/token
                input_tokens = len(input_text) // 2  # 重新赋值
                
                token_count = 0  # 重置计数
                async for chunk in rag_service.llm.stream_price_predict(
                    parsed_entities_str, detail_answer, request.question, messages
                ):
                    # 记录首个 token 时间 (TTFT)
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = first_token_time - llm_start_time
                        rag_service.logger.info(f"⚡ TTFT (首个Token响应): {ttft:.3f}s")
                    
                    full_text += chunk
                    token_count += 1
                    yield chunk + "\n"

                # 记录 LLM 生成结束时间
                llm_end_time = time.time()
                llm_total_time = llm_end_time - llm_start_time
                
                yield "\n[END]\n"

                messages.append({"role": "assistant", "content": full_text})
                
                # 打印 LLM 性能统计（只有成功生成内容时才打印）
                if first_token_time is not None and token_count > 0:
                    log_llm_generation_performance(
                        logger=rag_service.logger,
                        input_tokens=input_tokens,
                        output_tokens=token_count,
                        ttft=ttft,
                        total_time=llm_total_time
                    )
                else:
                    rag_service.logger.warning("⚠️  LLM 未返回任何 token，跳过性能统计")
            # 获取流式处理的元数据
            metadata = rag_service.get_last_price_metadata()
            # 限制最多 10 条（即最近 5 轮对话）
            if len(messages) > 10:
                del messages[:len(messages) - 6]

            # 计算总耗时（仅用于日志）
            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            
            # 打印总体性能统计
            if llm_start_time and llm_end_time and first_token_time:
                log_total_request_performance(
                    logger=rag_service.logger,
                    total_time=total_time,
                    llm_time=llm_end_time - llm_start_time,
                    ttft=first_token_time - llm_start_time
                )
            
            meta = {
                "text": full_text,
                "intent": "price_recommendation",
                "channel": channel_result.value,
                "metadata": metadata,
                "success": metadata.get("success", False)
            }
            
            # 如果有价格数据，添加到响应中
            if "price_data" in metadata:
                meta["price_data"] = metadata["price_data"]
            
            yield json.dumps(meta, ensure_ascii=False) + "\n"
            
        except Exception as e:
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(price_answer_generator(), media_type="text/plain")


# 当问题并非直接的知识库问答 or 数据库问答时，采用普通大模型结合上下文进行问答
# @router.post("/api/v1/query/question")
# async def query_stream(request: QueryRequest):
#     """
#     流式查询接口

#     Args:
#         request (QueryRequest): 查询请求

#     Returns:
#         StreamingResponse: 流式响应

#     example: 
#         {"question": "什么是工程造价", "num_docs": 10}
#     """
#     if not rag_service:
#         raise HTTPException(status_code=503, detail="RAG服务未初始化")

#     async def answer_generator() -> AsyncGenerator[str, None]:
#         try:

#             question = request.question
#             full_text = ""
#             messages.append({"role": "user", "content": question})
#             async for chunk in rag_service.llm.predict(messages):
#                 full_text += chunk
#                 yield chunk + "\n"

#             yield "\n[END]\n"
#             messages.append({"role": "assistant", "content": full_text})

#            # ✅ 限制最多 10 条（即最近 5 轮对话）
#             if len(messages) > 10:
#                 del messages[:len(messages) - 6]


#             meta = {
#                 "text": full_text,
#                 "contexts": ""           # 原样返回，不做任何路径处理
#             }
#             yield json.dumps(meta, ensure_ascii=False) + "\n"

#         except Exception as e:
#             yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

#     return StreamingResponse(answer_generator(), media_type="text/plain")

@router.get("/")
async def root():
    """
    根路径接口

    返回服务运行状态信息。

    Returns:
        dict: 服务运行状态信息。
    """
    return {
        "message": "Easy-RAG API服务运行中",
        "version": "1.0.0",
        "status": "healthy"
    }

# @router.get("/health")
# async def health_check():
#     """
#     健康检查接口

#     返回服务的健康状态信息。

#     Returns:
#         dict: 服务健康状态信息。
#     """
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "service": "Easy-RAG API"
#     }

@router.get(
    "/api/v1/task_status",
    summary="查询任务状态接口",
    description="""
    查询任务状态接口（上传文件和批量删除文件两个接口会返回 task_id，由于这两个操作需要更新知识库所以耗时较久，可通过这个接口查询任务状态）。
    
    - 根据任务 ID 查询任务的当前状态。
    - 返回 JSON 格式，包括 task_id 和当前状态。
    """,
    responses={
        200: {
            "description": "任务状态信息",
            "content": {
                "application/json": {
                    "example": {
                        "task_id": "8e1daab8-3789-49ab-b91c-9f266f961c4f",
                        "status": "pending"
                    }
                }
            }
        }
    }
)
async def get_task_status(task_id: str):
    status = task_status_dict.get(task_id, None)
    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"task_id": task_id, "status": status}


@router.get(
    "/api/v1/service/status",
    summary="查询服务状态接口",
    description="""
    查询当前服务状态，包括知识库重建状态。
    
    - 可用于前端判断知识库是否正在更新
    - 如果 is_rebuilding 为 true，知识问答功能将不可用
    - 价格推荐功能不受知识库重建影响，始终可用
    """,
    responses={
        200: {
            "description": "服务状态信息",
            "content": {
                "application/json": {
                    "example": {
                        "is_rebuilding": False,
                        "rebuild_progress": {
                            "stage": "idle",
                            "message": "就绪",
                            "percent": 100
                        },
                        "retriever_ready": True,
                        "llm_ready": True,
                        "corpus_count": 42
                    }
                }
            }
        }
    }
)
async def get_service_status():
    """
    获取当前服务状态
    
    Returns:
        dict: 服务状态信息
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    return rag_service.get_service_status()

@router.post(
    "/api/v1/query/price/direct",
    summary="直接价格查询接口",
    description="""
    直接价格查询接口（流式）
    
    该接口接收材料的完整信息，使用 materialName、province 和 city 字段查询数据库数据，目前只根据提供数据中的这三个字段筛选。
    
    - 支持根据材料信息列表查询价格数据
    - 返回结果包括 answer 文本、意图、渠道、元数据、价格数据等
    - 如果未查询到符合条件的价格数据，会返回 success=false，并给出错误信息
    - 响应类型：text/plain
    """,
    responses={
        200: {
            "description": "直接价格查询结果",
            "content": {
                "text/plain": {
                    "example": {
                        "text": "广东省深圳市铝合金门窗型材最近三年平均价格约为 22 元/kg",
                        "intent": "price_recommendation",
                        "channel": "information_price",
                        "metadata": {
                            "channel": "information_price",
                            "entities": {"materialName": "铝合金门窗型材"},
                            "price_analysis": {
                                "kg": {
                                    "total_count": 10,
                                    "valid_count": 10,
                                    "price_range": [21.0, 23.0],
                                    "mean_price": 22.0,
                                    "median_price": 22.0,
                                    "recommend_kmeans": {
                                        "mode": "single",
                                        "prices": [22.0],
                                        "reason": "条件不足: silhouette=0.7, gap_ratio=0.05, 小簇=False"
                                    }
                                }
                            },
                            "md_table": ""
                        },
                        "success": True,
                        "price_data": []
                    }
                }
            }
        }
    }
)
async def query_price_direct(
    request: DirectPriceQueryRequest = Body(
        ...,
        example={
            "datatype": "informaterial",
            "question": "查找铝合金门窗型材的价格数据",
            "list": [{
                "id": 6,
                "materialId": "1369392Jvbw",
                "accountId": "testoxidmwedxdkseucdnvksfnzmdfnzd",
                "materialName": "铝合金门窗型材",
                "materialCode": "01512001",
                "checkState": 1,
                "belongDataPool": 2,
                "domain": "",
                "categoryOneLevel": "",
                "categoryTwoLevel": "",
                "categoryThreeLevel": "",
                "standardCategoryName": None,
                "standardCategoryCode": None,
                "standardCategoryUnit": None,
                "standardCategoryConfidence": None,
                "standardFeatures": None,
                "label": "铝合金门窗型材,银白氧化,kg,深圳市造价站,深圳市造价站,广东省,深圳市",
                "price": 22.0,
                "taxRate": 0.0,
                "materialModelSpec": "银白氧化",
                "releaseDepartment": "",
                "releaseTime": "2018-08-01",
                "releaseDate": None,
                "unit": "kg",
                "province": "",
                "city": "",
                "materialDescribe": "",
                "savePath": "",
                "createTime": "2025-10-24 15:19:00",
                "createTimeDate": "2025-07-31",
                "updateTime": "2025-12-01 16:51:24",
                "updateTimeDate": "2025-07-31",
                "categoryOneLevelName": None,
                "categoryTwoLevelName": None,
                "provinceId": "25390907-898d-4d62-9e07-f1c9cb4eaaae",
                "cityId": "76968431-d0e0-47ab-9ae9-31aff28237f3"
            }]
        }
    )
) -> StreamingResponse:

    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")

    async def direct_price_answer_generator() -> AsyncGenerator[str, None]:

        try:
            # ---------- 1️⃣ 基础校验 ----------
            if not request.list:
                yield json.dumps(
                    {"success": False, "error_message": "材料列表不能为空"},
                    ensure_ascii=False
                ) + "\n"
                return

            material_item = request.list[0]

            # ---------- 2️⃣ 渠道说明（与 query_price 对齐） ----------
            yield "识别到渠道类型: information_price\n"
            yield "\n[END]\n"

            # ---------- 3️⃣ 实体信息 ----------
            entities = {
                "materialName": material_item.materialName,
                "province": material_item.province,
                "city": material_item.city
            }

            yield (
                "解析到实体结果: "
                + json.dumps(entities, ensure_ascii=False, indent=2)
                + "\n"
            )
            yield "\n[END]\n"

            # ---------- 4️⃣ 核心查询逻辑 ----------
            result = await rag_service.process_direct_price_query(
                request.datatype,
                material_item,
                request.question
            )

            if not result.get("success", False):
                yield json.dumps(
                    {
                        "text": "",
                        "intent": "price_recommendation",
                        "channel": "information_price",
                        "metadata": result.get("metadata", {}),
                        "success": False,
                        "error_message": result.get(
                            "error_message",
                            "未查询到符合条件的价格数据"
                        )
                    },
                    ensure_ascii=False
                ) + "\n"
                return

            # ---------- 5️⃣ 如需 LLM，可在此流式输出 ----------
            # 如果你 direct 不想走 LLM，这一段可以直接删掉
            full_text = ""
            if result.get("detail_answer"):
                async for chunk in rag_service.llm.stream_price_predict(
                    entities,
                    result.get("detail_answer", ""),
                    request.question,
                    messages=[]
                ):
                    full_text += chunk
                    yield chunk + "\n"

                yield "\n[END]\n"
            else:
                full_text = result.get("answer", "")

            # ---------- 6️⃣ 最终 meta ----------
            meta = {
                "text": full_text or result.get("answer", ""),
                "intent": "price_recommendation",
                "channel": "information_price",
                "metadata": result.get("metadata", {}),
                "success": True
            }

            if "price_data" in result:
                meta["price_data"] = result["price_data"]

            yield json.dumps(meta, ensure_ascii=False) + "\n"

        except Exception as e:
            logging.exception("直接价格查询（流式）失败")
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        direct_price_answer_generator(),
        media_type="text/plain"
    )


# ========== 渐进式实体补全 - 对话式查询接口 ==========

class DialogueQueryRequest(BaseModel):
    """对话式查询请求"""
    session_id: Optional[str] = Field(None, description="会话ID，首次请求可为空")
    user_input: str = Field(..., description="用户输入")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": None,
                "user_input": "查一下钢筋的价格"
            }
        }


class DialogueExecuteRequest(BaseModel):
    """执行对话查询请求"""
    session_id: str = Field(..., description="会话ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc12345"
            }
        }


@router.post(
    "/api/v1/query/dialogue",
    summary="对话式价格查询（渐进式实体补全）",
    description="""
    支持多轮对话的渐进式实体补全价格查询接口。
    
    特点：
    - 无需一次性提供所有查询条件
    - 系统会引导用户逐步补充必要信息
    - 支持会话保持，可跨请求继续对话
    - 会话有效期10分钟
    
    使用流程：
    1. 首次调用：session_id为空，提供用户输入
    2. 根据返回的响应，补充缺失信息
    3. 携带session_id继续对话，直到 can_query=true
    4. 调用 /api/v1/query/dialogue/execute 执行查询
    """,
    responses={
        200: {
            "description": "对话状态响应",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "abc12345",
                        "status": "collecting",
                        "entities": {"materialName": "钢筋"},
                        "channel": "information_price",
                        "channel_info": {
                            "channel": "information_price",
                            "confidence": 0.5,
                            "reason": "未明确指定渠道，默认查询信息价",
                            "inferred": True
                        },
                        "is_complete": False,
                        "can_query": False,
                        "response_text": "请问是哪个省份的材料？（如：广东省、江苏省）",
                        "next_question": {
                            "field": "province",
                            "label": "省份",
                            "prompt": "请问是哪个省份的材料？（如：广东省、江苏省）"
                        },
                        "quick_options": [
                            {"text": "广东省", "value": "广东省"},
                            {"text": "江苏省", "value": "江苏省"}
                        ]
                    }
                }
            }
        }
    }
)
async def dialogue_query(request: DialogueQueryRequest):
    """
    对话式价格查询接口
    
    支持渐进式收集查询条件，通过多轮对话完成价格查询
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    try:
        result = await rag_service.process_dialogue_query(
            session_id=request.session_id,
            user_input=request.user_input
        )
        return result
    except Exception as e:
        logging.exception("对话式查询失败")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post(
    "/api/v1/query/dialogue/execute",
    summary="执行对话查询",
    description="""
    执行对话会话中已收集条件的查询。
    
    当 dialogue 接口返回 can_query=true 时，
    可调用此接口执行实际的价格查询。
    """,
    responses={
        200: {
            "description": "查询结果",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "session_id": "abc12345",
                        "entities": {
                            "materialName": "钢筋",
                            "province": "广东省",
                            "city": "深圳市"
                        },
                        "price_result": {
                            "answer": "...",
                            "success": True,
                            "total_count": 156
                        }
                    }
                }
            }
        }
    }
)
async def dialogue_execute(request: DialogueExecuteRequest):
    """
    执行对话查询
    
    使用已收集的实体执行价格查询
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    try:
        result = await rag_service.execute_dialogue_query(
            session_id=request.session_id
        )
        return result
    except Exception as e:
        logging.exception("执行对话查询失败")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.get(
    "/api/v1/query/dialogue/{session_id}",
    summary="获取对话会话状态",
    description="获取指定会话ID的当前状态、已收集实体等信息"
)
async def get_dialogue_session(session_id: str):
    """获取对话会话状态"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    session = rag_service.get_dialogue_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    
    return session


@router.delete(
    "/api/v1/query/dialogue/{session_id}",
    summary="清除对话会话",
    description="手动清除指定会话，释放资源"
)
async def clear_dialogue_session(session_id: str):
    """清除对话会话"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    rag_service.clear_dialogue_session(session_id)
    return {"success": True, "message": f"会话 {session_id} 已清除"}