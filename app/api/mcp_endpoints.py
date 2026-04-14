"""
MCP 相关 API 端点
提供Agent对话和工具调用接口
"""

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional, Dict, Any
import json
import logging

from app.schemas.rag import QueryRequest
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger("easy_rag_api")

# MCP Server 和 Agent 实例（由 main.py 注入）
mcp_server = None
material_price_agent = None


class AgentQueryRequest(BaseModel):
    """Agent查询请求"""
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID，首次可为空")


class AgentQueryResponse(BaseModel):
    """Agent查询响应"""
    success: bool
    session_id: str
    state: str
    turn_count: int
    message: str
    data: Dict[str, Any]
    suggested_actions: list


class ToolCallRequest(BaseModel):
    """工具调用请求"""
    tool_name: str = Field(..., description="工具名称")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="工具参数")


# ================================================================================
# Agent 对话接口
# ================================================================================

@router.post(
    "/api/v1/agent/chat",
    response_model=AgentQueryResponse,
    summary="Agent对话接口",
    description="""
    基于MCP工具的Agent对话接口，支持多轮对话。
    
    - 首次调用可不传 session_id，系统会自动创建
    - 后续调用需携带返回的 session_id 以保持对话上下文
    - Agent会自动决定调用哪些工具来完成查询
    """,
    responses={
        200: {
            "description": "Agent响应",
            "content": {
                "application/json": {
                    "examples": [
                        {
                            "success": True,
                            "session_id": "abc123",
                            "state": "collecting_entities",
                            "turn_count": 1,
                            "message": "收到，您想查询「钢筋」的价格信息...",
                            "data": {
                                "entities": {"materialName": "钢筋"},
                                "candidates": ["钢筋HRB400", "钢筋HRB500"]
                            },
                            "suggested_actions": ["广东省", "江苏省", "北京市"]
                        }
                    ]
                }
            }
        }
    }
)
async def agent_chat(request: AgentQueryRequest):
    """
    Agent对话接口
    
    支持材料价格查询的完整多轮对话流程
    """
    if not material_price_agent:
        raise HTTPException(status_code=503, detail="Agent服务未初始化")
    
    try:
        logger.info(f"[Agent API] 收到消息: session={request.session_id}, message={request.message}")
        
        result = await material_price_agent.process_message(
            user_message=request.message,
            session_id=request.session_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Agent处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@router.post(
    "/api/v1/agent/chat/stream",
    summary="Agent流式对话接口",
    description="""
    Agent流式对话接口，以SSE格式返回思考过程和最终结果。
    
    - 可以实时观察Agent的思考过程（thoughts）
    - 可以看到每一步的工具调用和返回结果
    - 适合需要展示Agent推理过程的场景
    
    事件类型：
    - start: 开始处理
    - thought: LLM 当前思考内容（可能包含 tool_calls 意向）
    - tool_call: 开始调用某个工具
    - observation: 工具执行结果
    - final: 最终回复和完整数据
    - error: 处理异常
    """
)
async def agent_chat_stream(request: AgentQueryRequest):
    """Agent流式对话接口（SSE格式）"""
    if not material_price_agent:
        raise HTTPException(status_code=503, detail="Agent服务未初始化")
    
    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in material_price_agent.process_message_stream(
                user_message=request.message,
                session_id=request.session_id
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
                
        except Exception as e:
            logger.error(f"流式处理失败: {e}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }, ensure_ascii=False) + "\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain"
    )


# ================================================================================
# MCP 工具接口
# ================================================================================

@router.get(
    "/api/v1/mcp/tools",
    summary="获取可用工具列表",
    description="获取所有可用的MCP工具定义"
)
async def get_mcp_tools():
    """获取可用工具列表"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务未初始化")
    
    try:
        tools = mcp_server.get_available_tools()
        return {
            "success": True,
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        logger.error(f"获取工具列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/v1/mcp/tools/call",
    summary="调用MCP工具",
    description="直接调用指定的MCP工具"
)
async def call_mcp_tool(request: ToolCallRequest):
    """调用MCP工具"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务未初始化")
    
    try:
        logger.info(f"[MCP API] 调用工具: {request.tool_name}, 参数: {request.parameters}")
        
        result = await mcp_server.call_tool(
            tool_name=request.tool_name,
            parameters=request.parameters
        )
        
        return {
            "success": result.status == "success",
            "result": result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"工具调用失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================================
# 首次查询接口（简化版）
# ================================================================================

@router.post(
    "/api/v1/mcp/first-query",
    summary="首次查询接口",
    description="""
    处理用户的首次查询，返回候选材料和补充建议。
    
    这是Agent对话的简化入口，适合只需要首次查询功能的场景。
    """
)
async def mcp_first_query(request: AgentQueryRequest):
    """首次查询接口"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务未初始化")
    
    try:
        result = await mcp_server.handle_first_query(
            user_input=request.message,
            session_id=request.session_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"首次查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================================
# 会话管理接口
# ================================================================================

@router.get(
    "/api/v1/agent/session/{session_id}",
    summary="获取会话状态",
    description="获取指定会话的当前状态和收集的实体信息"
)
async def get_agent_session(session_id: str):
    """获取会话状态"""
    if not material_price_agent:
        raise HTTPException(status_code=503, detail="Agent服务未初始化")
    
    try:
        context = material_price_agent.contexts.get(session_id)
        if not context:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return {
            "success": True,
            "session_id": session_id,
            "state": context.state.value,
            "turn_count": context.turn_count,
            "entities": context.collected_entities,
            "channel": context.channel,
            "selected_candidate": context.selected_candidate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/api/v1/agent/session/{session_id}",
    summary="清除会话",
    description="清除指定的Agent会话"
)
async def clear_agent_session(session_id: str):
    """清除会话"""
    if not material_price_agent:
        raise HTTPException(status_code=503, detail="Agent服务未初始化")
    
    try:
        if session_id in material_price_agent.contexts:
            del material_price_agent.contexts[session_id]
        
        return {
            "success": True,
            "message": f"会话 {session_id} 已清除"
        }
        
    except Exception as e:
        logger.error(f"清除会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================================
# 初始化函数（由 main.py 调用）
# ================================================================================

def init_mcp_endpoints(rag_service):
    """
    初始化MCP端点
    
    Args:
        rag_service: RAGService 实例
    """
    global mcp_server, material_price_agent
    
    from app.mcp.server import MCPServer
    from app.mcp.agent import MaterialPriceAgent
    
    mcp_server = MCPServer(rag_service)
    material_price_agent = MaterialPriceAgent(mcp_server, rag_service.llm)
    
    logger.info("MCP端点初始化完成")
