"""
MCP Server - 处理工具调用和Agent协调
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import logging

from .tools import MaterialPriceTools, ToolResult, ToolStatus

logger = logging.getLogger("easy_rag_api")


@dataclass
class AgentThought:
    """Agent思考过程记录"""
    step: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str = ""
    next_step: str = ""


class MCPServer:
    """
    MCP Server
    - 注册和管理工具
    - 处理工具调用
    - 协调Agent执行流程
    """
    
    def __init__(self, rag_service):
        """
        初始化 MCP Server
        
        Args:
            rag_service: RAGService 实例
        """
        self.rag_service = rag_service
        self.tools = MaterialPriceTools(rag_service)
        self.logger = logging.getLogger("easy_rag_api")
        
        # 工具映射表
        self._tool_map: Dict[str, Callable] = {
            "quick_search_materials": self.tools.quick_search_materials,
            "query_price_data": self.tools.query_price_data,
            "analyze_prices": self.tools.analyze_prices,
            "get_material_candidates": self.tools.get_material_candidates,
            "extract_entities": self.tools.extract_entities,
            "identify_channel": self.tools.identify_channel,
        }
    
    # ============================================================================
    # 工具管理
    # ============================================================================
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取所有可用工具的定义"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "required": tool.required
            }
            for tool in self.tools.available_tools
        ]
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        调用指定工具
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            
        Returns:
            ToolResult: 工具执行结果
        """
        if tool_name not in self._tool_map:
            return ToolResult(
                status=ToolStatus.ERROR,
                message=f"未知工具: {tool_name}"
            )
        
        try:
            tool_func = self._tool_map[tool_name]
            result = await tool_func(**parameters)
            return result
        except Exception as e:
            self.logger.error(f"工具调用失败 {tool_name}: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                message=f"工具执行失败: {str(e)}"
            )
    
    # ============================================================================
    # Agent 协调（用于多轮对话）
    # ============================================================================
    
    async def handle_first_query(
        self, 
        user_input: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理用户首次查询
        
        流程：
        1. 提取实体
        2. 识别渠道
        3. 快速搜索材料候选
        4. 返回补充建议和候选材料
        
        Args:
            user_input: 用户输入
            session_id: 会话ID（可选）
            
        Returns:
            包含首次查询结果的响应
        """
        self.logger.info(f"[MCP Agent] 处理首次查询: {user_input}")
        
        thoughts: List[AgentThought] = []
        
        # Step 1: 提取实体
        step = 1
        thought1 = AgentThought(
            step=step,
            thought="需要首先理解用户意图，提取关键实体信息",
            action="extract_entities",
            action_input={"question": user_input}
        )
        
        entity_result = await self.call_tool("extract_entities", {"question": user_input})
        thought1.observation = f"提取到实体: {entity_result.data}"
        thought1.next_step = "识别查询渠道" if entity_result.status == ToolStatus.SUCCESS else "请求用户提供更多信息"
        thoughts.append(thought1)
        
        if entity_result.status != ToolStatus.SUCCESS:
            return self._build_response(
                success=False,
                message="无法提取查询实体，请提供更明确的问题",
                thoughts=thoughts,
                suggested_actions=["请明确指定材料名称，如：钢筋、水泥"]
            )
        
        entities = entity_result.data
        material_name = entities.get("materialName", "")
        
        # Step 2: 识别渠道
        step += 1
        thought2 = AgentThought(
            step=step,
            thought="用户可能指定了查询渠道，需要识别",
            action="identify_channel",
            action_input={"question": user_input}
        )
        
        channel_result = await self.call_tool("identify_channel", {"question": user_input})
        thought2.observation = f"识别到渠道: {channel_result.data.get('channel', 'unknown')}"
        thought2.next_step = "快速搜索材料候选"
        thoughts.append(thought2)
        
        channel = channel_result.data.get("channel", "information_price")
        
        # Step 3: 快速搜索材料候选
        step += 1
        thought3 = AgentThought(
            step=step,
            thought=f"用户查询材料: {material_name}，需要获取相关候选",
            action="quick_search_materials",
            action_input={"keyword": material_name, "limit": 10}
        )
        
        search_result = await self.call_tool(
            "quick_search_materials", 
            {"keyword": material_name, "limit": 10}
        )
        thought3.observation = f"找到 {len(search_result.data.get('candidates', []))} 个候选"
        thought3.next_step = "返回结果给用户"
        thoughts.append(thought3)
        
        # 构建响应
        candidates = search_result.data.get("candidates", [])
        related_keywords = search_result.data.get("related_keywords", [])
        missing_fields = entity_result.metadata.get("missing_fields", [])
        
        # 生成自然语言回复
        response_text = self._generate_first_response(
            material_name=material_name,
            candidates=candidates,
            missing_fields=missing_fields,
            related_keywords=related_keywords,
            channel=channel
        )
        
        return self._build_response(
            success=True,
            message=response_text,
            data={
                "entities": entities,
                "channel": channel,
                "candidates": candidates,
                "related_keywords": related_keywords,
                "missing_fields": missing_fields
            },
            thoughts=thoughts,
            suggested_actions=entity_result.suggested_next_steps,
            session_id=session_id
        )
    
    async def handle_follow_up_query(
        self,
        user_input: str,
        session_id: str,
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理用户后续查询（补充信息或新查询）
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            conversation_context: 对话上下文（包含已收集的实体等）
            
        Returns:
            包含查询结果的响应
        """
        self.logger.info(f"[MCP Agent] 处理后续查询: {user_input}")
        
        thoughts: List[AgentThought] = []
        
        # 获取已收集的实体
        collected_entities = conversation_context.get("entities", {})
        
        # Step 1: 从用户输入中提取新实体
        step = 1
        thought1 = AgentThought(
            step=step,
            thought="从用户补充输入中提取新的实体信息",
            action="extract_entities",
            action_input={"question": user_input}
        )
        
        entity_result = await self.call_tool("extract_entities", {"question": user_input})
        thought1.observation = f"提取到新实体: {entity_result.data}"
        thoughts.append(thought1)
        
        # 合并实体（新值覆盖旧值）
        new_entities = entity_result.data
        merged_entities = {**collected_entities, **new_entities}
        
        # Step 2: 检查是否具备查询条件
        step += 1
        has_material = bool(merged_entities.get("materialName"))
        has_province = bool(merged_entities.get("province"))
        
        if not has_material:
            # 仍然缺少材料名称
            return self._build_response(
                success=False,
                message="请告诉我您想查询什么材料的价格？",
                data={"entities": merged_entities},
                thoughts=thoughts,
                suggested_actions=["提供材料名称，如：钢筋、水泥、铝合金"]
            )
        
        # Step 3: 执行价格查询
        thought2 = AgentThought(
            step=step,
            thought="实体信息已收集，可以执行价格查询",
            action="query_price_data",
            action_input={
                "channel": conversation_context.get("channel", "information_price"),
                **{k.replace("Name", "_name").replace("Spec", "_spec"): v 
                   for k, v in merged_entities.items()}
            }
        )
        
        query_result = await self.call_tool("query_price_data", {
            "channel": conversation_context.get("channel", "information_price"),
            "material_name": merged_entities.get("materialName"),
            "province": merged_entities.get("province"),
            "city": merged_entities.get("city"),
            "material_model_spec": merged_entities.get("materialModelSpec"),
            "brand": merged_entities.get("brand")
        })
        
        thought2.observation = f"查询结果: {query_result.message}"
        thoughts.append(thought2)
        
        # Step 4: 如果查询成功且有数据，进行深度分析
        if query_result.status == ToolStatus.SUCCESS and query_result.data.get("total_count", 0) > 0:
            step += 1
            thought3 = AgentThought(
                step=step,
                thought="查询成功，需要对价格数据进行深度分析",
                action="analyze_prices",
                action_input={"price_data": query_result.data.get("price_data", [])}
            )
            
            price_data = query_result.data.get("price_data", [])
            analysis_result = await self.call_tool("analyze_prices", {"price_data": price_data})
            thought3.observation = f"分析完成: {analysis_result.message}"
            thoughts.append(thought3)
            
            # 构建包含分析结果的响应
            response_text = self._generate_analysis_response(
                entities=merged_entities,
                total_count=query_result.data.get("total_count", 0),
                analysis=analysis_result.data
            )
        else:
            # 查询失败或无数据
            response_text = query_result.message
        
        return self._build_response(
            success=query_result.status == ToolStatus.SUCCESS,
            message=response_text,
            data={
                "entities": merged_entities,
                "query_result": query_result.to_dict(),
                "analysis": analysis_result.to_dict() if 'analysis_result' in locals() else None
            },
            thoughts=thoughts,
            suggested_actions=query_result.suggested_next_steps,
            session_id=session_id
        )
    
    # ============================================================================
    # 响应生成
    # ============================================================================
    
    def _generate_first_response(
        self,
        material_name: str,
        candidates: List[Dict[str, Any]],
        missing_fields: List[str],
        related_keywords: List[str],
        channel: str
    ) -> str:
        """生成首次查询的自然语言回复"""
        lines = []
        
        # 1. 确认收到的材料名称
        if material_name:
            lines.append(f"收到，您想查询「{material_name}」的价格信息。")
        
        # 2. 渠道信息
        channel_names = {
            "information_price": "信息价",
            "manufacturer_price": "厂商报价",
            "zc_price": "智诚信息价"
        }
        lines.append(f"查询渠道：{channel_names.get(channel, '信息价')}")
        
        # 3. 候选材料
        if candidates:
            exact_match = [c for c in candidates if c.get("exact_match")]
            if exact_match:
                lines.append(f"\n✓ 找到精确匹配的材料：{exact_match[0]['name']}")
            else:
                lines.append(f"\n为您找到以下相关材料：")
                for i, cand in enumerate(candidates[:5], 1):
                    lines.append(f"  {i}. {cand['name']}")
        
        # 4. 相关关键词建议
        if related_keywords:
            lines.append(f"\n💡 相关关键词：{', '.join(related_keywords[:3])}")
        
        # 5. 缺失信息提示
        if missing_fields:
            field_names = {
                "province": "省份",
                "city": "城市",
                "materialModelSpec": "规格型号"
            }
            missing_labels = [field_names.get(f, f) for f in missing_fields]
            lines.append(f"\n📋 为了给您提供准确的价格信息，还需要：")
            for label in missing_labels:
                lines.append(f"  • {label}")
        
        return "\n".join(lines)
    
    def _generate_analysis_response(
        self,
        entities: Dict[str, Any],
        total_count: int,
        analysis: Dict[str, Any]
    ) -> str:
        """生成包含分析结果的回复"""
        lines = []
        
        material = entities.get("materialName", "")
        province = entities.get("province", "")
        city = entities.get("city", "")
        
        location = ""
        if city:
            location = f"{province}{city}"
        elif province:
            location = province
        else:
            location = "全国"
        
        lines.append(f"📊 「{material}」在{location}的价格分析：")
        lines.append(f"共查询到 {total_count} 条价格数据\n")
        
        # 分析结果
        results = analysis.get("results", {})
        most_common_unit = analysis.get("most_common_unit", "")
        
        if most_common_unit and most_common_unit in results:
            unit_analysis = results[most_common_unit]
            recommend = unit_analysis.get("recommend_kmeans", {})
            
            if recommend.get("mode") == "two-tier":
                prices = recommend.get("prices", [])
                lines.append(f"💰 推荐价格（两档）：")
                lines.append(f"  • 低价位：{prices[0]:.2f} 元/{most_common_unit}")
                lines.append(f"  • 高价位：{prices[1]:.2f} 元/{most_common_unit}")
            else:
                prices = recommend.get("prices", [])
                if prices:
                    lines.append(f"💰 推荐价格：{prices[0]:.2f} 元/{most_common_unit}")
            
            lines.append(f"\n📈 价格区间：{unit_analysis.get('price_range', ['-', '-'])[0]:.2f} - {unit_analysis.get('price_range', ['-', '-'])[1]:.2f}")
            lines.append(f"📊 平均价格：{unit_analysis.get('mean_price', 0):.2f}")
        
        return "\n".join(lines)
    
    def _build_response(
        self,
        success: bool,
        message: str,
        data: Dict[str, Any] = None,
        thoughts: List[AgentThought] = None,
        suggested_actions: List[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """构建标准响应格式"""
        return {
            "success": success,
            "message": message,
            "data": data or {},
            "thoughts": [
                {
                    "step": t.step,
                    "thought": t.thought,
                    "action": t.action,
                    "observation": t.observation,
                    "next_step": t.next_step
                }
                for t in (thoughts or [])
            ],
            "suggested_actions": suggested_actions or [],
            "session_id": session_id
        }
