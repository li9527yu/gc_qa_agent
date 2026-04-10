"""
Agent 实现 - 基于MCP工具的多轮对话Agent
处理材料价格查询的完整对话流程
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from .server import MCPServer
from .tools import ToolStatus

logger = logging.getLogger("easy_rag_api")


class AgentState(str, Enum):
    """Agent状态"""
    IDLE = "idle"                    # 空闲
    COLLECTING_ENTITIES = "collecting_entities"  # 收集实体
    CONFIRMING_CANDIDATE = "confirming_candidate"  # 确认候选材料
    READY_TO_QUERY = "ready_to_query"  # 准备查询
    QUERYING = "querying"            # 查询中
    ANALYZING = "analyzing"          # 分析中
    COMPLETED = "completed"          # 完成


@dataclass
class AgentContext:
    """Agent上下文"""
    session_id: str
    state: AgentState = AgentState.IDLE
    collected_entities: Dict[str, Any] = field(default_factory=dict)
    channel: str = "information_price"
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected_candidate: Optional[str] = None
    query_result: Optional[Dict[str, Any]] = None
    turn_count: int = 0
    
    def update_entities(self, new_entities: Dict[str, Any]):
        """更新实体（合并）"""
        for key, value in new_entities.items():
            if value is not None and value != "":
                self.collected_entities[key] = value
    
    def is_ready_to_query(self) -> bool:
        """检查是否具备查询条件"""
        return bool(self.collected_entities.get("materialName"))
    
    def get_missing_fields(self) -> List[str]:
        """获取缺失的关键字段"""
        required = ["materialName", "province", "city"]
        return [f for f in required if not self.collected_entities.get(f)]


class MaterialPriceAgent:
    """
    材料价格查询 Agent
    
    支持多轮对话，自主决策调用哪些工具
    """
    
    def __init__(self, mcp_server: MCPServer):
        """
        初始化 Agent
        
        Args:
            mcp_server: MCP Server 实例
        """
        self.mcp_server = mcp_server
        self.logger = logging.getLogger("easy_rag_api")
        
        # 会话上下文存储
        self.contexts: Dict[str, AgentContext] = {}
    
    async def process_message(
        self, 
        user_message: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理用户消息（主入口）
        
        Args:
            user_message: 用户消息
            session_id: 会话ID（首次可为空）
            
        Returns:
            Agent响应
        """
        # 获取或创建上下文
        context = self._get_or_create_context(session_id)
        context.turn_count += 1
        
        self.logger.info(f"[Agent] 处理消息: session={context.session_id}, turn={context.turn_count}, state={context.state}")
        
        # 根据当前状态决定处理逻辑
        if context.turn_count == 1 or context.state == AgentState.IDLE:
            # 首次查询
            return await self._handle_first_turn(context, user_message)
        elif context.state == AgentState.CONFIRMING_CANDIDATE:
            # 确认候选材料
            return await self._handle_candidate_confirmation(context, user_message)
        elif context.state == AgentState.COLLECTING_ENTITIES:
            # 补充实体信息
            return await self._handle_entity_collection(context, user_message)
        elif context.state == AgentState.READY_TO_QUERY:
            # 准备查询，用户可能确认或修改
            return await self._handle_ready_state(context, user_message)
        elif context.state in [AgentState.COMPLETED, AgentState.QUERYING]:
            # 已完成或需要新查询
            return await self._handle_new_or_followup(context, user_message)
        else:
            # 默认处理
            return await self._handle_default(context, user_message)
    
    async def _handle_first_turn(
        self, 
        context: AgentContext, 
        user_message: str
    ) -> Dict[str, Any]:
        """
        处理第一轮对话（首次查询）
        
        流程：
        1. 提取实体
        2. 识别渠道
        3. 搜索材料候选
        4. 返回候选和补充建议
        """
        self.logger.info(f"[Agent] 首次查询处理: {user_message}")
        
        # Step 1: 提取实体
        entity_result = await self.mcp_server.call_tool(
            "extract_entities", 
            {"question": user_message}
        )
        
        if entity_result.status != ToolStatus.SUCCESS:
            context.state = AgentState.COLLECTING_ENTITIES
            return self._build_agent_response(
                context,
                "您好！我可以帮您查询材料价格。请告诉我您想查询什么材料？",
                suggested_actions=["钢筋", "水泥", "混凝土", "铝合金"]
            )
        
        # 更新上下文
        context.update_entities(entity_result.data)
        
        # Step 2: 识别渠道
        channel_result = await self.mcp_server.call_tool(
            "identify_channel",
            {"question": user_message}
        )
        if channel_result.status == ToolStatus.SUCCESS:
            context.channel = channel_result.data.get("channel", "information_price")
        
        material_name = context.collected_entities.get("materialName", "")
        
        # Step 3: 搜索材料候选
        search_result = await self.mcp_server.call_tool(
            "quick_search_materials",
            {"keyword": material_name, "limit": 10}
        )
        
        if search_result.status == ToolStatus.SUCCESS:
            context.candidates = search_result.data.get("candidates", [])
        
        # Step 4: 判断是否需要确认候选
        if context.candidates:
            exact_matches = [c for c in context.candidates if c.get("exact_match")]
            
            if exact_matches:
                # 有精确匹配，直接确认
                context.selected_candidate = exact_matches[0]["name"]
                context.collected_entities["materialName"] = context.selected_candidate
                
                # 检查是否还有其他缺失信息
                missing = context.get_missing_fields()
                if missing:
                    context.state = AgentState.COLLECTING_ENTITIES
                    return self._build_response_for_missing(context, missing)
                else:
                    context.state = AgentState.READY_TO_QUERY
                    return self._build_ready_response(context)
            else:
                # 需要用户确认候选
                context.state = AgentState.CONFIRMING_CANDIDATE
                return self._build_candidate_selection_response(context)
        else:
            # 无候选，继续收集信息
            context.state = AgentState.COLLECTING_ENTITIES
            return self._build_agent_response(
                context,
                f"未找到与「{material_name}」相关的材料。请尝试其他关键词，如：钢筋、水泥、混凝土等。",
                suggested_actions=["钢筋", "水泥", "混凝土", "铝合金", "玻璃"]
            )
    
    async def _handle_candidate_confirmation(
        self, 
        context: AgentContext, 
        user_message: str
    ) -> Dict[str, Any]:
        """处理候选材料确认"""
        self.logger.info(f"[Agent] 候选确认处理: {user_message}")
        
        # 解析用户选择
        selected = self._parse_candidate_selection(user_message, context.candidates)
        
        if selected:
            context.selected_candidate = selected
            context.collected_entities["materialName"] = selected
            
            # 检查是否还有其他缺失信息
            missing = context.get_missing_fields()
            if missing:
                context.state = AgentState.COLLECTING_ENTITIES
                return self._build_response_for_missing(context, missing)
            else:
                context.state = AgentState.READY_TO_QUERY
                return self._build_ready_response(context)
        else:
            # 无法解析选择，重新提示
            return self._build_candidate_selection_response(context)
    
    async def _handle_entity_collection(
        self, 
        context: AgentContext, 
        user_message: str
    ) -> Dict[str, Any]:
        """处理实体信息补充"""
        self.logger.info(f"[Agent] 实体补充处理: {user_message}")
        
        # 从用户输入中提取新实体
        entity_result = await self.mcp_server.call_tool(
            "extract_entities",
            {"question": user_message}
        )
        
        if entity_result.status == ToolStatus.SUCCESS:
            context.update_entities(entity_result.data)
        
        # 检查是否仍缺失字段
        missing = context.get_missing_fields()
        
        if missing:
            return self._build_response_for_missing(context, missing)
        else:
            context.state = AgentState.READY_TO_QUERY
            return self._build_ready_response(context)
    
    async def _handle_ready_state(
        self, 
        context: AgentContext, 
        user_message: str
    ) -> Dict[str, Any]:
        """处理准备查询状态（用户确认或修改）"""
        self.logger.info(f"[Agent] 准备查询状态处理: {user_message}")
        
        # 判断用户意图
        intent = self._parse_intent(user_message)
        
        if intent == "confirm":
            # 用户确认，执行查询
            return await self._execute_price_query(context)
        elif intent == "modify":
            # 用户想修改条件
            context.state = AgentState.COLLECTING_ENTITIES
            return self._build_agent_response(
                context,
                "好的，请告诉我需要修改的信息。",
                suggested_actions=["修改省份", "修改城市", "修改规格"]
            )
        else:
            # 默认再次确认
            return self._build_ready_response(context)
    
    async def _handle_new_or_followup(
        self, 
        context: AgentContext, 
        user_message: str
    ) -> Dict[str, Any]:
        """
        处理新查询或跟进问题
        
        检测是否是：
        1. 新材料的查询（重置上下文）
        2. 当前材料的深入问题
        """
        self.logger.info(f"[Agent] 新查询/跟进处理: {user_message}")
        
        # 检测是否是新查询
        is_new_query = await self._detect_new_query(context, user_message)
        
        if is_new_query:
            # 重置上下文，作为新查询处理
            context.collected_entities = {}
            context.candidates = []
            context.selected_candidate = None
            context.query_result = None
            context.turn_count = 1
            context.state = AgentState.IDLE
            return await self._handle_first_turn(context, user_message)
        else:
            # 视为当前查询的跟进
            # 可能是追问更多详情
            return await self._handle_followup_question(context, user_message)
    
    async def _handle_default(
        self, 
        context: AgentContext, 
        user_message: str
    ) -> Dict[str, Any]:
        """默认处理"""
        return self._build_agent_response(
            context,
            "我不太理解您的意思。您可以：\n1. 查询新材料价格\n2. 补充查询条件\n3. 查看已查询的结果",
            suggested_actions=["查询新材料", "补充条件", "查看结果"]
        )
    
    async def _execute_price_query(self, context: AgentContext) -> Dict[str, Any]:
        """执行价格查询"""
        context.state = AgentState.QUERYING
        
        self.logger.info(f"[Agent] 执行价格查询: {context.collected_entities}")
        
        # 调用查询工具
        query_result = await self.mcp_server.call_tool(
            "query_price_data",
            {
                "channel": context.channel,
                "material_name": context.collected_entities.get("materialName"),
                "province": context.collected_entities.get("province"),
                "city": context.collected_entities.get("city"),
                "material_model_spec": context.collected_entities.get("materialModelSpec"),
                "brand": context.collected_entities.get("brand")
            }
        )
        
        if query_result.status != ToolStatus.SUCCESS:
            context.state = AgentState.COLLECTING_ENTITIES
            return self._build_agent_response(
                context,
                f"查询失败：{query_result.message}。请尝试调整查询条件。",
                suggested_actions=["更换材料名称", "更换省份", "放宽条件"]
            )
        
        # 保存查询结果
        context.query_result = query_result.data
        context.state = AgentState.COMPLETED
        
        # 如果数据量合适，进行深度分析
        total_count = query_result.data.get("total_count", 0)
        price_data = query_result.data.get("price_data", [])
        
        if total_count > 0 and price_data:
            analysis_result = await self.mcp_server.call_tool(
                "analyze_prices",
                {"price_data": price_data}
            )
            
            # 构建包含分析的响应
            return self._build_query_result_response(
                context, 
                query_result.data,
                analysis_result.data if analysis_result.status == ToolStatus.SUCCESS else None
            )
        
        return self._build_query_result_response(context, query_result.data)
    
    async def _handle_followup_question(
        self, 
        context: AgentContext, 
        user_message: str
    ) -> Dict[str, Any]:
        """处理跟进问题"""
        # 简单实现：根据关键词返回相关信息
        user_lower = user_message.lower()
        
        if not context.query_result:
            return self._build_agent_response(
                context,
                "请先完成价格查询。",
                suggested_actions=["开始查询"]
            )
        
        # 根据用户问题返回相应信息
        if any(kw in user_lower for kw in ["便宜", "低", "优惠"]):
            return self._build_agent_response(
                context,
                "根据分析，建议您关注价格区间下限的材料。具体价格数据已在之前的回复中提供。",
                suggested_actions=["查看最低价格", "查看价格分布"]
            )
        elif any(kw in user_lower for kw in ["贵", "高", "平均"]):
            return self._build_agent_response(
                context,
                "您可以查看平均价格和中位数价格作为参考。",
                suggested_actions=["查看平均价格", "查看价格趋势"]
            )
        else:
            return self._build_agent_response(
                context,
                "您还有其他关于价格的问题吗？或者要查询其他材料？",
                suggested_actions=["查询新材料", "对比价格", "查看详细数据"]
            )
    
    # ============================================================================
    # 辅助方法
    # ============================================================================
    
    def _get_or_create_context(self, session_id: Optional[str]) -> AgentContext:
        """获取或创建Agent上下文"""
        if session_id and session_id in self.contexts:
            return self.contexts[session_id]
        
        # 创建新上下文
        import uuid
        new_session_id = session_id or str(uuid.uuid4())[:12]
        context = AgentContext(session_id=new_session_id)
        self.contexts[new_session_id] = context
        return context
    
    def _parse_candidate_selection(
        self, 
        user_message: str, 
        candidates: List[Dict[str, Any]]
    ) -> Optional[str]:
        """解析用户选择的候选材料"""
        # 尝试匹配数字选择
        import re
        number_match = re.search(r'(?:第)?(\d+)(?:个|号)?', user_message)
        if number_match:
            idx = int(number_match.group(1)) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]["name"]
        
        # 尝试直接匹配材料名称
        for cand in candidates:
            if cand["name"] in user_message:
                return cand["name"]
        
        # 模糊匹配
        for cand in candidates:
            if cand["name"] in user_message or user_message in cand["name"]:
                return cand["name"]
        
        return None
    
    def _parse_intent(self, user_message: str) -> str:
        """解析用户意图"""
        confirm_keywords = ["确认", "是的", "没错", "对", "可以", "好", "行", "查吧", "开始", "ok", "yes"]
        modify_keywords = ["修改", "不对", "错了", "换", "改", "重新", "不是"]
        
        user_lower = user_message.lower()
        
        for kw in confirm_keywords:
            if kw in user_lower:
                return "confirm"
        
        for kw in modify_keywords:
            if kw in user_lower:
                return "modify"
        
        return "unknown"
    
    async def _detect_new_query(
        self, 
        context: AgentContext, 
        user_message: str
    ) -> bool:
        """检测是否是新查询"""
        # 提取实体，看材料名称是否变化
        entity_result = await self.mcp_server.call_tool(
            "extract_entities",
            {"question": user_message}
        )
        
        if entity_result.status == ToolStatus.SUCCESS:
            new_material = entity_result.data.get("materialName")
            old_material = context.collected_entities.get("materialName")
            
            if new_material and new_material != old_material:
                return True
        
        # 检测切换信号词
        switch_signals = ["另外", "还有", "再查", "换一个", "其他", "新的", "别的材料"]
        for signal in switch_signals:
            if signal in user_message:
                return True
        
        return False
    
    def _build_candidate_selection_response(self, context: AgentContext) -> Dict[str, Any]:
        """构建候选选择响应"""
        lines = [
            f"找到多个与「{context.collected_entities.get('materialName')}」相关的材料，请选择：",
            ""
        ]
        
        for i, cand in enumerate(context.candidates[:5], 1):
            lines.append(f"  {i}. {cand['name']}")
        
        lines.append("")
        lines.append("请回复序号（如：1）或材料名称")
        
        return self._build_agent_response(
            context,
            "\n".join(lines),
            suggested_actions=[str(i+1) for i in range(min(5, len(context.candidates)))]
        )
    
    def _build_response_for_missing(
        self, 
        context: AgentContext, 
        missing: List[str]
    ) -> Dict[str, Any]:
        """构建缺失字段提示响应"""
        field_prompts = {
            "materialName": "请告诉我您想查询什么材料的价格？（如：钢筋、水泥）",
            "province": "请问是哪个省份的材料？（如：广东省、江苏省）",
            "city": "具体是哪个城市？（如：深圳市、南京市）"
        }
        
        # 取第一个缺失字段的提示
        prompt = field_prompts.get(missing[0], f"请补充{missing[0]}")
        
        # 快捷选项
        quick_options = {
            "province": ["广东省", "江苏省", "浙江省", "北京市", "上海市"],
            "city": ["深圳市", "广州市", "南京市", "杭州市", "上海市"]
        }
        
        return self._build_agent_response(
            context,
            prompt,
            suggested_actions=quick_options.get(missing[0], [])
        )
    
    def _build_ready_response(self, context: AgentContext) -> Dict[str, Any]:
        """构建准备查询响应"""
        material = context.collected_entities.get("materialName", "")
        province = context.collected_entities.get("province", "")
        city = context.collected_entities.get("city", "")
        
        location = ""
        if city:
            location = f"{province}{city}"
        elif province:
            location = province
        
        lines = [
            f"查询条件已收集完毕！",
            f"",
            f"📋 查询信息：",
            f"  • 材料名称：{material}",
        ]
        
        if location:
            lines.append(f"  • 地区：{location}")
        
        spec = context.collected_entities.get("materialModelSpec")
        if spec:
            lines.append(f"  • 规格型号：{spec}")
        
        lines.append("")
        lines.append("是否确认查询？")
        
        return self._build_agent_response(
            context,
            "\n".join(lines),
            suggested_actions=["✅ 确认查询", "🔄 修改条件"]
        )
    
    def _build_query_result_response(
        self, 
        context: AgentContext, 
        query_result: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """构建查询结果响应"""
        material = context.collected_entities.get("materialName", "")
        total_count = query_result.get("total_count", 0)
        
        lines = [
            f"✅ 「{material}」价格查询完成！",
            f"",
            f"📊 共查询到 {total_count} 条价格数据",
        ]
        
        if analysis:
            results = analysis.get("results", {})
            most_common_unit = analysis.get("most_common_unit", "")
            
            if most_common_unit and most_common_unit in results:
                unit_analysis = results[most_common_unit]
                recommend = unit_analysis.get("recommend_kmeans", {})
                
                lines.append("")
                lines.append(f"💰 推荐价格（单位：{most_common_unit}）：")
                
                if recommend.get("mode") == "two-tier":
                    prices = recommend.get("prices", [])
                    lines.append(f"  • 低价位：{prices[0]:.2f} 元")
                    lines.append(f"  • 高价位：{prices[1]:.2f} 元")
                else:
                    prices = recommend.get("prices", [])
                    if prices:
                        lines.append(f"  • {prices[0]:.2f} 元")
                
                price_range = unit_analysis.get("price_range", [0, 0])
                lines.append(f"\n📈 价格区间：{price_range[0]:.2f} - {price_range[1]:.2f} 元")
                lines.append(f"📊 平均价格：{unit_analysis.get('mean_price', 0):.2f} 元")
        
        lines.append("")
        lines.append("您还想了解什么？")
        
        return self._build_agent_response(
            context,
            "\n".join(lines),
            data={
                "query_result": query_result,
                "analysis": analysis
            },
            suggested_actions=["查询新材料", "查看详细数据", "对比价格"]
        )
    
    def _build_agent_response(
        self,
        context: AgentContext,
        message: str,
        data: Dict[str, Any] = None,
        suggested_actions: List[str] = None
    ) -> Dict[str, Any]:
        """构建Agent响应"""
        return {
            "success": True,
            "session_id": context.session_id,
            "state": context.state.value,
            "turn_count": context.turn_count,
            "message": message,
            "data": {
                **(data or {}),
                "entities": context.collected_entities,
                "channel": context.channel,
                "candidates": context.candidates,
                "selected_candidate": context.selected_candidate
            },
            "suggested_actions": suggested_actions or []
        }
