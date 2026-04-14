"""
Agent 实现 - 基于 Function Call 的 ReAct Agent
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


SYSTEM_PROMPT = """\
你是「材价通」——一个专业的材料价格查询智能助手。你的目标是通过调用工具，帮助用户查询建筑材料的价格信息。

## 可用工具说明

1. **extract_entities**
   - 作用：从用户问题中提取查询参数（材料名、省份、城市、规格、品牌、时间等）
   - 使用时机：每次收到用户输入时，优先调用以更新实体信息

2. **quick_search_materials**
   - 作用：根据材料关键词模糊搜索相关材料名称、常见省份、常见规格
   - 使用时机：用户条件不全（缺少省份/城市/规格）或材料名可能不精确时，**必须优先调用**

3. **query_price_data**
   - 作用：根据已收集的实体执行实际价格查询
   - 使用时机：条件较全时调用。至少要有 materialName

4. **analyze_prices**
   - 作用：对查询到的价格数据进行统计分析（K-means聚类、价格区间等）
   - 使用时机：query_price_data 成功返回数据后，可以调用以生成更深入的分析

## 决策规则（必须遵守）

1. **条件不全时**：
   - 如果用户只给了材料名，缺少省份、城市、规格等，**必须先调用 quick_search_materials**
   - 基于 quick_search_materials 返回的候选材料和常见地区/规格，向用户说明缺少的条件，并提供具体候选建议
   - **渠道确认**：如果用户没有明确指定查询渠道（"信息价"或"厂商报价"），必须主动询问用户确认。说明两个渠道的区别：
     - 信息价：政府发布的指导价，适合一般工程造价参考
     - 厂商报价：品牌厂商的实际报价，适合有品牌要求的场景
   - 示例回复："找到相关材料：钢筋HRB400、钢筋HRB500。当前查询渠道：信息价（推断）。如果您想查询厂商报价，请告诉我。另外还缺少省份信息，请问是哪个省份？（如：广东省、江苏省）"

2. **条件齐全时**：
   - 直接调用 query_price_data 查询价格
   - 查询成功后，可以调用 analyze_prices 进行分析
   - 用自然语言向用户说明查询结果

3. **用户不听建议、坚持直接查询时**：
   - 如果用户说"直接查"、"不用了"、"就这样"、"查吧"等，**尊重用户意愿**
   - 直接用当前已收集的条件调用 query_price_data，不再追问

4. **查询结果为空时**：
   - 向用户解释可能的原因：该地区无此材料数据、规格过于具体、时间范围无数据、材料名称不匹配等
   - 给出具体建议："未查询到广东省深圳市的钢筋价格，可能原因：1) 该地区暂无数据，可尝试放宽到广东省；2) 材料名称可能不准确，可尝试钢筋HRB400"

5. **多轮对话**：
   - 始终维护已收集的实体，新信息覆盖旧信息
   - 不要重复询问已经知道的条件

## 输出要求

- 当你需要调用工具时，请通过 function call 输出
- 当你可以直接回复用户时，用自然语言、简洁、友好地回复
- 所有价格数据必须准确传达，不要编造
"""


@dataclass
class AgentContext:
    """Agent上下文"""
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    channel: str = "information_price"
    query_result: Optional[Dict[str, Any]] = None
    turn_count: int = 0
    thoughts: List[Dict[str, Any]] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def add_tool_result(self, tool_name: str, tool_result: Dict[str, Any]):
        """添加工具执行结果到上下文"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_result.get("tool_call_id", ""),
            "name": tool_name,
            "content": json.dumps(tool_result.get("data"), ensure_ascii=False)
        })

    def update_entities(self, new_entities: Dict[str, Any]):
        for key, value in new_entities.items():
            if value is not None and value != "":
                self.entities[key] = value

    def is_ready_to_query(self) -> bool:
        return bool(self.entities.get("materialName"))

    def get_missing_fields(self) -> List[str]:
        required = ["materialName", "province", "city"]
        missing = [f for f in required if not self.entities.get(f)]
        # 如果渠道不明确（非用户主动指定），也视为需要补充
        if not self.entities.get("channel_specified"):
            missing.append("channel")
        return missing


class MaterialPriceAgent:
    """
    材料价格查询 Agent - 基于 Function Call 的 ReAct
    """

    def __init__(self, mcp_server: MCPServer, llm_client):
        self.mcp_server = mcp_server
        self.llm = llm_client
        self.logger = logging.getLogger("easy_rag_api")
        self.contexts: Dict[str, AgentContext] = {}
        self.max_steps = 5

    def _get_or_create_context(self, session_id: Optional[str]) -> AgentContext:
        if session_id and session_id in self.contexts:
            return self.contexts[session_id]
        import uuid
        new_id = session_id or str(uuid.uuid4())[:12]
        context = AgentContext(session_id=new_id)
        context.add_message("system", SYSTEM_PROMPT)
        self.contexts[new_id] = context
        return context

    def _get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """获取 OpenAI 格式的工具定义"""
        raw_tools = self.mcp_server.get_available_tools()
        tools = []
        for t in raw_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"]
                }
            })
        return tools

    async def process_message(
        self,
        user_message: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理用户消息（主入口）"""
        async for event in self.process_message_stream(user_message, session_id):
            if event["type"] == "final":
                return event["data"]
        # 理论上不会走到这里
        return {"success": False, "message": "处理异常"}

    async def process_message_stream(
        self,
        user_message: str,
        session_id: Optional[str] = None
    ):
        """
        流式处理用户消息，yield 每个步骤的事件
        
        事件类型：
        - start: 开始处理
        - thought: LLM 思考/回复（包含 tool_calls 意向）
        - tool_call: 开始调用工具
        - observation: 工具执行结果
        - final: 最终回复和完整数据
        """
        from typing import AsyncGenerator
        
        context = self._get_or_create_context(session_id)
        context.turn_count += 1
        context.add_message("user", user_message)

        self.logger.info(
            f"[Agent Stream] 开始处理: session={context.session_id}, "
            f"turn={context.turn_count}, entities={context.entities}"
        )

        yield {
            "type": "start",
            "data": {
                "session_id": context.session_id,
                "turn_count": context.turn_count,
                "user_message": user_message
            }
        }

        # ReAct 循环
        for step in range(1, self.max_steps + 1):
            self.logger.info(f"[Agent Stream] ReAct Step {step}")

            # 调用 LLM
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=context.messages,
                tools=self._get_tools_for_llm(),
                tool_choice="auto",
                temperature=0.2,
            )

            message = response.choices[0].message

            # 记录 thought
            thought = {
                "step": step,
                "content": message.content or "",
                "tool_calls": []
            }

            if message.tool_calls:
                for tc in message.tool_calls:
                    thought["tool_calls"].append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    })
                context.thoughts.append(thought)

                yield {
                    "type": "thought",
                    "data": thought
                }

                # 先把 assistant 的 tool_calls 消息加入上下文
                context.messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # 执行每个工具调用
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        params = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        params = {}

                    self.logger.info(f"[Agent Stream] 调用工具: {tool_name}, 参数: {params}")

                    yield {
                        "type": "tool_call",
                        "data": {
                            "name": tool_name,
                            "params": params
                        }
                    }

                    # 特殊处理：合并上下文实体到 query_price_data
                    if tool_name == "query_price_data":
                        params = self._merge_entities_for_query(context, params)

                    result = await self.mcp_server.call_tool(tool_name, params)

                    # 更新上下文实体
                    if tool_name == "extract_entities" and result.data:
                        context.update_entities(result.data)
                    if tool_name == "identify_channel" and result.data:
                        context.channel = result.data.get("channel", context.channel)
                        # 如果渠道是明确指定的（confidence=1.0），标记为已确认
                        if result.data.get("confidence") == 1.0:
                            context.entities["channel_specified"] = True
                    if tool_name == "quick_search_materials":
                        candidates = result.data.get("candidates", [])
                        exact = [c for c in candidates if c.get("exact_match")]
                        if exact and not context.entities.get("materialName"):
                            context.entities["materialName"] = exact[0]["name"]

                    if tool_name == "query_price_data":
                        context.query_result = result.to_dict()

                    obs_data = {
                        "status": result.status.value,
                        "message": result.message,
                        "data": result.data,
                    }

                    context.add_tool_result(tool_name, {
                        **obs_data,
                        "tool_call_id": tc.id
                    })

                    yield {
                        "type": "observation",
                        "data": {
                            "tool": tool_name,
                            "result": obs_data
                        }
                    }

                # 继续下一轮
                continue

            else:
                # 没有工具调用，直接回复用户
                context.thoughts.append(thought)
                context.add_message("assistant", message.content or "")
                self.logger.info(f"[Agent Stream] 最终回复: {message.content}")

                response_data = self._build_response(context, message.content or "")
                yield {
                    "type": "final",
                    "data": response_data
                }
                return

        # 超过最大步数
        fallback = "抱歉，这个问题我需要多想一想，您可以尝试说得更具体一些，比如告诉我材料名称和省份。"
        context.add_message("assistant", fallback)
        response_data = self._build_response(context, fallback)
        yield {
            "type": "final",
            "data": response_data
        }

    def _merge_entities_for_query(self, context: AgentContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """合并上下文实体到 query_price_data 参数"""
        entities = context.entities.copy()
        # 如果 params 里有 entities 字段，优先用里面的
        if "entities" in params:
            entities.update(params["entities"])

        # 将 camelCase 转为下划线格式（如果存在）
        name_map = {
            "materialName": "material_name",
            "materialModelSpec": "material_model_spec",
        }
        for old_key, new_key in name_map.items():
            if old_key in entities and new_key not in entities:
                entities[new_key] = entities.pop(old_key)

        return {
            "channel": params.get("channel", context.channel),
            "material_name": entities.get("material_name") or entities.get("materialName", ""),
            "province": entities.get("province", ""),
            "city": entities.get("city", ""),
            "material_model_spec": entities.get("material_model_spec") or entities.get("materialModelSpec", ""),
            "brand": entities.get("brand", ""),
        }

    def _build_response(
        self,
        context: AgentContext,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "session_id": context.session_id,
            "state": "completed",
            "turn_count": context.turn_count,
            "message": message,
            "data": {
                **(data or {}),
                "entities": context.entities,
                "channel": context.channel,
                "query_result": context.query_result,
                "thoughts": context.thoughts
            },
            "suggested_actions": self._generate_suggested_actions(context)
        }

    def _generate_suggested_actions(self, context: AgentContext) -> List[str]:
        """根据当前状态生成建议操作"""
        if not context.entities.get("materialName"):
            return ["钢筋", "水泥", "混凝土", "铝合金"]

        missing = context.get_missing_fields()
        if "channel" in missing:
            return ["信息价", "厂商报价"]
        if "province" in missing:
            return ["广东省", "江苏省", "浙江省", "北京市", "上海市"]
        if "city" in missing:
            return ["深圳市", "广州市", "南京市", "杭州市"]

        if context.query_result:
            return ["查询新材料", "查看详细数据", "对比价格"]

        return ["直接查询", "补充条件"]
