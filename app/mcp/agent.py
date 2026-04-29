"""
Agent 实现 - 基于 Function Call 的 ReAct Agent
处理材料价格查询的完整对话流程
"""

import json
import asyncio
import time
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

    MAX_TOOL_CONTENT_LEN: int = 4000

    def add_tool_result(self, tool_name: str, tool_result: Dict[str, Any]):
        """添加工具执行结果到上下文（带摘要截断）"""
        data = tool_result.get("data")
        content = self._summarize_tool_data(tool_name, data)
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_result.get("tool_call_id", ""),
            "name": tool_name,
            "content": content
        })

    def _summarize_tool_data(self, tool_name: str, data: Any) -> str:
        """对过大的工具返回数据进行摘要，防止撑爆上下文"""
        import json

        if not isinstance(data, dict):
            raw = json.dumps(data, ensure_ascii=False)
            if len(raw) > self.MAX_TOOL_CONTENT_LEN:
                return raw[:self.MAX_TOOL_CONTENT_LEN] + "...[内容已截断]"
            return raw

        if tool_name == "query_price_data" and "price_data" in data:
            summarized = {
                "success": data.get("success"),
                "intent": data.get("intent"),
                "total_count": data.get("total_count"),
                "most_common_unit": data.get("most_common_unit"),
                "answer": data.get("answer"),
            }
            price_data = data.get("price_data", [])
            summarized["price_data_summary"] = f"共 {len(price_data)} 条记录，已省略明细"
            if isinstance(price_data, list) and len(price_data) > 0:
                summarized["price_data_samples"] = price_data[:3]
            if data.get("report"):
                summarized["report"] = data.get("report")
            if data.get("summary_metrics"):
                summarized["summary_metrics"] = data.get("summary_metrics")
            return json.dumps(summarized, ensure_ascii=False)

        if tool_name == "analyze_prices":
            raw = json.dumps(data, ensure_ascii=False)
            if len(raw) > self.MAX_TOOL_CONTENT_LEN:
                simplified = {
                    "most_common_unit": data.get("most_common_unit"),
                    "most_common_records_count": len(data.get("most_common_records", [])),
                    "note": "详细分析结果较长，已摘要"
                }
                return json.dumps(simplified, ensure_ascii=False)
            return raw

        if tool_name == "quick_search_materials":
            simplified = {
                "keyword": data.get("keyword"),
                "candidates": data.get("candidates", [])[:5],
                "common_provinces": data.get("common_provinces", [])[:5],
                "total_candidates": len(data.get("candidates", []))
            }
            return json.dumps(simplified, ensure_ascii=False)

        raw = json.dumps(data, ensure_ascii=False)
        if len(raw) > self.MAX_TOOL_CONTENT_LEN:
            return raw[:self.MAX_TOOL_CONTENT_LEN] + "...[内容已截断]"
        return raw

    def get_llm_messages(self, max_non_system: int = 8) -> List[Dict[str, Any]]:
        """
        获取给 LLM 的截断后 messages。
        保留 system prompt + 最近 N 条非 system 消息。
        """
        if not self.messages:
            return self.messages

        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        other_msgs = [m for m in self.messages if m.get("role") != "system"]

        if len(other_msgs) <= max_non_system:
            return self.messages

        recent = other_msgs[-max_non_system:]

        # 安全检查：如果截断后第一条是 tool，必须补前面的 assistant(tool_calls)
        if recent and recent[0].get("role") == "tool":
            start_idx = len(other_msgs) - max_non_system - 1
            if start_idx >= 0:
                recent = other_msgs[start_idx:]

        return system_msgs + recent

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

    def _make_trace_id(self, context: AgentContext) -> str:
        return f"{context.session_id}-t{context.turn_count}"

    def _compact_query_result(self, query_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """压缩 query_result，用于持久化恢复。"""
        if not isinstance(query_result, dict):
            return None

        data = query_result.get("data", {})
        compact_data = None
        if isinstance(data, dict):
            compact_data = {
                key: data.get(key)
                for key in (
                    "success",
                    "intent",
                    "total_count",
                    "most_common_unit",
                    "parsed_entities",
                    "detail_answer",
                    "report",
                    "summary_metrics",
                    "chart_data",
                    "table_data",
                    "price_data",
                )
                if key in data
            }

        return {
            "status": query_result.get("status"),
            "message": query_result.get("message"),
            "metadata": query_result.get("metadata"),
            "suggested_next_steps": query_result.get("suggested_next_steps"),
            "data": compact_data,
        }

    def export_context_snapshot(self, session_id: str) -> Dict[str, Any]:
        """导出可持久化的 AgentContext 快照。"""
        context = self.contexts.get(session_id)
        if not context:
            return {}

        return {
            "session_id": context.session_id,
            "turn_count": context.turn_count,
            "channel": context.channel,
            "entities": context.entities,
            "query_result": self._compact_query_result(context.query_result),
        }

    def restore_context_from_conversation(self, conversation_session) -> Optional[AgentContext]:
        """
        从统一会话记录恢复 AgentContext。
        仅在当前进程内没有对应 context 时执行恢复。
        """
        if not conversation_session:
            return None

        session_id = conversation_session.session_id
        if session_id in self.contexts:
            return self.contexts[session_id]

        context = AgentContext(session_id=session_id)
        context.add_message("system", SYSTEM_PROMPT)

        for msg in conversation_session.messages[-12:]:
            if msg.role in {"user", "assistant"}:
                context.add_message(msg.role, msg.content)

        snapshot = {}
        for msg in reversed(conversation_session.messages):
            metadata = getattr(msg, "metadata", None) or {}
            if metadata.get("agent_context"):
                snapshot = metadata["agent_context"]
                break

        if snapshot:
            context.turn_count = int(snapshot.get("turn_count") or 0)
            context.channel = snapshot.get("channel") or context.channel
            context.entities = dict(snapshot.get("entities") or {})
            compact_query_result = snapshot.get("query_result")
            if isinstance(compact_query_result, dict):
                context.query_result = compact_query_result
        else:
            context.turn_count = sum(1 for msg in conversation_session.messages if msg.role == "user")

        self.contexts[session_id] = context
        self.logger.info(
            f"[Agent Restore] 已恢复会话: session={session_id}, "
            f"turn_count={context.turn_count}, entities={self._summarize_entities(context.entities)}"
        )
        return context

    def _safe_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    def _shorten_text(self, text: Any, limit: int = 160) -> str:
        if text is None:
            return ""
        normalized = str(text).replace("\n", "\\n")
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit] + "..."

    def _summarize_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "materialName",
            "province",
            "city",
            "materialModelSpec",
            "brand",
            "channel",
            "channel_specified",
        ]
        return {
            key: value for key, value in entities.items()
            if key in keys and value not in (None, "", [], {})
        }

    def _summarize_tool_result(self, tool_name: str, result: Any) -> Dict[str, Any]:
        data = getattr(result, "data", None)
        summary = {
            "status": getattr(getattr(result, "status", None), "value", None),
            "message": self._shorten_text(getattr(result, "message", ""), 120),
        }
        if isinstance(data, dict):
            for key in ("success", "intent", "total_count", "most_common_unit", "error_message"):
                if key in data:
                    summary[key] = data.get(key)
            if "price_data" in data and isinstance(data["price_data"], list):
                summary["price_data_count"] = len(data["price_data"])
            if "candidates" in data and isinstance(data["candidates"], list):
                summary["candidate_count"] = len(data["candidates"])
        elif isinstance(data, list):
            summary["data_count"] = len(data)
        elif data is not None:
            summary["data_preview"] = self._shorten_text(data, 120)
        if getattr(result, "metadata", None):
            summary["metadata"] = result.metadata
        return summary

    def _log_trace(self, trace_id: str, stage: str, **kwargs):
        detail = ", ".join(
            f"{key}={self._safe_json(value)}"
            for key, value in kwargs.items()
            if value is not None
        )
        message = f"[Agent Trace][{trace_id}] {stage}"
        if detail:
            message = f"{message} | {detail}"
        self.logger.info(message)

    def _tool_label(self, tool_name: str) -> str:
        labels = {
            "extract_entities": "提取查询实体",
            "quick_search_materials": "搜索候选材料",
            "get_material_candidates": "获取候选材料",
            "identify_channel": "识别价格渠道",
            "query_price_data": "查询价格数据",
            "analyze_prices": "分析价格结果",
        }
        return labels.get(tool_name, tool_name)

    def _make_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        display_text: str,
        next_type: Optional[str] = None,
        display_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = {
            "type": event_type,
            "data": data,
            "display": {
                "title": display_title or event_type,
                "text": display_text,
            },
        }
        if next_type is not None:
            event["next_type"] = next_type
        return event

    def _make_announce_event(
        self,
        next_type: str,
        message: str,
        step: Optional[int] = None,
        tool_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        data = {
            "next_type": next_type,
            "message": message,
        }
        if step is not None:
            data["step"] = step
        if tool_name:
            data["tool"] = tool_name
            data["tool_label"] = self._tool_label(tool_name)
        return self._make_event(
            "announce",
            data,
            display_text=message,
            next_type=next_type,
            display_title="即将开始",
        )

    def _get_or_create_context(self, session_id: Optional[str]) -> AgentContext:
        if session_id and session_id in self.contexts:
            return self.contexts[session_id]
        import uuid
        new_id = session_id or str(uuid.uuid4())[:12]
        context = AgentContext(session_id=new_id)
        context.add_message("system", SYSTEM_PROMPT)
        self.contexts[new_id] = context
        return context

    def _get_tools_for_llm(self, context: AgentContext) -> List[Dict[str, Any]]:
        """根据上下文动态获取 OpenAI 格式的工具定义"""
        raw_tools = {t["name"]: t for t in self.mcp_server.get_available_tools()}
        tools = []

        # 始终保留实体提取
        if "extract_entities" in raw_tools:
            tools.append(self._format_tool(raw_tools["extract_entities"]))

        # 没有材料名时：搜索类工具优先
        if not context.entities.get("materialName"):
            for name in ["quick_search_materials", "get_material_candidates"]:
                if name in raw_tools:
                    tools.append(self._format_tool(raw_tools[name]))
            return tools

        # 有材料名但渠道未明确确认
        if not context.entities.get("channel_specified"):
            if "identify_channel" in raw_tools:
                tools.append(self._format_tool(raw_tools["identify_channel"]))

        # 核心查询工具
        if "query_price_data" in raw_tools:
            tools.append(self._format_tool(raw_tools["query_price_data"]))

        # 只有已有查询结果时，才给分析工具
        if context.query_result and context.query_result.get("success"):
            if "analyze_prices" in raw_tools:
                tools.append(self._format_tool(raw_tools["analyze_prices"]))

        return tools

    def _format_tool(self, raw_tool: Dict) -> Dict:
        """格式化为 OpenAI Function Calling 标准格式"""
        return {
            "type": "function",
            "function": {
                "name": raw_tool["name"],
                "description": raw_tool["description"],
                "parameters": raw_tool["parameters"]
            }
        }

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
        trace_id = self._make_trace_id(context)

        self._log_trace(
            trace_id,
            "request.start",
            session_id=context.session_id,
            turn=context.turn_count,
            user_message=self._shorten_text(user_message, 200),
            entities=self._summarize_entities(context.entities),
        )

        yield self._make_event(
            "start",
            {
                "session_id": context.session_id,
                "turn_count": context.turn_count,
                "user_message": user_message
            },
            display_text=f"已接收问题，开始处理第 {context.turn_count} 轮对话",
            next_type="announce",
            display_title="开始处理",
        )

        # ReAct 循环
        for step in range(1, self.max_steps + 1):
            step_started_at = time.time()
            tools_for_llm = self._get_tools_for_llm(context)
            self._log_trace(
                trace_id,
                "react.step.start",
                step=step,
                available_tools=[tool["function"]["name"] for tool in tools_for_llm],
                entities=self._summarize_entities(context.entities),
            )
            yield self._make_announce_event(
                "thought",
                f"正在分析问题并规划第 {step} 步",
                step=step,
            )

            # 调用 LLM（使用截断后的 messages 和动态工具）
            messages_for_llm = context.get_llm_messages()
            llm_started_at = time.time()
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages_for_llm,
                tools=tools_for_llm,
                tool_choice="auto",
                temperature=0.2,
            )
            llm_duration_ms = int((time.time() - llm_started_at) * 1000)

            message = response.choices[0].message
            self._log_trace(
                trace_id,
                "react.step.llm_response",
                step=step,
                duration_ms=llm_duration_ms,
                has_tool_calls=bool(message.tool_calls),
                tool_call_count=len(message.tool_calls or []),
                content_preview=self._shorten_text(message.content or "", 200),
            )

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

                first_tool_name = message.tool_calls[0].function.name if message.tool_calls else None
                thought_text = (
                    f"第 {step} 步已确定下一步动作，准备调用 {len(message.tool_calls)} 个工具"
                    if message.tool_calls else
                    f"第 {step} 步完成思考"
                )
                yield self._make_event(
                    "thought",
                    thought,
                    display_text=thought_text,
                    next_type="announce" if first_tool_name else "final",
                    display_title="思考结果",
                )

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

                    tool_started_at = time.time()
                    self._log_trace(
                        trace_id,
                        "tool.call.start",
                        step=step,
                        tool=tool_name,
                        params=params,
                    )

                    yield self._make_announce_event(
                        "tool_call",
                        f"正在调用工具：{self._tool_label(tool_name)}",
                        step=step,
                        tool_name=tool_name,
                    )
                    yield self._make_event(
                        "tool_call",
                        {
                            "name": tool_name,
                            "params": params
                        },
                        display_text=f"开始执行工具：{self._tool_label(tool_name)}",
                        next_type="observation",
                        display_title="调用工具",
                    )

                    # 参数兼容：LLM 有时会把 material_name 当作 quick_search_materials 的参数
                    if tool_name == "quick_search_materials":
                        if "material_name" in params and "keyword" not in params:
                            params["keyword"] = params.pop("material_name")
                        if "channel" not in params and context.channel:
                            params["channel"] = context.channel
                        allowed_params = {"keyword", "limit", "channel"}
                        params = {
                            key: value for key, value in params.items()
                            if key in allowed_params
                        }

                    # 特殊处理：合并上下文实体到 query_price_data
                    if tool_name == "query_price_data":
                        params = self._merge_entities_for_query(context, params)
                        params["generate_report"] = False
                        params["user_question"] = user_message
                        params["trace_id"] = trace_id

                    result = await self.mcp_server.call_tool(tool_name, params)
                    tool_duration_ms = int((time.time() - tool_started_at) * 1000)

                    # 更新上下文实体
                    if tool_name == "extract_entities" and result.data:
                        context.update_entities(result.data)
                    if tool_name == "identify_channel" and result.data:
                        context.channel = result.data.get("channel", context.channel)
                        # 如果渠道是明确指定的（confidence=1.0），标记为已确认
                        if result.data.get("confidence") == 1.0:
                            context.entities["channel_specified"] = True
                    if tool_name == "quick_search_materials" and result.data:
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
                    self._log_trace(
                        trace_id,
                        "tool.call.end",
                        step=step,
                        tool=tool_name,
                        duration_ms=tool_duration_ms,
                        result=self._summarize_tool_result(tool_name, result),
                        entities_after=self._summarize_entities(context.entities),
                    )

                    next_event_type = "report_start" if tool_name == "query_price_data" and result.data else "announce"
                    yield self._make_event(
                        "observation",
                        {
                            "tool": tool_name,
                            "result": obs_data
                        },
                        display_text=f"{self._tool_label(tool_name)}已完成：{self._shorten_text(result.message or '执行完成', 120)}",
                        next_type=next_event_type,
                        display_title="工具结果",
                    )

                    # 查询成功后，在流式接口中直接流式生成报告并结束，避免再次把大结果送回 LLM 导致上下文超限
                    if (
                        tool_name == "query_price_data"
                        and result.status in [ToolStatus.SUCCESS, ToolStatus.PARTIAL]
                        and result.data
                    ):
                        streamed_report = None
                        async for report_event in self._stream_price_report(context, user_message, trace_id):
                            if report_event["type"] == "report_chunk":
                                streamed_report = (streamed_report or "") + report_event["data"]["content"]
                            yield report_event

                        final_text = (
                            context.query_result.get("data", {}).get("report")
                            if context.query_result and context.query_result.get("data")
                            else None
                        ) or streamed_report or result.data.get("report") or result.message
                        context.add_message("assistant", final_text)
                        self._log_trace(
                            trace_id,
                            "request.final",
                            step=step,
                            source="query_price_data",
                            final_text_preview=self._shorten_text(final_text, 240),
                            total_step_duration_ms=int((time.time() - step_started_at) * 1000),
                        )
                        response_data = self._build_response(context, final_text)
                        yield self._make_event(
                            "final",
                            response_data,
                            display_text="处理完成，已生成最终回答",
                            display_title="最终结果",
                        )
                        return

                # 继续下一轮
                continue

            else:
                # 没有工具调用，直接回复用户
                context.thoughts.append(thought)
                final_text = (
                    context.query_result.get("data", {}).get("report")
                    if context.query_result and context.query_result.get("data")
                    else None
                ) or (message.content or "")
                context.add_message("assistant", final_text)
                self._log_trace(
                    trace_id,
                    "request.final",
                    step=step,
                    source="llm_direct_reply",
                    final_text_preview=self._shorten_text(final_text, 240),
                    total_step_duration_ms=int((time.time() - step_started_at) * 1000),
                )

                response_data = self._build_response(context, final_text)
                yield self._make_event(
                    "final",
                    response_data,
                    display_text="处理完成，已直接生成回复",
                    display_title="最终结果",
                )
                return

        # 超过最大步数
        fallback = "抱歉，这个问题我需要多想一想，您可以尝试说得更具体一些，比如告诉我材料名称和省份。"
        context.add_message("assistant", fallback)
        self._log_trace(
            trace_id,
            "request.max_steps_reached",
            max_steps=self.max_steps,
            entities=self._summarize_entities(context.entities),
        )
        response_data = self._build_response(context, fallback)
        yield self._make_event(
            "final",
            response_data,
            display_text="已达到最大执行步数，返回兜底回复",
            display_title="最终结果",
        )

    def _merge_entities_for_query(self, context: AgentContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """合并上下文实体到 query_price_data 参数"""
        entities = context.entities.copy()
        # 如果 params 里有 entities 字段，优先用里面的
        if "entities" in params:
            entities.update(params["entities"])
        for key, value in params.items():
            if key != "entities" and value not in (None, ""):
                entities[key] = value

        # 将 camelCase 转为下划线格式（如果存在）
        name_map = {
            "materialName": "material_name",
            "materialModelSpec": "material_model_spec",
            "startReleaseDate": "start_release_date",
            "endReleaseDate": "end_release_date",
            "startReleaseTime": "start_release_date",
            "endReleaseTime": "end_release_date",
            "releaseDate": "start_release_date",
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
            "start_release_date": entities.get("start_release_date", ""),
            "end_release_date": entities.get("end_release_date", ""),
        }

    async def _stream_price_report(
        self,
        context: AgentContext,
        user_message: str,
        trace_id: Optional[str] = None
    ):
        """流式生成价格报告，并将 chunk 持续返回给前端。"""
        if not context.query_result or not context.query_result.get("data"):
            return

        query_data = context.query_result["data"]
        parsed_entities = query_data.get("parsed_entities", "{}")
        detail_answer = query_data.get("detail_answer", "{}")
        full_report = ""

        yield self._make_announce_event(
            "report_start",
            "正在生成价格报告",
        )
        yield self._make_event(
            "report_start",
            {
                "message": "开始生成价格报告"
            },
            display_text="开始生成价格报告",
            next_type="report_chunk",
            display_title="生成报告",
        )

        report_started_at = time.time()
        chunk_count = 0
        if trace_id:
            self._log_trace(
                trace_id,
                "report.start",
                parsed_entities=self._summarize_entities(context.entities),
            )

        try:
            async for chunk in self.llm.stream_price_predict(
                parsed_entities=parsed_entities,
                detail_answer=detail_answer,
                query=user_message,
                messages=[]
            ):
                if not chunk:
                    continue
                if chunk.startswith("\n[ERROR]"):
                    raise RuntimeError(chunk.strip())
                full_report += chunk
                chunk_count += 1
                yield self._make_event(
                    "report_chunk",
                    {
                        "content": chunk
                    },
                    display_text=self._shorten_text(chunk, 160),
                    next_type="report_chunk",
                    display_title="报告片段",
                )
        except Exception as e:
            self.logger.error(f"[Agent Trace][{trace_id}] report.error | error={self._safe_json(str(e))}")
            fallback = self.mcp_server.rag_service._build_price_report_fallback(
                parsed_entities=context.entities,
                channel_result=context.channel,
                summary_metrics=query_data.get("summary_metrics", {}),
                total_count=query_data.get("total_count", 0)
            )
            full_report = fallback
            yield self._make_event(
                "report_chunk",
                {
                    "content": fallback
                },
                display_text="报告生成失败，已切换为兜底报告",
                next_type="final",
                display_title="报告片段",
            )
        finally:
            if trace_id:
                self._log_trace(
                    trace_id,
                    "report.end",
                    duration_ms=int((time.time() - report_started_at) * 1000),
                    chunk_count=chunk_count,
                    used_fallback=not bool(full_report and chunk_count > 0),
                    report_preview=self._shorten_text(full_report, 200),
                )

        if full_report and context.query_result.get("data") is not None:
            context.query_result["data"]["report"] = full_report

    def _build_response(
        self,
        context: AgentContext,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        query_data = context.query_result.get("data", {}) if context.query_result else {}
        final_message = query_data.get("report") or message
        return {
            "success": True,
            "session_id": context.session_id,
            "state": "completed",
            "turn_count": context.turn_count,
            "message": final_message,
            "data": {
                **(data or {}),
                "entities": context.entities,
                "channel": context.channel,
                "query_result": context.query_result,
                "report": query_data.get("report"),
                "summary_metrics": query_data.get("summary_metrics"),
                "chart_data": query_data.get("chart_data"),
                "table_data": query_data.get("table_data"),
                "price_data": query_data.get("price_data"),
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
