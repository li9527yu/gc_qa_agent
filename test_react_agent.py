"""
测试 ReAct Agent 的核心逻辑（使用本地真实的 Qwen3-32B）
"""
import asyncio
import json
import sys
sys.path.insert(0, "/home/tpc/suda/rag/rag_agent")

from app.llm_deepseek import LLMPredictor
from app.mcp.server import MCPServer
from app.mcp.agent import MaterialPriceAgent
from app.mcp.tools import ToolResult, ToolStatus


class MockRAGService:
    """最小化的 RAGService Mock，只实现 Agent 需要的方法"""
    
    def __init__(self):
        self.llm = LLMPredictor()
        self.logger = self.llm.logger
    
    async def extract_entities(self, question: str):
        # 简单规则提取
        entities = {}
        if "钢筋" in question:
            entities["materialName"] = "钢筋"
        if "水泥" in question:
            entities["materialName"] = "水泥"
        if "广东省" in question or "广东" in question:
            entities["province"] = "广东省"
        if "深圳市" in question or "深圳" in question:
            entities["city"] = "深圳市"
        if "江苏省" in question or "江苏" in question:
            entities["province"] = "江苏省"
        if "南京市" in question or "南京" in question:
            entities["city"] = "南京市"
        return entities
    
    async def identify_channel(self, question, parsed_entities=None):
        from app.services.rag_service import ChannelType
        from app.utils.channel_inferencer import ChannelInferenceResult, ChannelType as InferencerChannelType
        return ChannelType.INFORMATION_PRICE, ChannelInferenceResult(
            channel=InferencerChannelType.INFORMATION_PRICE,
            confidence=0.8,
            reason="默认信息价",
            matched_keywords=[]
        )
    
    async def process_price_recommendation(self, channel, entities):
        material = entities.get("materialName", "")
        province = entities.get("province", "")
        city = entities.get("city", "")
        
        # 模拟查询结果
        if material == "钢筋" and province == "广东省" and city == "深圳市":
            return {
                "success": True,
                "total_count": 156,
                "price_data": [
                    {"materialName": "钢筋", "materialModelSpec": "HRB400", "price": 24.5, "unit": "kg", "province": "广东省", "city": "深圳市", "releaseDate": "2024-01"},
                ],
                "answer": "",
                "detail_answer": json.dumps({"kg": {"total_count": 156, "mean_price": 24.5}}),
                "metadata": {"channel": channel, "entities": entities}
            }
        elif material == "钢筋":
            # 条件不全但直接查，返回大量数据
            return {
                "success": True,
                "total_count": 3000,
                "price_data": [],
                "answer": "",
                "detail_answer": "",
                "metadata": {"channel": channel, "entities": entities}
            }
        else:
            return {
                "success": False,
                "total_count": 0,
                "error_message": "未查询到数据",
                "metadata": {}
            }
    
    def _get_infor_material(self, data: dict):
        # 返回候选和样本统计用的数据
        material_name = data.get("materialName", "")
        if "钢筋" in material_name:
            samples = [
                {"materialName": "钢筋", "materialModelSpec": "HRB400", "province": "广东省", "city": "深圳市", "price": 24.5, "unit": "kg"},
                {"materialName": "钢筋", "materialModelSpec": "HRB500", "province": "广东省", "city": "广州市", "price": 26.0, "unit": "kg"},
                {"materialName": "钢筋", "materialModelSpec": "HRB400", "province": "江苏省", "city": "南京市", "price": 23.8, "unit": "kg"},
            ]
            return {
                "code": 200,
                "data": {"list": samples, "totalCount": len(samples)}
            }
        return {"code": 200, "data": {"list": [], "totalCount": 0}}


async def test_agent():
    print("=" * 60)
    print("初始化 Agent...")
    
    rag_service = MockRAGService()
    mcp_server = MCPServer(rag_service)
    agent = MaterialPriceAgent(mcp_server, rag_service.llm)
    
    print("Agent 初始化完成")
    print("=" * 60)
    
    # 测试 1：首次查询，条件不全
    print("\n【测试1】用户：查一下钢筋的价格")
    result1 = await agent.process_message("查一下钢筋的价格")
    print(f"Agent 回复：{result1['message']}")
    print(f"Thoughts 步数：{len(result1['data']['thoughts'])}")
    print(f"收集到的实体：{result1['data']['entities']}")
    print(f"建议操作：{result1['suggested_actions']}")
    
    session_id = result1["session_id"]
    
    # 测试 2：补充条件
    print("\n【测试2】用户：广东省深圳市的")
    result2 = await agent.process_message("广东省深圳市的", session_id=session_id)
    print(f"Agent 回复：{result2['message']}")
    print(f"Thoughts 步数：{len(result2['data']['thoughts'])}")
    print(f"收集到的实体：{result2['data']['entities']}")
    print(f"查询结果：{result2['data'].get('query_result')}")
    
    # 测试 3：用户不听建议，直接查
    print("\n【测试3】新建会话，用户：直接查钢筋价格")
    result3 = await agent.process_message("直接查钢筋价格")
    print(f"Agent 回复：{result3['message']}")
    print(f"Thoughts 步数：{len(result3['data']['thoughts'])}")
    print(f"查询结果：{result3['data'].get('query_result')}")


if __name__ == "__main__":
    asyncio.run(test_agent())
