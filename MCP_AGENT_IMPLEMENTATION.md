# 材料价格查询 Agent 与 MCP 工具封装方案

> 文档版本: v1.0  
> 创建时间: 2026-04-09  
> 适用项目: easy-rag

---

## 一、方案概述

### 1.1 目标

将现有的材料价格查询系统升级为支持**多轮对话**、**工具自主调用**的智能Agent系统，并封装为标准化的**MCP（Model Context Protocol）工具**。

### 1.2 核心改进

| 能力 | 改进前 | 改进后 |
|-----|--------|--------|
| 首次查询 | 直接尝试查询，失败返回错误 | 智能补充建议 + 候选材料推荐 |
| 工具调用 | 硬编码调用链 | MCP标准接口，可动态组合 |
| 多轮对话 | 简单的实体收集 | Agent状态机，自主决策 |
| 扩展性 | 新增功能需修改多处 | 新增工具即可被Agent使用 |

---

## 二、架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        API 层 (FastAPI)                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ /api/v1/query/* │  │ /api/v1/agent/* │  │ /api/v1/mcp/* │ │
│  │ 原有接口        │  │ Agent对话       │  │ MCP工具       │ │
│  └─────────────────┘  └─────────────────┘  └───────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      Agent 层                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              MaterialPriceAgent                        │  │
│  │  - 状态管理（IDLE → COLLECTING → READY → COMPLETED）   │  │
│  │  - 多轮对话协调                                        │  │
│  │  - 工具调用决策                                        │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      MCP Server 层                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  MCPServer                             │  │
│  │  - 工具注册与管理                                       │  │
│  │  - 工具调用路由                                        │  │
│  │  - Agent流程协调（首次查询、后续查询）                   │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      MCP Tools 层                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ 数据查询工具  │ │ 数据分析工具  │ │ 辅助工具             │ │
│  │ • 快速搜索   │ │ • 价格分析   │ │ • 实体提取           │ │
│  │ • 详细查询   │ │ • K-means聚类│ │ • 渠道识别           │ │
│  │ • 候选获取   │ │ • 统计计算   │ │ • 可视化生成         │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      原服务层                                │
│  RAGService / DialogueManager / ConversationManager         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 新增文件结构

```
app/
├── mcp/
│   ├── __init__.py          # 模块初始化
│   ├── tools.py             # MCP工具定义和实现
│   ├── server.py            # MCP Server
│   └── agent.py             # Agent实现
├── api/
│   ├── endpoints.py         # 原有接口（已存在）
│   └── mcp_endpoints.py     # MCP相关API端点
└── api_server.py            # 主入口（修改）
```

---

## 三、MCP 工具设计

### 3.1 工具列表

| 工具名 | 功能 | 输入 | 输出 |
|-------|------|------|------|
| `quick_search_materials` | 快速搜索材料候选 | keyword, limit | 候选列表、相关关键词 |
| `query_price_data` | 详细价格查询 | channel, material_name, province... | 价格数据、统计信息 |
| `analyze_prices` | 价格数据分析 | price_data | K-means聚类、价格区间、统计指标 |
| `get_material_candidates` | 获取候选材料 | keyword, channel | 候选材料列表 |
| `extract_entities` | 实体提取 | question | materialName, province, city等 |
| `identify_channel` | 渠道识别 | question | channel, confidence |

### 3.2 工具返回格式（标准化）

```python
@dataclass
class ToolResult:
    status: ToolStatus          # success / error / partial
    data: Any                   # 工具返回数据
    message: str                # 可读的消息
    metadata: Dict[str, Any]    # 元数据
    suggested_next_steps: List[str]  # 建议的下一步操作
```

---

## 四、Agent 对话流程

### 4.1 状态机

```
                    ┌─────────────┐
         ┌──────────│    IDLE     │◄──────────┐
         │          └──────┬──────┘           │
         │                 │ 首次查询          │
         │                 ▼                  │
         │      ┌─────────────────────┐       │
         │      │  CONFIRMING_CANDIDATE│       │
         │      │   （确认候选材料）   │       │
         │      └──────────┬──────────┘       │
         │                 │                  │
         │   候选确认      ▼                  │
         │      ┌─────────────────────┐       │
         └──────│  COLLECTING_ENTITIES │       │
   （新查询）    │   （收集实体信息）   │       │
                └──────────┬──────────┘       │
                           │                  │
           实体收集完成     ▼                  │
                ┌─────────────────────┐       │
                │   READY_TO_QUERY    │───────┘
                │   （准备查询）       │
                └──────────┬──────────┘
                           │
              用户确认     ▼
                ┌─────────────────────┐
                │     QUERYING        │
                │   （执行查询）       │
                └──────────┬──────────┘
                           │
              查询完成     ▼
                ┌─────────────────────┐
                │    COMPLETED        │
                │   （查询完成）       │
                └─────────────────────┘
```

### 4.2 首次查询流程

```
用户: "查一下钢筋的价格"

Agent:
  1. 提取实体: {materialName: "钢筋"}
  2. 识别渠道: "information_price"
  3. 快速搜索: 获取候选材料
  
响应:
  "收到，您想查询「钢筋」的价格信息。
   查询渠道：信息价
   
   ✓ 找到精确匹配的材料：钢筋
   
   📋 为了给您提供准确的价格信息，还需要：
     • 省份
     • 城市
   
   快捷选项: [广东省] [江苏省] [北京市] [上海市]"
```

### 4.3 后续查询流程

```
用户: "广东省深圳市的"

Agent:
  1. 提取实体: {province: "广东省", city: "深圳市"}
  2. 合并实体: {materialName: "钢筋", province: "广东省", city: "深圳市"}
  3. 检查完整性: OK
  4. 执行查询
  5. 数据分析: K-means聚类等

响应:
  "✅ 「钢筋」在广东省深圳市的价格查询完成！
   📊 共查询到 156 条价格数据
   
   💰 推荐价格（单位：kg）：
     • 24.50 元
   
   📈 价格区间：23.00 - 26.00 元
   📊 平均价格：24.35 元"
```

---

## 五、API 接口

### 5.1 Agent 对话接口

```http
POST /api/v1/agent/chat
Content-Type: application/json

{
  "message": "查一下钢筋的价格",
  "session_id": null  // 首次可为空
}

Response:
{
  "success": true,
  "session_id": "abc123",
  "state": "collecting_entities",
  "turn_count": 1,
  "message": "收到，您想查询...",
  "data": {
    "entities": {"materialName": "钢筋"},
    "candidates": [...],
    "missing_fields": ["province", "city"]
  },
  "suggested_actions": ["广东省", "江苏省", "北京市"]
}
```

### 5.2 MCP 工具调用接口

```http
POST /api/v1/mcp/tools/call
Content-Type: application/json

{
  "tool_name": "quick_search_materials",
  "parameters": {
    "keyword": "钢筋",
    "limit": 10
  }
}

Response:
{
  "success": true,
  "result": {
    "status": "success",
    "data": {
      "candidates": [...],
      "related_keywords": [...]
    },
    "message": "找到 8 个相关材料",
    "suggested_next_steps": [...]
  }
}
```

### 5.3 工具列表接口

```http
GET /api/v1/mcp/tools

Response:
{
  "success": true,
  "tools": [
    {
      "name": "quick_search_materials",
      "description": "快速搜索材料...",
      "parameters": {...},
      "required": ["keyword"]
    },
    ...
  ],
  "count": 6
}
```

---

## 六、使用示例

### 6.1 简单对话示例

```python
import requests

# 首次查询
response = requests.post("/api/v1/agent/chat", json={
    "message": "查一下钢筋的价格"
})
data = response.json()
session_id = data["session_id"]
# 响应提示需要省份和城市

# 补充省份和城市
response = requests.post("/api/v1/agent/chat", json={
    "message": "广东省深圳市的",
    "session_id": session_id
})
# 响应包含价格分析结果
```

### 6.2 直接使用工具

```python
# 获取候选材料
candidates = requests.post("/api/v1/mcp/tools/call", json={
    "tool_name": "get_material_candidates",
    "parameters": {"keyword": "混凝土", "limit": 5}
})

# 查询价格
price_data = requests.post("/api/v1/mcp/tools/call", json={
    "tool_name": "query_price_data",
    "parameters": {
        "channel": "information_price",
        "material_name": "钢筋",
        "province": "广东省",
        "city": "深圳市"
    }
})

# 分析价格
analysis = requests.post("/api/v1/mcp/tools/call", json={
    "tool_name": "analyze_prices",
    "parameters": {
        "price_data": price_data.json()["result"]["data"]["price_data"]
    }
})
```

---

## 七、与现有系统的集成

### 7.1 兼容现有接口

现有的 `/api/v1/query/stream` 和 `/api/v1/query/price` 接口**保持不变**，新增的 MCP/Agent 接口作为**扩展功能**提供。

### 7.2 数据复用

MCP 工具直接复用现有的服务：
- `RAGService.extract_entities()` → `extract_entities` 工具
- `RAGService.process_price_recommendation()` → `query_price_data` 工具
- `price_tools.analyze_by_unit()` → `analyze_prices` 工具

### 7.3 会话管理

Agent 使用独立的上下文管理（`AgentContext`），但可与现有的 `ConversationManager` 集成，实现跨功能的状态同步。

---

## 八、扩展指南

### 8.1 添加新工具

1. 在 `tools.py` 中定义工具
```python
def _define_new_tool(self) -> ToolDefinition:
    return ToolDefinition(
        name="new_tool",
        description="...",
        parameters={...},
        required=[...]
    )

async def new_tool(self, param1: str) -> ToolResult:
    # 实现工具逻辑
    return ToolResult(status=ToolStatus.SUCCESS, data=...)
```

2. 在 `server.py` 中注册
```python
self._tool_map["new_tool"] = self.tools.new_tool
```

### 8.2 自定义Agent行为

修改 `agent.py` 中的状态处理逻辑：
```python
async def _handle_custom_state(self, context, message):
    # 自定义处理逻辑
    pass
```

---

## 九、总结

### 9.1 核心收益

1. **标准化工具接口**: 所有功能通过MCP标准接口暴露，易于集成和扩展
2. **智能多轮对话**: Agent自主决策，无需硬编码对话流程
3. **更好的用户体验**: 首次查询提供候选和补充建议，降低使用门槛
4. **可观测性**: Agent思考过程可追踪，便于调试和优化

### 9.2 后续优化方向

1. **LLM驱动的Agent**: 使用LLM进行意图识别和工具选择
2. **长期记忆**: 集成长期记忆，记住用户偏好
3. **更多工具**: 添加价格预测、趋势分析等高级工具
4. **可视化增强**: 自动生成价格走势图、对比图等

---

**相关文件**:
- `app/mcp/tools.py`
- `app/mcp/server.py`
- `app/mcp/agent.py`
- `app/api/mcp_endpoints.py`
