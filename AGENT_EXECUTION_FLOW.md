# Agent接口执行流程详解

> 文档版本: v1.0  
> 更新时间: 2026-04-09  
> 对应代码: `app/mcp/agent.py`

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Agent执行流程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   用户请求                                                                  │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  process_message()                                                │     │
│   │  ├─ 获取/创建上下文 (session)                                      │     │
│   │  └─ 状态路由分发                                                   │     │
│   │       ├─ IDLE → _handle_first_turn()                              │     │
│   │       ├─ CONFIRMING_CANDIDATE → _handle_candidate_confirmation()  │     │
│   │       ├─ COLLECTING_ENTITIES → _handle_entity_collection()        │     │
│   │       ├─ READY_TO_QUERY → _handle_ready_state()                   │     │
│   │       ├─ COMPLETED/QUERYING → _handle_new_or_followup()           │     │
│   │       └─ default → _handle_default()                              │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│      │                                                                      │
│      ▼                                                                      │
│   MCP工具调用                                                               │
│      ├─ extract_entities      (实体提取)                                    │
│      ├─ identify_channel      (渠道识别)                                    │
│      ├─ quick_search_materials(候选搜索)                                    │
│      ├─ query_price_data      (价格查询)                                    │
│      └─ analyze_prices        (价格分析)                                    │
│      │                                                                      │
│      ▼                                                                      │
│   响应构建                                                                  │
│      └─ _build_agent_response()                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、Agent状态机

### 2.1 状态定义

```python
class AgentState(str, Enum):
    IDLE = "idle"                           # 空闲/初始状态
    COLLECTING_ENTITIES = "collecting_entities"      # 收集实体
    CONFIRMING_CANDIDATE = "confirming_candidate"    # 确认候选材料
    READY_TO_QUERY = "ready_to_query"                # 准备查询
    QUERYING = "querying"                            # 查询中
    ANALYZING = "analyzing"                          # 分析中
    COMPLETED = "completed"                          # 完成
```

### 2.2 状态转换图

```
                              ┌─────────────┐
                              │    IDLE     │
                              │   (初始)    │
                              └──────┬──────┘
                                     │
                                     │ 用户首次输入
                                     │ "查一下钢筋价格"
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        _handle_first_turn()                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Step 1: extract_entities                                      │  │
│  │  Step 2: identify_channel                                      │  │
│  │  Step 3: quick_search_materials                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
         ┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
         │  无精确匹配      │ │  有精确匹配   │ │  提取失败    │
         │  需要候选确认    │ │  继续检查    │ │  重新询问    │
         └────────┬────────┘ └──────┬───────┘ └──────┬───────┘
                  │                 │                │
                  ▼                 │                ▼
         ┌─────────────────┐       │        ┌──────────────┐
         │CONFIRMING_CANDI-│       │        │  COLLECTING  │
         │    DATE         │       │        │  _ENTITIES   │
         │  (确认候选)      │       │        │ (询问材料名) │
         └────────┬────────┘       │        └──────────────┘
                  │                │                ▲
                  │ 用户选择候选    │                │
                  ▼                │                │
         ┌─────────────────┐       │                │
         │  候选已确认      │       │                │
         └────────┬────────┘       │                │
                  │                │                │
                  └────────────────┼────────────────┘
                                   │
                                   ▼
                    检查缺失字段 [materialName, province, city]
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              │              ▼
         ┌─────────────────┐      │      ┌──────────────┐
         │  有缺失字段      │      │      │  字段完整    │
         │  (province/city) │      │      │              │
         └────────┬────────┘      │      └──────┬───────┘
                  │               │             │
                  ▼               │             ▼
         ┌─────────────────┐      │      ┌──────────────┐
         │ COLLECTING_ENT- │      │      │ READY_TO_    │
         │   ITIES         │      │      │   QUERY      │
         │ (收集缺失信息)   │◄─────┘      │  (准备查询)   │
         └────────┬────────┘             └──────┬───────┘
                  │                             │
                  │ 用户补充信息                 │ 用户确认查询
                  │ "广东省深圳市"              │ "确认查询"
                  │                             │
                  └─────────────────────────────┘
                                                │
                                                ▼
                                     ┌────────────────────┐
                                     │     QUERYING       │
                                     │  _execute_price_   │
                                     │   query()          │
                                     │                    │
                                     │ ├─query_price_data │
                                     │ └─analyze_prices   │
                                     └────────┬───────────┘
                                              │
                                              ▼
                                     ┌────────────────────┐
                                     │    COMPLETED       │
                                     │   (查询完成)        │
                                     └────────┬───────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
         ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
         │  用户问新问题    │       │ 用户问跟进问题   │       │  用户查新材料   │
         │  "价格走势如何"  │       │  "最便宜多少"   │       │ "再查水泥价格" │
         └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
                  │                         │                         │
                  ▼                         ▼                         ▼
         ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
         │  _handle_follow-│       │  _handle_follow-│       │  重置上下文     │
         │  up_question()  │       │  up_question()  │       │  → IDLE         │
         └─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## 三、详细执行流程

### 3.1 主入口: process_message()

```python
async def process_message(self, user_message: str, session_id: Optional[str] = None):
    """
    1. 获取或创建上下文 (AgentContext)
    2. 增加回合计数
    3. 根据当前状态路由到对应处理器
    """
```

**处理逻辑**:
```
if turn == 1 or state == IDLE:
    → _handle_first_turn()          # 首次查询
elif state == CONFIRMING_CANDIDATE:
    → _handle_candidate_confirmation()  # 确认候选
elif state == COLLECTING_ENTITIES:
    → _handle_entity_collection()   # 收集实体
elif state == READY_TO_QUERY:
    → _handle_ready_state()         # 准备查询
elif state in [COMPLETED, QUERYING]:
    → _handle_new_or_followup()     # 新查询或跟进
else:
    → _handle_default()             # 默认处理
```

---

### 3.2 首次查询: _handle_first_turn()

```
用户输入: "查一下钢筋的价格"

执行流程:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 提取实体                                                 │
│ Tool: extract_entities                                          │
│ Input: {"question": "查一下钢筋的价格"}                           │
│ Output: {"materialName": "钢筋"}                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 识别渠道                                                 │
│ Tool: identify_channel                                          │
│ Input: {"question": "查一下钢筋的价格"}                           │
│ Output: {"channel": "information_price", "confidence": 0.5}       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 搜索材料候选                                             │
│ Tool: quick_search_materials                                    │
│ Input: {"keyword": "钢筋", "limit": 10}                           │
│ Output: {                                                       │
│   "candidates": [                                               │
│     {"name": "钢筋", "match_score": 1.0, "exact_match": true},    │
│     {"name": "螺纹钢", "match_score": 0.8, "exact_match": false} │
│   ],                                                            │
│   "related_keywords": ["HRB400", "盘螺"]                         │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    判断是否有精确匹配
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             │             ▼
        ┌───────────┐         │     ┌───────────┐
        │ 有精确匹配 │         │     │ 无精确匹配 │
        └─────┬─────┘         │     └─────┬─────┘
              │               │           │
              │               │           ▼
              │               │    ┌─────────────────┐
              │               │    │ 需要用户确认候选 │
              │               │    │ state = CONFIRMING_
              │               │    │   CANDIDATE      │
              │               │    └─────────────────┘
              │               │
              ▼               │
    ┌─────────────────┐       │
    │ 检查其他缺失字段 │       │
    │ [province, city]│       │
    └────────┬────────┘       │
             │                │
     ┌───────┴───────┐        │
     │               │        │
     ▼               ▼        │
┌─────────┐   ┌────────────┐  │
│还有缺失  │   │ 字段完整    │  │
│字段     │   │            │  │
└────┬────┘   └─────┬──────┘  │
     │              │         │
     ▼              ▼         │
┌────────────┐ ┌────────────┐ │
│ COLLECTING │ │ READY_TO_  │ │
│ _ENTITIES  │ │   QUERY    │ │
└────────────┘ └────────────┘ │
```

---

### 3.3 候选确认: _handle_candidate_confirmation()

```
当前状态: CONFIRMING_CANDIDATE
用户输入: "1" 或 "钢筋"

执行流程:
1. 解析用户选择 (_parse_candidate_selection)
   ├─ 尝试匹配数字: "1" → candidates[0]
   ├─ 尝试匹配名称: "钢筋" → 查找对应候选
   └─ 模糊匹配

2. 如果解析成功:
   ├─ 更新 context.selected_candidate
   ├─ 更新 context.collected_entities["materialName"]
   ├─ 检查缺失字段
   │   ├─ 有缺失 → COLLECTING_ENTITIES
   │   └─ 无缺失 → READY_TO_QUERY

3. 如果解析失败:
   └─ 重新返回候选选择提示
```

---

### 3.4 实体收集: _handle_entity_collection()

```
当前状态: COLLECTING_ENTITIES
用户输入: "广东省深圳市的"

执行流程:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 提取新实体                                               │
│ Tool: extract_entities                                          │
│ Input: {"question": "广东省深圳市的"}                             │
│ Output: {"province": "广东省", "city": "深圳市"}                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 合并实体到上下文                                          │
│ Before: {"materialName": "钢筋"}                                  │
│ After:  {"materialName": "钢筋", "province": "广东省", "city": "深圳市"}
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    检查是否还有缺失字段
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
            ┌───────────┐       ┌───────────┐
            │ 还有缺失   │       │ 字段完整   │
            │ 继续询问   │       │           │
            └───────────┘       └─────┬─────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ READY_TO_    │
                              │   QUERY      │
                              └──────────────┘
```

---

### 3.5 准备查询: _handle_ready_state()

```
当前状态: READY_TO_QUERY
用户输入: 需要解析意图

意图解析 (_parse_intent):
┌─────────────────────────────────────────────────────────────┐
│ confirm_keywords = ["确认", "是的", "没错", "对", "好", "ok"]    │
│ modify_keywords = ["修改", "不对", "错了", "换", "改"]          │
│                                                             │
│ 用户输入: "确认查询" → intent = "confirm"                     │
│ 用户输入: "修改省份" → intent = "modify"                      │
│ 用户输入: "等等"    → intent = "unknown"                      │
└─────────────────────────────────────────────────────────────┘

处理分支:
├─ intent == "confirm" → _execute_price_query()
├─ intent == "modify"  → COLLECTING_ENTITIES
└─ intent == "unknown" → 再次显示 READY_TO_QUERY 提示
```

---

### 3.6 执行查询: _execute_price_query()

```
状态流转: READY_TO_QUERY → QUERYING → COMPLETED

执行流程:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 查询价格数据                                             │
│ Tool: query_price_data                                          │
│ Input: {                                                        │
│   "channel": "information_price",                               │
│   "material_name": "钢筋",                                       │
│   "province": "广东省",                                          │
│   "city": "深圳市"                                               │
│ }                                                               │
│ Output: {                                                       │
│   "success": true,                                              │
│   "total_count": 156,                                           │
│   "price_data": [...],                                          │
│   "metadata": {...}                                             │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 分析价格数据 (如果查询成功)                               │
│ Tool: analyze_prices                                            │
│ Input: {"price_data": [...]}                                    │
│ Output: {                                                       │
│   "results": {                                                  │
│     "kg": {                                                     │
│       "total_count": 156,                                       │
│       "mean_price": 24.35,                                      │
│       "recommend_kmeans": {                                     │
│         "mode": "single",                                       │
│         "prices": [24.5]                                        │
│       }                                                         │
│     }                                                           │
│   },                                                            │
│   "most_common_unit": "kg"                                      │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 构建结果响应                                             │
│ 生成自然语言回复:                                                │
│ "✅ 「钢筋」价格查询完成！                                       │
│  📊 共查询到 156 条价格数据                                      │
│  💰 推荐价格（单位：kg）：24.50 元                                │
│  📈 价格区间：23.00 - 26.00 元                                   │
│  📊 平均价格：24.35 元"                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3.7 新查询检测: _handle_new_or_followup()

```
当前状态: COMPLETED
用户输入: 需要判断是否为新查询

检测逻辑:
1. 提取实体，检查材料名称是否变化
   ├─ 新材料名称 ≠ 旧材料名称 → 是新查询
   └─ 相同 → 继续检测

2. 检测切换信号词
   ["另外", "还有", "再查", "换一个", "其他", "新的", "别的材料"]
   ├─ 包含信号词 → 是新查询
   └─ 不包含 → 是跟进问题

处理分支:
├─ 是新查询:
│   ├─ 重置上下文 (清空entities, candidates, result)
│   ├─ state = IDLE
│   └─ 调用 _handle_first_turn()
│
└─ 是跟进问题:
    └─ 调用 _handle_followup_question()
```

---

## 四、工具调用链汇总

### 4.1 各处理器工具调用情况

| 处理器 | 调用的工具 | 说明 |
|-------|-----------|------|
| `_handle_first_turn` | extract_entities → identify_channel → quick_search_materials | 首次查询必需 |
| `_handle_candidate_confirmation` | 无 | 纯逻辑处理 |
| `_handle_entity_collection` | extract_entities | 提取补充信息 |
| `_handle_ready_state` | 无 | 纯逻辑处理 |
| `_execute_price_query` | query_price_data → analyze_prices | 查询和分析 |
| `_handle_new_or_followup` | extract_entities | 检测新材料 |

### 4.2 完整对话示例工具链

```
对话示例: 查询钢筋价格

Turn 1: "查一下钢筋价格"
├─ extract_entities       → {materialName: "钢筋"}
├─ identify_channel       → {channel: "information_price"}
└─ quick_search_materials → {candidates: [...]}

Turn 2: "广东省深圳市"
└─ extract_entities       → {province: "广东省", city: "深圳市"}

Turn 3: "确认查询"
├─ query_price_data       → {price_data: [...], total_count: 156}
└─ analyze_prices         → {results: {...}, most_common_unit: "kg"}

Turn 4: "再查一下水泥"
└─ extract_entities       → {materialName: "水泥"} (检测到新材料)
→ 重置上下文，回到Turn 1流程

Total: 一次完整查询平均调用 5-6 个工具
```

---

## 五、响应数据结构

### 5.1 标准响应格式

```json
{
  "success": true,
  "session_id": "abc123",
  "state": "collecting_entities",
  "turn_count": 2,
  "message": "请问是哪个省份的材料？",
  "data": {
    "entities": {
      "materialName": "钢筋"
    },
    "channel": "information_price",
    "candidates": [...],
    "selected_candidate": "钢筋",
    "query_result": null          // 查询完成后填充
  },
  "suggested_actions": [
    "广东省", "江苏省", "浙江省", "北京市", "上海市"
  ]
}
```

### 5.2 各状态典型响应

| 状态 | message示例 | suggested_actions示例 |
|-----|------------|---------------------|
| `IDLE`→`COLLECTING` | "请告诉我您想查询什么材料？" | ["钢筋", "水泥", "混凝土"] |
| `CONFIRMING_CANDIDATE` | "找到多个相关材料，请选择：\n1. 钢筋\n2. 螺纹钢" | ["1", "2", "3"] |
| `COLLECTING_ENTITIES` | "请问是哪个省份的材料？" | ["广东省", "江苏省", "北京市"] |
| `READY_TO_QUERY` | "查询条件已收集完毕！\n📋 材料：钢筋\n📍 地区：广东省深圳市\n是否确认查询？" | ["✅ 确认查询", "🔄 修改条件"] |
| `COMPLETED` | "✅ 「钢筋」价格查询完成！\n📊 共查询到 156 条数据\n💰 推荐价格：24.50 元/kg" | ["查询新材料", "查看详细数据"] |

---

## 六、性能特点

### 6.1 耗时分析

```
单次请求处理耗时:
├─ 实体提取:           50-100ms  (LLM调用)
├─ 渠道识别:           50-100ms  (LLM调用)
├─ 候选搜索:          200-500ms  (API查询)
├─ 价格查询:          500ms-2s   (API查询)
├─ 价格分析:          100-300ms  (本地计算)
└─ 响应构建:           <50ms      (本地处理)

单次完整查询总耗时: 1-3s
多轮对话总耗时:     3-6s (3轮对话)
```

### 6.2 优化点

1. **缓存候选结果**: 相同材料名的候选可以缓存
2. **并行调用**: entity提取和channel识别可以并行
3. **流式响应**: 当前非流式，可以改为SSE流式

---

## 七、总结

### Agent核心特点

1. **状态驱动**: 基于状态机的清晰流程控制
2. **工具编排**: 自动决策调用哪些MCP工具
3. **上下文保持**: 会话级别保持查询状态
4. **智能提示**: 自动生成下一步操作提示
5. **多轮对话**: 支持复杂的渐进式信息收集

### 适用场景

- ✅ 自然语言交互的聊天式查询
- ✅ 需要渐进式收集信息的场景
- ✅ 需要候选材料推荐的场景
- ⚠️ 对响应速度要求极高的场景（建议用标准查询）
