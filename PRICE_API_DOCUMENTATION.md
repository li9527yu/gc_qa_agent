# 材料价格查询 API 文档

> 文档版本: v1.0  
> 更新时间: 2026-04-09  
> 适用项目: easy-rag

---

## 一、价格查询逻辑流程

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              前端层                                       │
│   Streamlit / Web / 其他客户端                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              API 层 (FastAPI)                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ /query/price    │  │ /query/dialogue │  │ /query/price/direct     │  │
│  │ 标准价格查询    │  │ 对话式查询      │  │ 直接查询（结构化数据）   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           服务层 (RAGService)                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ 渠道识别        │  │ 实体提取        │  │ 价格查询与分析          │  │
│  │ identify_channel│  │ extract_entities│  │ process_price_recommend │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           数据层                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ 信息价 API      │  │ 厂商报价 API    │  │ 智诚信息价 API          │  │
│  │ (infor)         │  │ (factory)       │  │ (zhichengInfo)          │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 价格查询核心流程

```
用户输入: "查一下广东省深圳市的钢筋价格"

Step 1: 渠道识别 (identify_channel)
├─ LLM 尝试精确识别渠道关键词
├─ 如果失败，启用智能推断器 (ChannelInferencer)
│   ├─ 基于特征词加权评分
│   ├─ 基于实体特征推断（如brand存在→倾向厂商报价）
│   └─ 默认兜底策略（有材料名→默认信息价）
└─ 返回: channel_type, confidence, reason

Step 2: 实体提取 (extract_entities)
├─ 使用 LLM + Prompt Template
├─ 提取字段:
│   ├─ materialName (材料名称) - 必需
│   ├─ province (省份) - 可选
│   ├─ city (城市) - 可选
│   ├─ materialModelSpec (规格型号) - 可选
│   ├─ brand (品牌) - 可选
│   └─ start/endReleaseDate (时间范围) - 可选
└─ 返回: parsed_entities (JSON格式)

Step 3: 数据查询 (_query_price_data)
├─ 根据渠道类型调用对应 API:
│   ├─ information_price → _get_infor_material()
│   ├─ manufacturer_price → _get_factory_material()
│   └─ zc_price → _get_zhicheng_info_material()
├─ 发送 HTTP POST 请求到后端价格服务
└─ 返回: 原始价格数据列表

Step 4: 数据过滤 (filter_items)
├─ 使用模糊匹配算法过滤材料名称
├─ 参数: alpha=0.4, T_keep=0.75, T_drop=0.50
└─ 返回: 过滤后的数据列表

Step 5: 价格分析 (analyze_by_unit)
├─ 按计量单位(unit)分组
├─ 每组进行统计分析:
│   ├─ 基础统计: count, min, max, mean, median
│   ├─ 异常值检测: IQR方法
│   └─ K-means聚类: 
│       ├─ 判断是单档还是两档价格
│       ├─ 计算推荐价格
│       └─ 生成判定理由
└─ 返回: price_analysis (包含各单位的分析结果)

Step 6: 结果格式化 (parse_material_response)
├─ 生成 Markdown 表格
└─ 返回: 结构化响应数据

Step 7: LLM生成自然语言回答 (stream_price_predict)
├─ 将分析结果传给 LLM
├─ 流式生成回答文本
└─ 返回: 完整回答 + metadata
```

### 1.3 对话式查询流程

```
用户: "查一下钢筋的价格"                            [首次请求]
  │
  ▼
系统: {
  "status": "collecting",                           [状态: 收集中]
  "entities": {"materialName": "钢筋"},             [已收集实体]
  "missing_fields": ["province", "city"],           [缺失字段]
  "response_text": "请问是哪个省份的材料？",        [系统回复]
  "quick_options": ["广东省", "江苏省", ...]        [快捷选项]
}
  │
  ▼
用户: "广东省深圳市的"                              [补充信息]
  │
  ▼
系统: {
  "status": "ready",                                [状态: 准备就绪]
  "entities": {                                     [完整实体]
    "materialName": "钢筋",
    "province": "广东省",
    "city": "深圳市"
  },
  "can_query": true,                                [可执行查询]
  "response_text": "信息已收集完整，正在查询..."
}
  │
  ▼
用户: 调用 /query/dialogue/execute                  [执行查询]
  │
  ▼
系统: 返回价格查询结果                              [最终结果]
```

---

## 二、API 接口列表

### 2.1 标准价格查询接口

#### `POST /api/v1/query/price`

**功能**: 标准价格推荐查询（流式响应）

**请求参数**:
```json
{
  "question": "从信息价查钢筋的价格",    // 必填, 用户问题
  "session_id": "abc123",               // 可选, 会话ID（多轮对话）
  "force_new": false                    // 可选, 强制创建新会话
}
```

**响应格式**: `text/plain` (流式)

**响应流程**:
```
1. 渠道识别结果 (JSON)
   [END]
2. 实体解析结果 (JSON)
   [END]
3. LLM生成的回答文本 (流式)
   [END]
4. 最终元数据 (JSON)
```

**完整响应示例**:
```
识别到渠道类型: {"channel": "information_price", "confidence": 1.0, "reason": "LLM识别"}

[END]

解析到实体结果: {"materialName": "钢筋", "province": "广东省", "city": "深圳市"}

[END]

根据查询结果，广东省深圳市钢筋的价格分析如下：
共查询到 156 条价格数据...
...

[END]

{
  "text": "完整回答文本...",
  "intent": "price_recommendation",
  "channel": "information_price",
  "metadata": {
    "channel": "information_price",
    "entities": {"materialName": "钢筋", ...},
    "price_analysis": {
      "kg": {
        "total_count": 156,
        "valid_count": 156,
        "price_range": [23.0, 26.0],
        "mean_price": 24.35,
        "median_price": 24.5,
        "recommend_kmeans": {
          "mode": "single",
          "prices": [24.5],
          "reason": "..."
        }
      }
    },
    "md_table": "| 编号 | 材料名称 | ... |",
    "performance": {...}
  },
  "success": true,
  "session_id": "abc123",
  "conversation_type": "price_query",
  "price_data": [...]    // 原始价格数据列表
}
```

**字段说明**:
| 字段 | 类型 | 说明 |
|-----|------|------|
| `channel` | string | 查询渠道: information_price/manufacturer_price/zc_price |
| `entities` | object | 提取的查询实体 |
| `price_analysis` | object | 价格分析结果（按单位分组） |
| `md_table` | string | Markdown格式的价格表格 |
| `performance` | object | 性能统计信息 |
| `price_data` | array | 原始价格数据（前N条） |

---

### 2.2 对话式查询接口

#### `POST /api/v1/query/dialogue`

**功能**: 渐进式实体补全的对话查询

**请求参数**:
```json
{
  "session_id": null,                   // 首次可为空
  "user_input": "查一下钢筋的价格"      // 用户输入
}
```

**响应**:
```json
{
  "session_id": "abc12345",
  "status": "collecting",               // collecting/ready/completed
  "entities": {"materialName": "钢筋"},
  "channel": "information_price",
  "channel_info": {
    "channel": "information_price",
    "confidence": 0.5,
    "reason": "未明确指定渠道，默认查询信息价",
    "inferred": true
  },
  "is_complete": false,
  "can_query": false,                   // true时可执行查询
  "response_text": "请问是哪个省份的材料？",
  "next_question": {
    "field": "province",
    "label": "省份",
    "prompt": "请问是哪个省份的材料？"
  },
  "quick_options": [
    {"text": "广东省", "value": "广东省"},
    {"text": "江苏省", "value": "江苏省"}
  ],
  "price_result": {...}                 // 查询完成后返回
}
```

---

#### `POST /api/v1/query/dialogue/execute`

**功能**: 执行对话会话中已收集条件的查询

**请求参数**:
```json
{
  "session_id": "abc12345"
}
```

**响应**:
```json
{
  "success": true,
  "session_id": "abc12345",
  "entities": {
    "materialName": "钢筋",
    "province": "广东省",
    "city": "深圳市"
  },
  "price_result": {
    "answer": "...",
    "success": true,
    "total_count": 156,
    "metadata": {...}
  }
}
```

---

#### `GET /api/v1/query/dialogue/{session_id}`

**功能**: 获取对话会话状态

**响应**:
```json
{
  "session_id": "abc12345",
  "status": "collecting",
  "entities": {...},
  "missing_fields": [...],
  "channel": "information_price",
  "history_count": 3
}
```

---

#### `DELETE /api/v1/query/dialogue/{session_id}`

**功能**: 清除对话会话

---

### 2.3 直接价格查询接口

#### `POST /api/v1/query/price/direct`

**功能**: 使用结构化材料数据直接查询价格

**请求参数**:
```json
{
  "datatype": "informaterial",          // 数据类型
  "question": "查找铝合金门窗型材的价格",
  "list": [{                            // 材料信息列表
    "materialName": "铝合金门窗型材",
    "province": "广东省",
    "city": "深圳市",
    "materialModelSpec": "银白氧化",
    "unit": "kg",
    ...
  }]
}
```

**响应格式**: `text/plain` (流式)

**响应示例**:
```
识别到渠道类型: information_price

[END]

解析到实体结果: {"materialName": "铝合金门窗型材", "province": "广东省", "city": "深圳市"}

[END]

广东省深圳市铝合金门窗型材最近三年平均价格约为 22 元/kg
...

[END]

{
  "text": "...",
  "intent": "price_recommendation",
  "channel": "information_price",
  "metadata": {...},
  "success": true,
  "price_data": [...]
}
```

---

### 2.4 服务状态接口

#### `GET /api/v1/service/status`

**功能**: 获取服务状态（包括知识库重建状态）

**响应**:
```json
{
  "is_rebuilding": false,
  "rebuild_progress": {
    "stage": "idle",
    "message": "就绪",
    "percent": 100
  },
  "retriever_ready": true,
  "llm_ready": true,
  "corpus_count": 42
}
```

**说明**:
- `is_rebuilding`: true时表示知识库正在重建，知识问答功能不可用
- 价格推荐功能不受知识库重建影响，始终可用

---

### 2.5 新增 MCP/Agent 接口

#### `POST /api/v1/agent/chat`

**功能**: Agent多轮对话（推荐用于新开发）

**请求**:
```json
{
  "message": "查一下钢筋的价格",
  "session_id": null
}
```

**响应**:
```json
{
  "success": true,
  "session_id": "abc123",
  "state": "collecting_entities",
  "turn_count": 1,
  "message": "收到，您想查询「钢筋」的价格信息...",
  "data": {
    "entities": {"materialName": "钢筋"},
    "candidates": [...],
    "missing_fields": ["province", "city"]
  },
  "suggested_actions": ["广东省", "江苏省", "北京市"]
}
```

---

## 三、前端集成指南

### 3.1 标准查询模式

```javascript
// 1. 简单价格查询
async function queryPrice(question) {
  const response = await fetch('/api/v1/query/price', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question})
  });
  
  // 处理流式响应
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    // 解析 [END] 分隔的数据块
    const parts = chunk.split('\n[END]\n');
    
    for (const part of parts) {
      if (part.trim().startsWith('{')) {
        // JSON 元数据
        const meta = JSON.parse(part);
        console.log('Metadata:', meta);
      } else if (part.trim()) {
        // 文本内容
        console.log('Text:', part);
      }
    }
  }
}
```

### 3.2 对话式查询模式

```javascript
// 2. 对话式查询（渐进式收集）
class PriceDialogue {
  constructor() {
    this.sessionId = null;
  }
  
  async sendMessage(userInput) {
    const response = await fetch('/api/v1/query/dialogue', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: this.sessionId,
        user_input: userInput
      })
    });
    
    const result = await response.json();
    this.sessionId = result.session_id;
    
    // 根据状态处理
    if (result.status === 'collecting') {
      // 显示提示和快捷选项
      this.showPrompt(result.response_text, result.quick_options);
    } else if (result.status === 'ready' && result.can_query) {
      // 执行查询
      this.executeQuery();
    }
    
    return result;
  }
  
  async executeQuery() {
    const response = await fetch('/api/v1/query/dialogue/execute', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: this.sessionId})
    });
    
    const result = await response.json();
    this.displayPriceResult(result.price_result);
    return result;
  }
}
```

### 3.3 Agent 对话模式（推荐）

```javascript
// 3. Agent 对话（最智能，推荐新开发使用）
class PriceAgent {
  constructor() {
    this.sessionId = null;
  }
  
  async chat(message) {
    const response = await fetch('/api/v1/agent/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message,
        session_id: this.sessionId
      })
    });
    
    const result = await response.json();
    this.sessionId = result.session_id;
    
    // Agent会自动处理多轮对话逻辑
    // 前端只需要显示 message 和 suggested_actions
    this.displayMessage(result.message);
    this.displaySuggestions(result.suggested_actions);
    
    // 如果有价格数据，显示图表和表格
    if (result.data?.query_result?.price_data) {
      this.displayPriceChart(result.data.query_result.price_data);
    }
    
    return result;
  }
}

// 使用示例
const agent = new PriceAgent();

// 第一次对话
await agent.chat("查一下钢筋的价格");
// 响应: "收到，您想查询「钢筋」的价格信息...请问是哪个省份？"

// 第二次对话（自动使用同一个session）
await agent.chat("广东省深圳市的");
// 响应: 直接返回价格分析结果
```

---

## 四、数据结构说明

### 4.1 价格分析结果结构

```typescript
interface PriceAnalysis {
  total_count: number;          // 原始记录数
  valid_count: number;          // 剔除异常值后数量
  price_range: [number, number]; // 价格区间 [min, max]
  mean_price: number;           // 平均价格
  median_price: number;         // 中位数价格
  recommend_kmeans: {
    mode: "single" | "two-tier"; // 单档/两档推荐
    prices: number[];           // 推荐价格列表
    reason: string;             // 判定理由
  };
}

interface UnitAnalysisResult {
  results: {
    [unit: string]: PriceAnalysis;  // 按单位分组的分析结果
  };
  most_common_unit: string;      // 最常见的计量单位
  most_common_records: Array<{
    materialName: string;
    materialModelSpec: string;
    price: number;
    unit: string;
    province: string;
    city: string;
    releaseDate: string;
    // ... 其他字段
  }>;
}
```

### 4.2 实体字段说明

| 字段名 | 类型 | 必需 | 说明 |
|-------|------|------|------|
| `materialName` | string | ✅ | 材料名称（如：钢筋、水泥） |
| `province` | string | ❌ | 省份（如：广东省） |
| `city` | string | ❌ | 城市（如：深圳市） |
| `materialModelSpec` | string | ❌ | 规格型号（如：HRB400） |
| `brand` | string | ❌ | 品牌（厂商报价时有效） |
| `startReleaseDate` | string | ❌ | 开始日期（格式：2024-01-01） |
| `endReleaseDate` | string | ❌ | 结束日期 |

---

## 五、错误处理

### 5.1 常见错误码

| 场景 | HTTP状态码 | 错误信息 |
|-----|-----------|---------|
| 服务未初始化 | 503 | "RAG服务未初始化" |
| 渠道无法识别 | 200 | metadata.success=false, error_message="无法识别查询渠道" |
| 查询结果为空 | 200 | metadata.success=false, error_message="未查询到符合条件的价格数据" |
| 会话不存在 | 404 | "会话不存在或已过期" |

### 5.2 前端错误处理建议

```javascript
function handlePriceError(result) {
  if (!result.success) {
    if (result.error_message.includes("未查询到")) {
      // 提示用户放宽条件
      showTip("未查询到价格数据，建议：\n1. 更换材料名称\n2. 放宽地区范围\n3. 尝试其他渠道");
    } else if (result.error_message.includes("渠道")) {
      // 提示用户选择渠道
      showChannelSelector();
    }
  }
}
```

---

## 六、性能指标

| 指标 | 典型值 | 说明 |
|-----|-------|------|
| 渠道识别 | 50-100ms | LLM调用时间 |
| 实体提取 | 50-100ms | LLM调用时间 |
| 数据查询 | 500ms-2s | 取决于数据量和网络 |
| 价格分析 | 100-300ms | 统计分析 + K-means |
| LLM回答生成 | 1-3s | 流式生成，首token 200-500ms |
| **总耗时** | **2-5s** | 完整查询流程 |

---

## 七、更新日志

### v1.0 (2026-04-09)
- 整理现有价格查询逻辑
- 完善API接口文档
- 添加前端集成示例
- 新增MCP/Agent接口说明

---

**相关文件**:
- `app/api/endpoints.py` - API端点实现
- `app/services/rag_service.py` - 价格查询服务
- `app/utils/dialogue_manager.py` - 对话管理
- `app/utils/channel_inferencer.py` - 渠道推断
- `app/mcp/` - MCP工具模块（新增）
