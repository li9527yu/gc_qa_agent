# RAGService 模块文档

## 1. 概述

`rag_service.py` 是 Easy-RAG 项目的核心服务模块，实现了知识问答(RAG)和价格推荐两大核心功能。该模块封装了检索、重排序、LLM推理、价格查询等完整业务流程。

**文件路径**: `/home/tpc/suda/rag/easy-rag/app/services/rag_service.py`

---

## 2. 类定义

### 2.1 ChannelType (Enum)

价格渠道类型枚举，定义三种价格数据源：

| 枚举值 | 说明 |
|--------|------|
| `ZC_PRICE` | 智诚信息价 |
| `MANUFACTURER_PRICE` | 厂商报价 |
| `INFORMATION_PRICE` | 信息价 |
| `UNKNOWN` | 未知渠道 |

---

### 2.2 RAGService (主类)

核心服务类，封装了完整的 RAG 和价格推荐业务逻辑。

#### 2.2.1 属性说明

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `logger` | Logger | 日志记录器 |
| `retriever` | Retriever | 文档检索器 (BM25 + 向量检索) |
| `reranker` | Reranker | 结果重排序器 |
| `llm` | LLMPredictor | LLM 推理客户端 |
| `reader` | Reader | 语料库读取器 |
| `corpus` | List | 文档语料库 |
| `semaphore` | Semaphore | 并发控制信号量 (限制10个并发) |
| `_last_price_metadata` | Dict | 最后一次价格查询的元数据 |

---

## 3. 核心方法详解

### 3.1 初始化方法

#### `__init__()`
构造函数，初始化各组件为 None，等待异步初始化。

#### `async initialize()`
异步初始化入口，完成以下步骤：

```
1. 加载语料库 (Reader)
2. 计算语料库哈希，检测是否需要重建向量库
3. 初始化检索器 (Retriever) - 混合检索：BM25 + 向量检索
4. 初始化重排序器 (Reranker) - BGE-reranker-large
5. 初始化 LLM 客户端 (LLMPredictor)
6. 加载价格推荐模板
```

**向量库重建逻辑**：
- 计算当前语料库哈希值
- 对比缓存的哈希值
- 如果不一致，自动重建 Milvus 向量库

---

### 3.2 意图识别方法

#### `async classify_intent(question: str) -> str`

识别用户查询意图。

**返回结果**：
- `price_recommendation` - 价格推荐
- `knowledge_qa` - 知识问答
- `other` - 其他
- `dangerous_sql` - 危险查询

**实现逻辑**：
```python
使用 LLM 预测意图模板
├── 成功 → 返回预测结果
└── 失败 → 默认返回 "knowledge_qa"
```

#### `async identify_channel(question: str) -> ChannelType`

识别价格查询的渠道类型。

**返回结果**：`ChannelType` 枚举值

---

### 3.3 实体抽取方法

#### `async extract_entities(question: str) -> dict`

从用户问题中抽取结构化查询参数。

**抽取字段**：
- `materialName` - 材料名称
- `materialModelSpec` - 规格型号
- `province` / `city` - 省份/城市
- `startReleaseDate` / `endReleaseDate` - 发布日期范围
- `minPrice` / `maxPrice` - 价格范围
- `brand` - 品牌（厂商报价）
- `grade` - 等级（智诚信息价）

**输出格式**：JSON 对象

---

### 3.4 价格推荐核心方法

#### `async process_price_recommendation(channel_result: str, parsed_entities: dict) -> Dict`

价格推荐的完整处理流程（非流式）。

**处理流程**：

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段1: 快速检查数据总数                                       │
│   └── 调用 _quick_check_data_count() 只获取总数，不获取详细数据  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 数据量检查                                                   │
│   └── check_data_volume_and_guide()                          │
│       ├── 超过阈值 (1000条) → 返回引导消息，提示细化条件        │
│       └── 数据量可控 → 继续下一阶段                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段2: 获取详细数据                                          │
│   └── 调用 _query_price_data() 获取完整数据                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 数据过滤                                                     │
│   └── filter_items() 使用模糊匹配过滤结果                     │
│       ├── alpha=0.4 (匹配阈值)                              │
│       ├── T_keep=0.75 (保留阈值)                            │
│       └── T_drop=0.50 (丢弃阈值)                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 数据分析                                                     │
│   └── analyze_by_unit()                                      │
│       ├── 按单位分组统计                                     │
│       ├── 计算价格统计信息 (均值、中位数、标准差等)            │
│       └── 识别最常见单位                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 生成响应                                                     │
│   ├── parse_material_response() - 生成 Markdown 表格         │
│   └── 组装完整响应数据                                       │
└─────────────────────────────────────────────────────────────┘
```

**返回数据结构**：
```python
{
    "answer": str,                    # 生成的回答
    "parsed_entities": str,           # 解析的实体(JSON)
    "detail_answer": str,             # 详细分析结果(JSON)
    "contexts": list,                 # 上下文(价格推荐为空)
    "metadata": {
        "channel": str,               # 渠道类型
        "entities": dict,             # 实体信息
        "price_analysis": dict,       # 价格分析结果
        "md_table": str,              # Markdown表格
        "performance": {              # 性能统计
            "data_processing": {...},
            "token_analysis": {...}
        }
    },
    "success": bool,                  # 是否成功
    "intent": str,                    # 意图类型
    "total_count": int,               # 数据总数
    "price_data": list,               # 价格数据列表
    "most_common_unit": str           # 最常见单位
}
```

---

#### `async stream_price_recommendation(channel_result: str, parsed_entities: dict) -> Dict`

价格推荐的流式处理入口。

**说明**：
- 目前实现为先获取完整结果，再返回
- 将结果存储在 `_last_price_metadata` 中供后续使用
- 返回的数据用于流式响应的初始数据包

---

### 3.5 价格查询辅助方法

#### `_quick_check_data_count(channel: str, entities: dict) -> int`

快速检查数据总数，不返回具体数据。

**用途**：在获取详细数据前，先判断数据量是否过大。

#### `_query_price_data(channel: str, entities: dict) -> dict`

根据渠道查询详细价格数据。

**支持的渠道**：
- `INFORMATION_PRICE` → `_get_infor_material()`
- `MANUFACTURER_PRICE` → `_get_factory_material()`
- `ZC_PRICE` → `_get_zhicheng_info_material()`

#### `_get_infor_material(data: dict) -> dict`

信息价 API 查询接口。

**请求参数**：
```python
{
    "accountId": "testoxidmwedxdkseucdnvksfnzmdfnzd",
    "matchMethod": 1,
    "materialName": str,
    "materialModelSpec": str,
    "province": str,
    "city": str,
    "startReleaseDate": str,
    "endReleaseDate": str,
    "minPrice": float,
    "maxPrice": float,
    "returnNumber": 10000,      # 最大返回数量
    "returnTotalCount": 1        # 返回总数
}
```

**认证方式**：使用 `generate_signature()` 生成签名

#### `_get_factory_material(data: dict) -> dict`

厂商报价 API 查询接口。

**特有字段**：
- `brand` - 品牌
- `supplyName` - 供应商名称

#### `_get_zhicheng_info_material(data: dict) -> dict`

智诚信息价 API 查询接口。

**特有字段**：
- `grade` - 材料等级

---

### 3.6 RAG 知识问答方法

#### `async process_single_query(question: str, num_docs: int = 10) -> Dict`

RAG 知识问答处理入口。

**处理流程**：

```
┌────────────────────────────────────────┐
│ 1. 文档检索 (Retriever)                │
│    └── BM25 + 向量检索混合检索          │
└────────────────────────────────────────┘
                   ↓
┌────────────────────────────────────────┐
│ 2. 结果重排序 (Reranker)               │
│    └── BGE-reranker-large 重排序        │
│    └── 返回 top N 结果                  │
└────────────────────────────────────────┘
                   ↓
┌────────────────────────────────────────┐
│ 3. 返回检索结果                         │
│    └── contexts: 检索到的文档片段       │
│    └── metadata: 性能统计信息           │
└────────────────────────────────────────┘
```

**注意**：该方法只返回检索和重排序结果，不调用 LLM 生成答案。LLM 生成在 `endpoints.py` 的流式接口中完成。

---

### 3.7 直接价格查询方法

#### `async process_direct_price_query(datatype: str, material_item, question: str = None) -> Dict`

直接价格查询处理入口，用于前端直接传递材料信息的场景。

**参数说明**：
- `datatype`: 数据类型 (`informaterial` / `factory` / `zhicheng`)
- `material_item`: 材料信息对象
- `question`: 可选的用户问题，用于提取额外查询条件

**处理逻辑**：

```
1. 根据 datatype 映射到 ChannelType
2. 从 material_item 提取基础查询参数
3. 如果提供了 question:
   └── 使用 LLM 提取额外查询条件
       └── 合并到查询参数中
4. 调用 process_price_recommendation() 处理
5. 如果成功，使用 LLM 生成自然语言回答
6. 返回完整结果
```

---

### 3.8 响应解析方法

#### `parse_material_response(channel: ChannelType, response_json: list, top_n: Optional[int] = None) -> str`

将价格数据解析为 Markdown 表格。

**不同渠道的表头**：

| 渠道 | 表头字段 |
|------|----------|
| INFORMATION_PRICE | 编号、材料名称、规格型号、价格、单位、省份、城市、发布时间 |
| MANUFACTURER_PRICE | 编号、材料名称、规格型号、**品牌**、价格、单位、省份、城市、发布时间 |
| ZC_PRICE | 编号、材料名称、规格型号、价格、单位、省份、城市、发布时间 |

---

### 3.9 工具方法

#### `_parse_llm_output(llm_output: str) -> dict`

解析 LLM 返回的 JSON 字符串，支持去除 Markdown 代码块标记。

#### `_manual_parse_response(response_str: str) -> dict`

手动解析简单的键值对格式，用于 JSON 解析失败时的降级处理。

#### `get_last_price_metadata() -> dict`

获取最后一次价格推荐的元数据。

---

## 4. 数据流图

### 4.1 知识问答流程

```
用户问题
    │
    ▼
┌─────────────────┐
│  意图识别        │
│ classify_intent │
└─────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│          知识问答分支                  │
│   process_single_query()              │
│                                       │
│   ┌─────────────┐    ┌─────────────┐ │
│   │   检索器     │───▶│   重排序器   │ │
│   │  Retriever │    │  Reranker   │ │
│   │  (BM25+向量)│    │ (BGE-large) │ │
│   └─────────────┘    └─────────────┘ │
│                              │        │
│                              ▼        │
│   返回检索结果(contexts)               │
└───────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────┐
│  LLM 生成答案 (在 endpoints.py 中)     │
│  - 构建 Prompt (模板 + 上下文 + 问题)  │
│  - 流式调用 LLM                       │
│  - 返回生成结果                        │
└───────────────────────────────────────┘
```

### 4.2 价格推荐流程

```
用户问题
    │
    ▼
┌─────────────────┐
│   意图识别       │
│ classify_intent │
└─────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│              价格推荐分支                      │
│                                               │
│  ┌─────────────────┐  ┌──────────────────┐   │
│  │   渠道识别       │  │    实体抽取       │   │
│  │ identify_channel│  │ extract_entities │   │
│  └─────────────────┘  └──────────────────┘   │
│           │                    │              │
│           └────────┬───────────┘              │
│                    ▼                          │
│  ┌──────────────────────────────────────┐    │
│  │  process_price_recommendation()      │    │
│  │                                      │    │
│  │  1. _quick_check_data_count()        │    │
│  │     └── 快速检查总数                  │    │
│  │                                      │    │
│  │  2. check_data_volume_and_guide()    │    │
│  │     └── 数据量评估                   │    │
│  │     └── 超过阈值? 返回引导消息        │    │
│  │                                      │    │
│  │  3. _query_price_data()              │    │
│  │     └── 获取详细数据                  │    │
│  │                                      │    │
│  │  4. filter_items()                   │    │
│  │     └── 模糊匹配过滤                  │    │
│  │                                      │    │
│  │  5. analyze_by_unit()                │    │
│  │     └── 按单位分组分析                │    │
│  │                                      │    │
│  │  6. parse_material_response()        │    │
│  │     └── 生成 Markdown 表格           │    │
│  └──────────────────────────────────────┘    │
└───────────────────────────────────────────────┘
    │
    ▼
返回结果 + 流式生成回答
```

---

## 5. 配置依赖

### 5.1 导入的配置项

| 配置项 | 来源 | 用途 |
|--------|------|------|
| `RELATED_DATA_PATH` | config.py | 语料库路径 |
| `DATA_PATH` | config.py | 数据路径 |
| `USE_MOCK_API` | config.py | 是否使用 Mock API |
| `MOCK_API_BASE_URL` | config.py | Mock API 地址 |
| `REAL_API_BASE_URL` | config.py | 真实 API 地址 |
| `API_SECRET_KEY` | config.py | API 签名密钥 |

### 5.2 模型路径

| 模型 | 路径 |
|------|------|
| Embedding 模型 | `/home/tpc/suda/rag/model/bge-large-zh-v1.5` |
| Reranker 模型 | `/home/tpc/suda/rag/model/bge-reranker-large` |

---

## 6. 性能统计

模块内置了详细的性能统计和日志记录：

### 6.1 价格查询统计

```
📊 价格查询数据处理统计
🔍 查询条件: {...}
📈 数据统计:
   - 原始数据量: X 条
   - 过滤后数量: Y 条
   - 返回数量: Z 条
🎫 Token 分析:
   - 原始数据 Token: X tokens
   - 过滤后数据 Token: Y tokens
   - 传给 LLM 的 Token: Z tokens
   - Token 优化率: X%
⏱️  数据处理时间:
   - 数据检查: X.XXXs
   - 数据分析: X.XXXs
   - 总耗时: X.XXXs
```

### 6.2 数据量过大警告

```
⚠️  数据量过大警告
🔍 查询条件: {...}
📈 数据量: X 条 (超过阈值 1000 条)
🎫 预计 Token: X tokens
💡 建议补充筛选条件: [...]
⏱️  响应时间: X.XXXs
```

---

## 7. 错误处理

模块实现了完善的错误处理和降级策略：

| 场景 | 处理方式 |
|------|----------|
| 意图识别失败 | 默认返回 `knowledge_qa` |
| 渠道识别失败 | 返回 `ChannelType.UNKNOWN` |
| 实体抽取失败 | 返回空字典 `{}` |
| 价格查询失败 | 返回 `success=False` 和错误信息 |
| 数据量为0 | 返回友好提示信息 |
| JSON 解析失败 | 尝试手动解析降级 |

---

## 8. 使用示例

### 8.1 初始化服务

```python
from app.services.rag_service import RAGService

service = RAGService()
await service.initialize()
```

### 8.2 意图识别

```python
intent = await service.classify_intent("查询钢筋的价格")
# 返回: "price_recommendation"
```

### 8.3 价格推荐

```python
# 渠道识别
channel = await service.identify_channel("查询钢筋的信息价")

# 实体抽取
entities = await service.extract_entities("查询上海市2024年钢筋的价格")

# 处理价格推荐
result = await service.process_price_recommendation(
    channel_result=channel.value,
    parsed_entities=entities
)
```

### 8.4 知识问答

```python
result = await service.process_single_query(
    question="什么是BIM技术？",
    num_docs=5
)
# 返回检索结果，供后续 LLM 生成答案
```

---

## 9. 相关文件

| 文件 | 说明 |
|------|------|
| `app/llm_deepseek.py` | LLM 客户端封装 |
| `app/retriever.py` | 检索器实现 |
| `app/reranker.py` | 重排序器实现 |
| `app/read_corpus.py` | 语料库读取 |
| `app/utils/prompt_template.py` | 提示词模板 |
| `app/utils/price_tools.py` | 价格分析工具 |
| `app/utils/fillter_tools.py` | 数据过滤工具 |
| `app/utils/data_filter_helper.py` | 数据量检查工具 |
| `app/utils/generate_signature.py` | API 签名生成 |

---

*文档生成时间: 2026-03-23*
