# Easy-RAG API curl 测试指南

> 文档版本: v1.0  
> 更新时间: 2026-04-09  
> 测试前请确保: 后端服务已启动 (python -m app.api_server)

---

## 一、快速测试

### 1.1 服务健康检查

```bash
# 检查服务是否运行
curl -s http://localhost:8001/ | jq

# 预期响应
{
  "message": "Easy-RAG API服务运行中",
  "version": "1.0.0",
  "status": "healthy"
}
```

### 1.2 服务状态检查

```bash
# 检查知识库和LLM状态
curl -s http://localhost:8001/api/v1/service/status | jq

# 预期响应
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

---

## 二、Agent对话接口测试

### 2.1 首次查询 - 简单材料名

```bash
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "查一下钢筋的价格"
  }' | jq
```

**预期响应**:
```json
{
  "success": true,
  "session_id": "abc123",
  "state": "collecting_entities",
  "turn_count": 1,
  "message": "收到，您想查询「钢筋」的价格信息。\n查询渠道：信息价\n\n✓ 找到精确匹配的材料：钢筋\n\n📋 为了给您提供准确的价格信息，还需要：\n  • 省份\n  • 城市",
  "data": {
    "entities": {
      "materialName": "钢筋"
    },
    "channel": "information_price",
    "candidates": [...],
    "selected_candidate": "钢筋"
  },
  "suggested_actions": ["广东省", "江苏省", "浙江省", "北京市", "上海市"]
}
```

### 2.2 补充省份信息

```bash
# 使用上一步返回的 session_id
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "广东省",
    "session_id": "abc123"
  }' | jq
```

### 2.3 补充城市信息

```bash
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "深圳市",
    "session_id": "abc123"
  }' | jq
```

**预期响应** (state变为 ready_to_query):
```json
{
  "success": true,
  "session_id": "abc123",
  "state": "ready_to_query",
  "turn_count": 3,
  "message": "查询条件已收集完毕！\n\n📋 查询信息：\n  • 材料名称：钢筋\n  • 地区：广东省深圳市\n\n是否确认查询？",
  "suggested_actions": ["✅ 确认查询", "🔄 修改条件"]
}
```

### 2.4 确认查询

```bash
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "确认查询",
    "session_id": "abc123"
  }' | jq
```

**预期响应** (state变为 completed，包含价格数据):
```json
{
  "success": true,
  "session_id": "abc123",
  "state": "completed",
  "turn_count": 4,
  "message": "✅ 「钢筋」价格查询完成！\n\n📊 共查询到 156 条价格数据\n\n💰 推荐价格（单位：kg）：24.50 元\n\n📈 价格区间：23.00 - 26.00 元\n📊 平均价格：24.35 元\n\n您还想了解什么？",
  "data": {
    "entities": {
      "materialName": "钢筋",
      "province": "广东省",
      "city": "深圳市"
    },
    "query_result": {
      "success": true,
      "total_count": 156,
      "price_data": [...],
      "metadata": {...}
    },
    "analysis": {...}
  },
  "suggested_actions": ["查询新材料", "查看详细数据", "对比价格"]
}
```

### 2.5 一次性完整查询

```bash
# 如果一次性提供完整信息，Agent会直接查询
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "查一下广东省深圳市钢筋的价格"
  }' | jq
```

### 2.6 查询新材料（切换话题）

```bash
# 使用已有session，但查询不同材料
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "再查一下水泥的价格",
    "session_id": "abc123"
  }' | jq
```

### 2.7 查询不同渠道

```bash
# 查询厂商报价
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "从厂商报价查钢筋的价格"
  }' | jq

# 查询智诚信息价
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "查一下智诚信息价的钢筋"
  }' | jq
```

---

## 三、MCP工具接口测试

### 3.1 获取工具列表

```bash
curl -s http://localhost:8001/api/v1/mcp/tools | jq
```

### 3.2 快速搜索材料

```bash
curl -s -X POST http://localhost:8001/api/v1/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "quick_search_materials",
    "parameters": {
      "keyword": "钢筋",
      "limit": 5
    }
  }' | jq
```

### 3.3 提取实体

```bash
curl -s -X POST http://localhost:8001/api/v1/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "extract_entities",
    "parameters": {
      "question": "查一下广东省深圳市HRB400钢筋的价格"
    }
  }' | jq
```

**预期响应**:
```json
{
  "success": true,
  "result": {
    "status": "success",
    "data": {
      "materialName": "钢筋",
      "province": "广东省",
      "city": "深圳市",
      "materialModelSpec": "HRB400"
    },
    "message": "实体提取完成",
    "metadata": {
      "extracted_fields": ["materialName", "province", "city", "materialModelSpec"],
      "missing_fields": []
    },
    "suggested_next_steps": ["实体信息完整，可以进行价格查询"]
  }
}
```

### 3.4 识别渠道

```bash
curl -s -X POST http://localhost:8001/api/v1/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "identify_channel",
    "parameters": {
      "question": "从厂商报价查钢筋价格"
    }
  }' | jq
```

### 3.5 查询价格数据

```bash
curl -s -X POST http://localhost:8001/api/v1/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "query_price_data",
    "parameters": {
      "channel": "information_price",
      "material_name": "钢筋",
      "province": "广东省",
      "city": "深圳市"
    }
  }' | jq '.result.data.total_count'
```

### 3.6 分析价格数据

```bash
# 先查询获取price_data，然后分析
curl -s -X POST http://localhost:8001/api/v1/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "analyze_prices",
    "parameters": {
      "price_data": [
        {"price": 23.5, "unit": "kg"},
        {"price": 24.0, "unit": "kg"},
        {"price": 24.5, "unit": "kg"},
        {"price": 25.0, "unit": "kg"}
      ],
      "unit": "auto"
    }
  }' | jq
```

### 3.7 获取候选材料

```bash
curl -s -X POST http://localhost:8001/api/v1/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "get_material_candidates",
    "parameters": {
      "keyword": "混凝土",
      "limit": 5
    }
  }' | jq '.result.data'
```

---

## 四、标准价格查询接口测试

### 4.1 流式价格查询

```bash
# 流式响应，需要逐行读取
curl -s -X POST http://localhost:8001/api/v1/query/price \
  -H "Content-Type: application/json" \
  -d '{
    "question": "从信息价查广东省深圳市钢筋的价格"
  }'
```

**响应格式** (流式):
```
识别到渠道类型: {"channel": "information_price", ...}

[END]

解析到实体结果: {"materialName": "钢筋", ...}

[END]

根据查询结果，广东省深圳市钢筋的价格分析如下：
...

[END]

{"text": "完整回答", "intent": "price_recommendation", "metadata": {...}}
```

### 4.2 带会话的价格查询

```bash
curl -s -X POST http://localhost:8001/api/v1/query/price \
  -H "Content-Type: application/json" \
  -d '{
    "question": "查一下水泥的价格",
    "session_id": "abc123",
    "force_new": true
  }'
```

---

## 五、知识问答接口测试

### 5.1 流式知识问答

```bash
curl -s -X POST http://localhost:8001/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是工程造价",
    "num_docs": 5
  }'
```

### 5.2 带会话的知识问答

```bash
curl -s -X POST http://localhost:8001/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "工程造价包括哪些费用",
    "session_id": "abc123",
    "num_docs": 5
  }'
```

---

## 六、对话式查询接口测试（已废弃）

> ⚠️ 以下接口已停止服务，请使用 `POST /api/v1/agent/chat` 替代。

<!--
### 6.1 创建对话

```bash
curl -s -X POST http://localhost:8001/api/v1/query/dialogue \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": null,
    "user_input": "查一下钢筋的价格"
  }' | jq
```

### 6.2 继续对话

```bash
curl -s -X POST http://localhost:8001/api/v1/query/dialogue \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "user_input": "广东省深圳市的"
  }' | jq
```

### 6.3 执行查询

```bash
curl -s -X POST http://localhost:8001/api/v1/query/dialogue/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123"
  }' | jq
```

### 6.4 获取会话状态

```bash
curl -s http://localhost:8001/api/v1/query/dialogue/abc123 | jq
```

### 6.5 清除会话

```bash
curl -s -X DELETE http://localhost:8001/api/v1/query/dialogue/abc123 | jq
```
-->

---

## 七、文件管理接口测试

### 7.1 获取文件列表

```bash
curl -s http://localhost:8001/api/v1/files | jq
```

### 7.2 上传文件

```bash
# 上传单个文件
curl -s -X POST http://localhost:8001/api/v1/upload \
  -F "files=@/path/to/your/file.pdf"

# 上传多个文件
curl -s -X POST http://localhost:8001/api/v1/upload \
  -F "files=@/path/to/file1.pdf" \
  -F "files=@/path/to/file2.txt"
```

### 7.3 删除文件

```bash
curl -s -X POST http://localhost:8001/api/v1/delete/batch \
  -H "Content-Type: application/json" \
  -d '{
    "filenames": ["file1.md", "file2.txt"]
  }' | jq
```

### 7.4 查询任务状态

```bash
# 使用上传或删除返回的task_id
curl -s "http://localhost:8001/api/v1/task_status?task_id=xxx" | jq
```

---

## 八、直接价格查询接口测试

### 8.1 结构化数据查询

```bash
curl -s -X POST http://localhost:8001/api/v1/query/price/direct \
  -H "Content-Type: application/json" \
  -d '{
    "datatype": "informaterial",
    "question": "查找钢筋的价格数据",
    "list": [{
      "materialName": "钢筋",
      "province": "广东省",
      "city": "深圳市",
      "materialModelSpec": "HRB400",
      "unit": "kg"
    }]
  }'
```

---

## 九、实用测试脚本

### 9.1 完整对话测试脚本

```bash
#!/bin/bash

BASE_URL="http://localhost:8001"

echo "=== 1. 首次查询 ==="
RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "查一下钢筋的价格"}')

echo $RESPONSE | jq
SESSION_ID=$(echo $RESPONSE | jq -r '.session_id')
STATE=$(echo $RESPONSE | jq -r '.state')

echo "Session ID: $SESSION_ID"
echo "State: $STATE"

if [ "$STATE" = "collecting_entities" ]; then
  echo ""
  echo "=== 2. 补充省份 ==="
  RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"广东省\", \"session_id\": \"$SESSION_ID\"}")
  echo $RESPONSE | jq
  STATE=$(echo $RESPONSE | jq -r '.state')
fi

if [ "$STATE" = "collecting_entities" ]; then
  echo ""
  echo "=== 3. 补充城市 ==="
  RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"深圳市\", \"session_id\": \"$SESSION_ID\"}")
  echo $RESPONSE | jq
  STATE=$(echo $RESPONSE | jq -r '.state')
fi

if [ "$STATE" = "ready_to_query" ]; then
  echo ""
  echo "=== 4. 确认查询 ==="
  RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"确认查询\", \"session_id\": \"$SESSION_ID\"}")
  echo $RESPONSE | jq
fi
```

保存为 `test_agent.sh`，然后执行:
```bash
chmod +x test_agent.sh
./test_agent.sh
```

### 9.2 性能测试脚本

```bash
#!/bin/bash

BASE_URL="http://localhost:8001"

echo "=== Agent接口性能测试 ==="

echo "测试1: 首次查询（无session）"
time curl -s -X POST "$BASE_URL/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "查一下钢筋的价格"}' > /dev/null

echo ""
echo "测试2: 标准价格查询（流式）"
time curl -s -X POST "$BASE_URL/api/v1/query/price" \
  -H "Content-Type: application/json" \
  -d '{"question": "查广东省深圳市钢筋价格"}' > /dev/null

echo ""
echo "测试3: MCP工具调用"
time curl -s -X POST "$BASE_URL/api/v1/mcp/tools/call" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "extract_entities", "parameters": {"question": "查钢筋价格"}}' > /dev/null
```

---

## 十、常见问题排查

### 10.1 服务未启动

```bash
curl: (7) Failed to connect to localhost port 8001: Connection refused

# 解决: 启动服务
python -m app.api_server
```

### 10.2 没有jq工具

```bash
# macOS
brew install jq

# Ubuntu/Debian
sudo apt-get install jq

# CentOS/RHEL
sudo yum install jq
```

### 10.3 中文乱码

```bash
# 确保终端使用UTF-8编码
export LANG=en_US.UTF-8
```

### 10.4 查看原始响应（不带jq）

```bash
# 去掉 | jq 查看原始响应
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "查一下钢筋的价格"}'
```

---

## 十一、测试检查清单

| 功能 | 命令 | 预期结果 |
|-----|------|---------|
| 服务健康 | `curl http://localhost:8001/` | status: healthy |
| Agent首次查询 | `curl -d '{"message":"查钢筋"}' .../agent/chat` | state: collecting_entities |
| Agent补充信息 | `curl -d '{"message":"广东省","session_id":"xxx"}' ...` | entities包含province |
| Agent确认查询 | `curl -d '{"message":"确认","session_id":"xxx"}' ...` | state: completed, 有price_data |
| MCP工具列表 | `curl .../mcp/tools` | 返回6个工具 |
| MCP提取实体 | `curl -d '{"tool_name":"extract_entities",...}' ...` | 返回materialName等 |
| 流式价格查询 | `curl -d '{"question":"查钢筋价格"}' .../query/price` | 流式返回，包含[END]分隔 |
| 知识问答 | `curl -d '{"question":"什么是工程造价"}' .../query/stream` | 流式返回答案 |

---

**测试完成！**
