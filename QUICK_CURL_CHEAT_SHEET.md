# curl 测试速查表

## 快速开始

```bash
# 1. 确保服务已启动
python -m app.api_server

# 2. 健康检查
curl http://localhost:8001/

# 3. 运行测试脚本
./test_api.sh all
```

---

## 最常用命令

### Agent对话（推荐）

```bash
# 首次查询
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "查一下钢筋的价格"}' | jq

# 补充信息（使用返回的session_id）
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "广东省深圳市", "session_id": "xxx"}' | jq

# 确认查询
curl -s -X POST http://localhost:8001/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "确认查询", "session_id": "xxx"}' | jq
```

### MCP工具

```bash
# 提取实体
curl -s -X POST http://localhost:8001/api/v1/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "extract_entities", "parameters": {"question": "查钢筋价格"}}' | jq

# 识别渠道
curl -s -X POST http://localhost:8001/api/v1/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "identify_channel", "parameters": {"question": "查厂商报价"}}' | jq
```

### 流式查询

```bash
# 价格查询（流式）
curl -s -X POST http://localhost:8001/api/v1/query/price \
  -H "Content-Type: application/json" \
  -d '{"question": "查广东省深圳市钢筋价格"}'

# 知识问答（流式）
curl -s -X POST http://localhost:8001/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是工程造价"}'
```

---

## 测试脚本用法

```bash
# 查看帮助
./test_api.sh help

# 快速健康检查
./test_api.sh health

# 测试Agent完整对话流程
./test_api.sh agent

# 测试MCP工具
./test_api.sh mcp

# 测试所有功能
./test_api.sh all

# 性能测试
./test_api.sh perf
```

---

## 响应状态速查

| State | 含义 | 下一步操作 |
|-------|------|-----------|
| `collecting_entities` | 收集实体中 | 补充省份/城市信息 |
| `confirming_candidate` | 确认候选材料 | 选择材料序号或名称 |
| `ready_to_query` | 准备查询 | 发送"确认查询" |
| `completed` | 已完成 | 查看价格结果或查询新材料 |
