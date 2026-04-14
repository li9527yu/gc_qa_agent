# Agent 会话持久化方案

> 文档版本: v1.0  
> 更新时间: 2026-04-13  
> 适用项目: easy-rag（ReAct Agent 价格查询模块）

---

## 一、背景与现状

当前 `MaterialPriceAgent`（`app/mcp/agent.py`）的会话上下文 `AgentContext` 存储在进程内存的 `self.contexts: Dict[str, AgentContext]` 中：

```python
class MaterialPriceAgent:
    def __init__(self, ...):
        self.contexts: Dict[str, AgentContext] = {}
```

### 存在的问题

| 问题 | 影响 |
|------|------|
| **服务重启丢失会话** | FastAPI 重启 / 热更新后，所有用户的对话上下文清空 |
| **无水平扩展能力** | 多实例部署时，A 实例创建的会话 B 实例无法识别 |
| **内存持续增长** | 会话无持久化淘汰机制，长期运行可能 OOM |
| **无法离线恢复** | 用户刷新页面后回到同一对话，只能靠前端缓存 trick |

---

## 二、方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **A. 本地 JSON 文件** | 实现简单、无外部依赖 | IO 慢、并发差、无法多机共享 | 单机 Demo |
| **B. Redis / KeyDB** | 性能高、支持 TTL、多机共享 | 需维护 Redis 服务 | **推荐方案** |
| **C. 关系型数据库** | 结构化强、可审计 | 写入慢、序列化成本高 | 需要长期审计 |
| **D. 前端 LocalStorage** | 零后端改造 | 容量小、不安全、无法跨端 | 临时兜底 |

**推荐方案：B（Redis + JSON 序列化）**

原因：
1. Agent 上下文是**半结构化**的（嵌套 dict / list），JSON 存储最自然
2. 对话有**明确的 TTL**（如 30 分钟无交互则过期），Redis 原生支持
3. 价格查询是**读多写少**的场景，Redis 性能完全满足
4. 团队已有 Redis 使用经验的可能性较高

---

## 三、推荐实现架构

```
┌─────────────┐      session_id        ┌─────────────┐
│   前端/Web   │ ──────────────────────▶ │  FastAPI    │
│   (stream)  │                         │  /agent/chat │
└─────────────┘                         └──────┬──────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ MaterialPriceAgent│
                                      │  - 优先查内存     │
                                      │  - miss 则查 Redis│
                                      └────────┬────────┘
                                               │
                       ┌───────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     Redis       │
              │  Key: agent:{sid}│
              │  Value: JSON     │
              │  TTL: 1800s      │
              └─────────────────┘
```

---

## 四、核心设计

### 4.1 存储结构

```json
{
  "session_id": "abc123",
  "turn_count": 3,
  "channel": "information_price",
  "entities": {
    "materialName": "钢筋",
    "province": "广东省",
    "city": "深圳市"
  },
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "查一下钢筋的价格"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "extract_entities", "content": "..."},
    {"role": "assistant", "content": "您的查询缺少省份..."},
    {"role": "user", "content": "广东省深圳市的"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "query_price_data", "content": "..."},
    {"role": "assistant", "content": "在广东省深圳市，钢筋..."}
  ],
  "thoughts": [
    {"step": 1, "content": "", "tool_calls": [...]},
    {"step": 2, "content": "", "tool_calls": [...]}
  ],
  "query_result": {...},
  "updated_at": 1744523948.0
}
```

### 4.2 Key 命名规范

```
agent:session:{session_id}
```

示例：`agent:session:abc123`

### 4.3 TTL 策略

- **默认 TTL**：`1800` 秒（30 分钟）
- **每次交互后刷新 TTL**：用户发消息或 Agent 回复后，更新 `updated_at` 并重设 TTL
- **最大消息长度保护**：单条 Redis Value 不超过 512KB（超过则截断早期 messages）

---

## 五、代码实现示例

### 5.1 新增 `app/utils/agent_session_store.py`

```python
"""
Agent 会话存储层
支持内存 + Redis 双级缓存
"""
import json
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("easy_rag_api")

# ========== 配置（后续可移到 config.py）==========
AGENT_SESSION_TTL = 1800  # 30 分钟
AGENT_SESSION_MAX_SIZE = 512 * 1024  # 512KB


class AgentSessionStore:
    """Agent 会话存储"""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("easy_rag_api")
    
    def _make_key(self, session_id: str) -> str:
        return f"agent:session:{session_id}"
    
    def _serialize(self, context: Dict[str, Any]) -> str:
        # 截断过长的 messages 以保护 Redis
        payload = context.copy()
        messages = payload.get("messages", [])
        
        # 保留 system prompt，截断早期的 user/assistant（保留最近 10 轮）
        while len(json.dumps(payload)) > AGENT_SESSION_MAX_SIZE and len(messages) > 3:
            # 删除 system 之后的第一条消息
            messages.pop(1)
            payload["messages"] = messages
        
        return json.dumps(payload, ensure_ascii=False)
    
    def _deserialize(self, raw: str) -> Dict[str, Any]:
        return json.loads(raw)
    
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话，先查内存再查 Redis"""
        # 1. 查本地内存
        if session_id in self.local_cache:
            return self.local_cache[session_id]
        
        # 2. 查 Redis
        if self.redis:
            try:
                key = self._make_key(session_id)
                raw = self.redis.get(key)
                if raw:
                    context = self._deserialize(raw)
                    # 回填本地缓存
                    self.local_cache[session_id] = context
                    return context
            except Exception as e:
                self.logger.warning(f"Redis 读取会话失败: {e}")
        
        return None
    
    def save(self, session_id: str, context: Dict[str, Any]) -> None:
        """保存会话到内存和 Redis"""
        context["updated_at"] = time.time()
        self.local_cache[session_id] = context
        
        if self.redis:
            try:
                key = self._make_key(session_id)
                raw = self._serialize(context)
                self.redis.setex(key, AGENT_SESSION_TTL, raw)
            except Exception as e:
                self.logger.warning(f"Redis 保存会话失败: {e}")
    
    def delete(self, session_id: str) -> None:
        """删除会话"""
        self.local_cache.pop(session_id, None)
        if self.redis:
            try:
                self.redis.delete(self._make_key(session_id))
            except Exception as e:
                self.logger.warning(f"Redis 删除会话失败: {e}")
    
    def clear_local_cache(self):
        """清理本地内存缓存（不影响 Redis）"""
        self.local_cache.clear()
```

### 5.2 修改 `app/mcp/agent.py`

```python
from app.utils.agent_session_store import AgentSessionStore

class MaterialPriceAgent:
    def __init__(self, mcp_server: MCPServer, llm_client, redis_client=None):
        self.mcp_server = mcp_server
        self.llm = llm_client
        self.logger = logging.getLogger("easy_rag_api")
        self.max_steps = 5
        
        # 替换纯内存存储为持久化存储层
        self.store = AgentSessionStore(redis_client=redis_client)
    
    def _get_or_create_context(self, session_id: Optional[str]) -> AgentContext:
        # 尝试从存储层恢复
        if session_id:
            raw = self.store.get(session_id)
            if raw:
                context = AgentContext(session_id=raw["session_id"])
                context.turn_count = raw.get("turn_count", 0)
                context.channel = raw.get("channel", "information_price")
                context.entities = raw.get("entities", {})
                context.messages = raw.get("messages", [])
                context.thoughts = raw.get("thoughts", [])
                context.query_result = raw.get("query_result")
                return context
        
        # 创建新会话
        import uuid
        new_id = session_id or str(uuid.uuid4())[:12]
        context = AgentContext(session_id=new_id)
        context.add_message("system", SYSTEM_PROMPT)
        return context
    
    def _persist_context(self, context: AgentContext) -> None:
        """将会话持久化"""
        payload = {
            "session_id": context.session_id,
            "turn_count": context.turn_count,
            "channel": context.channel,
            "entities": context.entities,
            "messages": context.messages,
            "thoughts": context.thoughts,
            "query_result": context.query_result,
        }
        self.store.save(context.session_id, payload)
```

在 `process_message_stream` 的 **每次事件 yield 之后** 或 **最终返回前** 调用 `_persist_context(context)` 即可。

### 5.3 修改 `app/api/mcp_endpoints.py` 初始化

```python
def init_mcp_endpoints(rag_service):
    global mcp_server, material_price_agent
    
    from app.mcp.server import MCPServer
    from app.mcp.agent import MaterialPriceAgent
    import redis
    
    mcp_server = MCPServer(rag_service)
    
    # 尝试连接 Redis（可选，失败则退化为内存存储）
    redis_client = None
    try:
        redis_client = redis.Redis(
            host="127.0.0.1",
            port=6379,
            db=0,
            decode_responses=True,
            socket_connect_timeout=2
        )
        redis_client.ping()
        logger.info("Redis 连接成功，Agent 会话将持久化")
    except Exception as e:
        logger.warning(f"Redis 未就绪，Agent 会话仅保存在内存中: {e}")
    
    material_price_agent = MaterialPriceAgent(mcp_server, rag_service.llm, redis_client)
    logger.info("MCP端点初始化完成")
```

---

## 六、迁移步骤

### Step 1：安装依赖

```bash
pip install redis>=5.0.0
```

### Step 2：启动 Redis（如未启动）

```bash
# Docker 方式
docker run -d --name redis-agent -p 6379:6379 redis:7-alpine

# 或本地已安装
redis-server --port 6379
```

### Step 3：新增存储层代码

- 创建 `app/utils/agent_session_store.py`
- 按上文实现 `AgentSessionStore`

### Step 4：改造 Agent

- `app/mcp/agent.py`：
  - 删除 `self.contexts: Dict[str, AgentContext]`
  - 引入 `AgentSessionStore`
  - 修改 `_get_or_create_context` 支持反序列化
  - 在流式/非流式处理结束时调用 `_persist_context`

### Step 5：修改初始化注入

- `app/api/mcp_endpoints.py`：`init_mcp_endpoints` 中传入 Redis 客户端

### Step 6：回归测试

```bash
# 1. 启动服务后创建会话
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "查一下钢筋的价格"}'

# 记录返回的 session_id，例如：abc123

# 2. 重启 FastAPI 服务
# ...

# 3. 用原 session_id 继续对话
curl -X POST http://localhost:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "message": "广东省深圳市的"}'

# 验证：Agent 应记得之前查询的是"钢筋"
```

---

## 七、边界情况处理

| 场景 | 处理策略 |
|------|---------|
| **Redis 不可用** | 优雅降级为纯内存存储，打印 warning |
| **会话过期** | 返回 404，提示用户"会话已过期，请重新开始" |
| **消息过长** | 自动截断早期 messages，保留 system + 最近 10 轮 |
| **JSON 序列化失败** | 跳过持久化，仅保留内存，不阻塞主流程 |
| **并发写同一 session** | 单进程 asyncio 天然串行；多实例场景下 Redis 最终一致性可接受 |

---

## 八、后续可扩展项

1. **会话审计表**
   - 关键对话（如完成价格查询）异步写入 MySQL/ClickHouse，用于运营分析

2. **多级缓存**
   - 本地 LRU Cache（如 `cachetools.TTLCache`）+ Redis，减少网络 RTT

3. **会话导出**
   - 提供 `/api/v1/agent/session/{sid}/export` 接口，供用户下载对话记录

---

## 九、总结

> **核心思路**：引入 `AgentSessionStore` 作为存储抽象层，底层优先使用 **Redis + JSON**，失败时优雅降级到内存。

这样可以在不大幅改动 Agent 核心逻辑的前提下，解决服务重启丢会话、无法水平扩展等关键问题。
