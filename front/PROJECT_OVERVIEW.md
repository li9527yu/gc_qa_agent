# 项目逻辑说明文档

> 本文档面向程序员，帮助快速理解当前项目的架构、模块职责与核心流程。

---

## 一、项目定位

本项目是 **RAG 智能对话系统的前端客户端**，基于 [Streamlit](https://streamlit.io/) 构建，通过 HTTP REST API 与后端大模型服务（端口 `8001`）交互。  
用户可通过 Web 界面进行：
- **知识问答**（基于上传文档的 RAG 检索问答）
- **价格推荐**（查询价格数据库并生成统计图表）
- **知识库文件管理**（上传 / 删除 / 查看 TXT、PDF、MD 文件）

---

## 二、目录结构

```
.
├── server/
│   ├── __init__.py                # 空文件，标识 Python 包
│   ├── web_app.py                 # 主程序：Streamlit 页面逻辑 + 后端 API 调用
│   └── price_chart_component.py   # 价格可视化组件（ECharts 图表封装）
├── run_client.sh                  # 启动脚本：streamlit run server/web_app.py
├── environment.yml                # Conda 环境配置（含完整依赖清单）
├── pip-requirements.txt           # 纯 pip 依赖清单
└── output.log                     # 运行日志（Streamlit 服务输出）
```

---

## 三、核心模块详解

### 3.1 `server/web_app.py`

**职责**：整个前端应用的入口，包含页面布局、状态管理、后端通信、意图路由。

#### 主要类与函数

| 名称 | 类型 | 说明 |
|------|------|------|
| `CustomApiRequest` | 类 | 封装对后端 `http://localhost:8001` 的 API 请求；含 `extract_meta_base64` 静态方法（解析 Base64 META 信息，当前代码中未实际使用） |
| `confirm_delete()` | 对话框 | Streamlit `@st.dialog` 二次确认弹窗，调用 `/api/v1/delete/batch` 批量删除文件 |
| `web()` | 主函数 | 构建完整页面，初始化 `session_state`，处理侧边栏与聊天主界面逻辑 |

#### 页面布局（`web()` 函数内部）

1. **侧边栏（Sidebar）**
   - **图表设置**：设置数据量阈值（`data_threshold`），当价格样本数 ≤ 阈值时不绘制图表。
   - **文件上传**：调用 `/api/v1/upload` 上传多文件（TXT/PDF/MD），上传成功后轮询 `/api/v1/task_status` 等待知识库更新完成。
   - **文件管理**：调用 `/api/v1/files` 获取文件列表，支持勾选后点击悬浮按钮批量删除。

2. **聊天主界面（Main）**
   - 展示历史消息（`st.session_state.messages`）。
   - 用户输入问题后，前端执行 **三步流水线**：

#### 聊天核心流程（三步流水线）

```
用户输入
    │
    ▼
┌─────────────────┐
│ 1. 意图识别      │  POST /api/v1/query/intent
│    intent       │  返回值：knowledge_qa / price_recommendation / dangerous_sql / 其他
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 2. 路由到对应接口 │  根据 intent 选择 endpoint：
│                 │    • price_recommendation → /api/v1/query/price
│                 │    • knowledge_qa         → /api/v1/query/stream
│                 │    • 其他                  → /api/v1/query/question
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 3. 流式接收 &    │  SSE/Chunk 流式读取，实时渲染打字机效果
│    后处理        │  最后一条 JSON 对象作为 meta，提取 text / contexts / price_data
└─────────────────┘
```

- **若意图为 `dangerous_sql`**：直接拒绝回答。
- **若意图为 `price_recommendation`**：
  - 从 `meta.price_data` 提取价格数据。
  - 若 `total_count == 0` 或 `channel == "unknown"` 给出警告。
  - 若 `total_count > data_threshold` 且 `success == True`，自动调用 `price_distribution_chart()` 与 `price_scatter_chart()` 渲染图表。
  - 否则仅展示原始 Markdown 表格。
- **若意图为 `knowledge_qa`**：
  - 从 `meta.contexts` 提取参考文档片段，存入 `last_docs` 供底部「参考文档」区域展示。

#### Session State 关键变量

| 变量名 | 作用 |
|--------|------|
| `messages` | 聊天历史记录（user / assistant） |
| `last_docs` | 最近一次回答的参考文档或价格表 |
| `kb_updating` | 知识库是否正在更新（上传/删除后轮询） |
| `kb_update_task_id` | 后端返回的任务 ID，用于轮询更新状态 |
| `file_list` | 当前知识库文件列表 |
| `selected_files` | 用户勾选待删除的文件名列表 |
| `data_threshold` | 价格数据量阈值，控制是否绘制图表 |
| `show_price_chart` / `_price_data_cache` | 价格图表弹窗缓存（当前代码中弹窗逻辑已简化，主要用 expander 直接展开） |

---

### 3.2 `server/price_chart_component.py`

**职责**：基于 `streamlit_echarts` 封装两种价格可视化图表。

| 函数 | 图表类型 | 数据来源 |
|------|----------|----------|
| `price_distribution_chart(price_data, bins=5)` | 柱状图（价格分布频数） | `price_data` 列表中的 `price` 字段，经 `pd.cut` 分箱统计 |
| `price_scatter_chart(price_data)` | 散点图（价格-时间关系） | `price_data` 列表中的 `releaseTime`（取年月）与 `price` 字段 |

- 使用 [Apache ECharts](https://echarts.apache.org/) 作为图表引擎，通过 `st_echarts()` 嵌入 Streamlit。
- 若 `price_data` 为空，则显示 `st.warning("暂无价格数据")`。

---

## 四、后端 API 接口清单

前端依赖的后端服务运行在 `http://localhost:8001`，接口规范如下：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/query/intent` | 意图识别，请求体 `{"question": "..."}` |
| `POST` | `/api/v1/query/stream` | 知识问答流式接口（SSE） |
| `POST` | `/api/v1/query/price` | 价格推荐接口（SSE/Chunk） |
| `POST` | `/api/v1/query/question` | 通用问题接口（SSE/Chunk） |
| `POST` | `/api/v1/upload` | 文件上传，`multipart/form-data` |
| `GET`  | `/api/v1/files` | 获取知识库文件列表 |
| `DELETE`| `/api/v1/delete/batch` | 批量删除文件，请求体 `{"filenames": [...]}` |
| `GET`  | `/api/v1/task_status?task_id=xxx` | 查询知识库更新任务状态 |

> **注意**：后端代码不在当前仓库中，当前仓库仅包含前端客户端。

---

## 五、启动方式

### 5.1 环境准备

项目使用 Python 3.10，依赖管理提供了两种方式：

- **Conda**：`conda env create -f environment.yml`
- **pip**：`pip install -r pip-requirements.txt`

核心依赖包括：`streamlit`、`requests`、`pandas`、`streamlit-echarts`。

### 5.2 运行前端

```bash
bash run_client.sh
```

等价于：

```bash
streamlit run server/web_app.py --server.port 8501 --server.fileWatcherType none
```

服务启动后，访问 `http://localhost:8501`。

---

## 六、关键设计要点

1. **纯前端项目**：所有大模型推理、向量检索、文件存储逻辑均下沉到 `localhost:8001` 后端。
2. **意图驱动路由**：通过先调用 `/query/intent` 判断用户问题类型，再决定调用哪个问答接口，实现同一聊天框支持多种业务场景。
3. **流式渲染**：问答接口均采用流式返回，前端通过 `iter_lines()` 逐段读取并实时更新 `placeholder`，提供打字机式交互体验。
4. **状态隔离**：利用 Streamlit 的 `session_state` 保存聊天记录、文件列表、任务轮询状态，避免页面刷新导致数据丢失。
5. **图表阈值控制**：通过侧边栏的 `data_threshold` 让用户（或开发者）控制价格样本不足时的降级策略，避免小样本误导决策。

---

## 七、快速定位代码

| 需求 | 对应文件/函数 |
|------|---------------|
| 修改页面标题 / 欢迎语 | `server/web_app.py` → `web()` 开头 |
| 修改后端服务地址 | `server/web_app.py` → `base_url = "http://localhost:8001"` |
| 调整图表样式 | `server/price_chart_component.py` → ECharts `option` 字典 |
| 修改数据量阈值默认值 | `server/web_app.py` → `st.number_input(..., value=10, ...)` |
| 新增意图路由 | `server/web_app.py` → 意图识别后的 `if/elif` 分支 |
| 修改启动端口 | `run_client.sh` → `--server.port` 参数 |
