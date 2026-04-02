# web_app_dialogue.py - 支持对话式价格查询（渐进式实体补全）
import streamlit as st
import requests
import json
import re
import base64
from datetime import datetime
import time
from price_chart_component import price_distribution_chart, price_scatter_chart


class CustomApiRequest:
    def __init__(self, base_url: str):
        self.base_url = base_url

    @staticmethod
    def extract_meta_base64(full_str: str):
        match = re.search(r"\[META\](.+)", full_str.strip())
        if not match:
            return None
        try:
            b64_str = match.group(1).strip()
            json_str = base64.b64decode(b64_str).decode("utf-8")
            return json.loads(json_str)
        except Exception as e:
            print("解析 Base64 META 错误：", str(e))
            return None
    
    # ========== 新增：对话式查询方法 ==========
    
    def dialogue_query(self, user_input: str, session_id: str = None) -> dict:
        """
        对话式价格查询（渐进式实体补全）
        
        Args:
            user_input: 用户输入
            session_id: 会话ID（首次调用可为None）
        
        Returns:
            包含会话状态、实体收集进度、下一步操作等的字典
        """
        url = f"{self.base_url}/api/v1/query/dialogue"
        payload = {
            "user_input": user_input,
            "session_id": session_id
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error_message": str(e)}
    
    def dialogue_execute(self, session_id: str) -> dict:
        """
        执行对话会话中已收集条件的查询
        
        Args:
            session_id: 会话ID
        
        Returns:
            价格查询结果
        """
        url = f"{self.base_url}/api/v1/query/dialogue/execute"
        payload = {"session_id": session_id}
        
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error_message": str(e)}
    
    def get_dialogue_session(self, session_id: str) -> dict:
        """获取对话会话状态"""
        url = f"{self.base_url}/api/v1/query/dialogue/{session_id}"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    def clear_dialogue_session(self, session_id: str) -> dict:
        """清除对话会话"""
        url = f"{self.base_url}/api/v1/query/dialogue/{session_id}"
        try:
            resp = requests.delete(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}


# ---------- 二次确认弹窗 ----------
@st.dialog("⚠️ 确认删除")
def confirm_delete(filenames: list[str]):
    st.write(f"确定删除 **{len(filenames)}** 个文件吗？此操作不可恢复。")
    c1, c2 = st.columns(2)
    if c1.button("确认", type="primary", use_container_width=True):
        try:
            url = f"{st.session_state.api.base_url}/api/v1/delete/batch"
            resp = requests.delete(url, json={"filenames": filenames}, timeout=8000)
            resp.raise_for_status()
            res = resp.json()

            if res.get("status") == "success":
                st.success("删除成功，知识库正在更新")
                task_id = res.get("task_id")
                st.session_state.kb_updating = True
                st.session_state.kb_update_task_id = task_id  # 保存 task_id 供主页面轮询
                st.session_state.selected_files.clear()
                st.session_state.file_list = []
                st.rerun()
            else:
                st.error(res.get("detail", "未知错误"))
        except Exception as e:
            st.error(str(e))
    if c2.button("取消", use_container_width=True):
        st.rerun()


# ---------- 处理对话式查询步骤 ----------
def handle_dialogue_step(api, prompt: str, container):
    """
    处理对话式查询的单步交互
    
    Args:
        api: CustomApiRequest 实例
        prompt: 用户输入
        container: Streamlit 容器（chat_message context）
    """
    try:
        # 调用对话式查询接口
        result = api.dialogue_query(
            user_input=prompt,
            session_id=st.session_state.dialogue_session_id
        )
        
        # 调试输出（开发时可开启）
        # st.write("DEBUG:", result)
        
        # 保存会话ID
        st.session_state.dialogue_session_id = result.get("session_id")
        st.session_state.dialogue_entities = result.get("entities", {})
        st.session_state.dialogue_status = result.get("status")
        
        # 在 container 中构建响应内容
        with container:
            # 构建响应文本
            response_text = ""
            
            # 显示渠道推断信息
            channel_info = result.get("channel_info", {})
            if channel_info.get("inferred"):
                response_text += f"💡 {channel_info.get('reason', '')}\n\n"
            
            response_text += result.get("response_text", "")
            
            # 显示当前收集的实体
            entities = result.get("entities", {})
            if entities:
                response_text += "\n\n📋 **已收集信息**：\n"
                field_names = {
                    "materialName": "材料名称",
                    "province": "省份",
                    "city": "城市",
                    "materialModelSpec": "规格型号",
                    "brand": "品牌"
                }
                for field, label in field_names.items():
                    value = entities.get(field)
                    if value:
                        response_text += f"- ✅ {label}：{value}\n"
            
            st.markdown(response_text, unsafe_allow_html=True)
            
            # 检查是否需要显示快捷选项
            quick_options = result.get("quick_options", [])
            can_query = result.get("can_query", False)
            status = result.get("status")
            
            # 保存结果到 session state 供按钮使用
            st.session_state.current_dialogue_result = result
            
            # 显示快捷选项按钮
            if quick_options:
                st.markdown("---")
                st.markdown("**快捷选项：**")
                cols = st.columns(min(len(quick_options), 4))
                for idx, option in enumerate(quick_options):
                    with cols[idx]:
                        btn_key = f"dlg_opt_{result.get('session_id', 'new')}_{idx}"
                        if st.button(option["text"], key=btn_key, use_container_width=True):
                            if option.get("action") == "query_now":
                                st.session_state.dialogue_action = "execute"
                                st.rerun()
                            elif option.get("action") == "continue_collect":
                                st.session_state.dialogue_action = "continue"
                                st.rerun()
                            elif option.get("value"):
                                st.session_state.dialogue_quick_value = option["value"]
                                st.rerun()
            
            # 如果可以查询且状态为ready，显示执行按钮
            if can_query and status == "ready":
                st.markdown("---")
                btn_key = f"execute_{result.get('session_id', 'new')}"
                if st.button("🔍 立即查询", key=btn_key, type="primary", use_container_width=True):
                    st.session_state.dialogue_action = "execute"
                    st.rerun()
            
            # 添加到消息历史（在按钮之后添加，避免重复）
            if not st.session_state.get("dialogue_message_added"):
                st.session_state.messages.append({"role": "assistant", "context": response_text})
                st.session_state.dialogue_message_added = True
                
    except Exception as e:
        with container:
            st.error(f"对话式查询失败：{e}")
            import traceback
            st.error(traceback.format_exc())
        # 重置状态
        reset_dialogue_state()


def execute_dialogue_and_show_result(api):
    """执行对话查询并显示结果"""
    with st.spinner("正在查询价格数据..."):
        result = api.dialogue_execute(st.session_state.dialogue_session_id)
        
        if result.get("success"):
            price_result = result.get("price_result", {})
            
            if price_result.get("success"):
                st.success("✅ 查询成功！")
                
                # 显示价格分析结果
                analysis = price_result.get("metadata", {}).get("price_analysis", {})
                if analysis:
                    with st.expander("📊 价格分析结果", expanded=True):
                        st.json(analysis)
                
                # 显示价格数据表格
                md_table = price_result.get("metadata", {}).get("md_table", "")
                if md_table:
                    with st.expander("📋 价格数据表", expanded=True):
                        st.markdown(md_table, unsafe_allow_html=True)
                
                # 显示图表
                price_data = price_result.get("price_data", [])
                total_count = price_result.get("total_count", 0)
                
                if price_data and total_count > st.session_state.get("data_threshold", 10):
                    with st.expander("📈 价格分布图", expanded=True):
                        price_distribution_chart(price_data, bins=5)
                    with st.expander("📍 价格散点图", expanded=True):
                        price_scatter_chart(price_data)
                
                # 保存到参考文档
                st.session_state.last_docs = [{
                    "filename": "价格查询结果",
                    "context": md_table if md_table else str(analysis),
                    "similarity_score": 1.0,
                    "retriever_name": "价格推荐接口",
                }]
                
            else:
                error_msg = price_result.get("error_message", "查询失败")
                st.error(f"❌ {error_msg}")
        else:
            error_msg = result.get("error_message", "执行查询失败")
            st.error(f"❌ {error_msg}")
        
        # 重置对话状态
        reset_dialogue_state()


def reset_dialogue_state():
    """重置对话状态"""
    st.session_state.dialogue_session_id = None
    st.session_state.dialogue_entities = {}
    st.session_state.dialogue_status = None
    st.session_state.dialogue_quick_value = None
    st.session_state.dialogue_action = None
    st.session_state.dialogue_message_added = False
    st.session_state.current_dialogue_result = None


# ---------- 页面 ----------
def web():
    st.set_page_config(page_title="智诚大模型对话", layout="wide")
    base_url = "http://localhost:8001"
    api = CustomApiRequest(base_url=base_url)
        
    # 1️⃣ 初始化 session_state 变量
    if "show_price_chart" not in st.session_state:
        st.session_state.show_price_chart = False
    if "_price_data_cache" not in st.session_state:
        st.session_state._price_data_cache = []
        
    # ---------- 初始化 upload_key ----------   
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    
    # ========== 新增：对话式查询状态初始化 ==========
    if "dialogue_session_id" not in st.session_state:
        st.session_state.dialogue_session_id = None
    if "dialogue_entities" not in st.session_state:
        st.session_state.dialogue_entities = {}
    if "dialogue_status" not in st.session_state:
        st.session_state.dialogue_status = None  # None, "collecting", "ready"
    if "dialogue_quick_value" not in st.session_state:
        st.session_state.dialogue_quick_value = None
    if "dialogue_action" not in st.session_state:
        st.session_state.dialogue_action = None  # None, "execute", "continue"
    if "dialogue_message_added" not in st.session_state:
        st.session_state.dialogue_message_added = False
    if "current_dialogue_result" not in st.session_state:
        st.session_state.current_dialogue_result = None

    st.session_state.api = api

    # ---------- 悬浮按钮 CSS ----------
    st.markdown(
        """
        <style>
        .fab {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 9999;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.markdown("---")
        st.subheader("📈 图表设置")
        st.session_state.data_threshold = st.number_input("设置数据量阈值（小于等于该值时不绘制图表）", min_value=0, value=10, step=1)
        st.subheader("📤 上传知识库文件")
        
        # ========== 新增：对话式查询控制面板 ==========
        st.markdown("---")
        st.subheader("🔄 对话查询状态")
        if st.session_state.dialogue_session_id:
            st.info(f"会话ID: {st.session_state.dialogue_session_id}")
            st.info(f"状态: {st.session_state.dialogue_status or '未知'}")
            if st.button("🚫 重置对话"):
                reset_dialogue_state()
                st.rerun()
        else:
            st.info("无活跃对话")
        
        st.markdown("---")
        
        # 插入自定义CSS
        st.markdown("""
                    <style>
                    /* 修改拖拽区域的提示文字 */
                    [data-testid="stFileUploaderDropzone"] div div::before {
                        content: "点此上传文件";
                        font-size: 16px;
                        color: #555;
                    }

                    /* 隐藏原本的英文拖拽提示 */
                    [data-testid="stFileUploaderDropzone"] div div span {
                        display: none !important;
                    }

                    /* 修改上传按钮文字 */
                    [data-testid="stFileUploader"] button {
                        display: none !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

        # 文件上传控件
        uploaded_files = st.file_uploader(
            "请选择要上传的文件（每文件大小限制：200MB；支持格式：TXT, PDF, MD",
            type=["txt", "pdf", "md"],
            accept_multiple_files=True,
            key="uploader_key"
        )
        if uploaded_files and st.button("开始上传"):
            with st.spinner("正在上传..."):
                try:
                    files = [("files", (f.name, f, f.type)) for f in uploaded_files]
                    r = requests.post(f"{api.base_url}/api/v1/upload", files=files, timeout=8000)
                    r.raise_for_status()
                    res = r.json()
                    print(res)
                    success = [x for x in res.get("results", []) if x.get("status") == "success"]
                    if success:
                        st.success(f"✅ 成功上传 {len(success)} 个文件，知识库正在更新")
                        st.session_state.kb_updating = True
                        st.session_state.kb_update_task_id = res.get("task_id")

                        # 🔽 清空上传组件
                        st.session_state.upload_key += 1
                        st.rerun()                           # 立即刷新页面
                    else:
                        st.error("所有文件上传失败")
                except Exception as e:
                    st.error(f"文件上传失败：{e}")

        # ===== 统一轮询：上传 & 删除 =====
        if st.session_state.get("kb_updating", False):
            with st.spinner("知识库正在更新..."):
                task_id = st.session_state.get("kb_update_task_id")
                if not task_id:
                    st.error("未获取任务ID")
                    st.session_state.kb_updating = False
                else:
                    progress_bar = st.progress(0)
                    for i in range(10000):
                        resp = requests.get(f"{api.base_url}/api/v1/task_status", params={"task_id": task_id}, timeout=5)
                        if resp.status_code != 200:
                            st.error("查询失败")
                            break
                        status_json = resp.json()
                        if status_json.get("status") == "success":
                            st.session_state.kb_updating = False
                            st.toast("数据库更新完成，可以开始提问啦！")
                            time.sleep(1.5)
                            st.rerun()
                        elif "failed" in status_json.get("status", ""):
                            st.session_state.kb_updating = False
                            st.error("❌ 知识库更新失败")
                            time.sleep(1.5)
                            st.rerun()
                        progress_bar.progress(min((i + 1) / 40, 1.0))
                        time.sleep(2)
                    else:
                        st.session_state.kb_updating = False
                        st.warning("⏱️ 更新超时，请稍后手动刷新")

        st.markdown("---")
        st.subheader("📁 文件管理")

        if "file_list" not in st.session_state:
            st.session_state.file_list = []
        if "selected_files" not in st.session_state:
            st.session_state.selected_files = []

        if st.button("🔄 刷新文件列表", key="refresh_files"):
            with st.spinner("正在获取文件列表..."):
                try:
                    r = requests.get(f"{api.base_url}/api/v1/files", timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    st.session_state.file_list = data.get("files", [])
                except Exception as e:
                    st.error(f"请求失败: {e}")

        if st.session_state.file_list:
            st.markdown("### 文件列表")
            for idx, file in enumerate(st.session_state.file_list):
                with st.container():
                    sel_col, del_col = st.columns([1, 1])
                    selected = sel_col.checkbox(f"{file['filename']}", key=f"sel_{idx}")

                    # ✅ 修复：同步勾选状态与 selected_files 列表
                    if selected:
                        if file["filename"] not in st.session_state.selected_files:
                            st.session_state.selected_files.append(file["filename"])
                    else:
                        if file["filename"] in st.session_state.selected_files:
                            st.session_state.selected_files.remove(file["filename"])

                    st.caption(f"大小：{file['size_mb']:.2f} MB")
                    st.caption(f"修改时间：{datetime.fromisoformat(file['modified_time']).strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown("---")
        else:
            st.info("暂无文件，请先上传或刷新")

    # ---------- 悬浮批量删除按钮 ----------
    if st.session_state.selected_files:
        with st.container():
            st.markdown('<div class="fab">', unsafe_allow_html=True)
            if st.button("🗑️ 批量删除", type="primary", key="fab_delete"):
                confirm_delete(st.session_state.selected_files)
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 聊天界面 ----------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_docs" not in st.session_state:
        st.session_state.last_docs = []

    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("## 👋 欢迎使用RAG对话助手")
        st.info("🌿 很高兴在此与你相遇，愿我们一同在知识的枝叶间，探寻微光与辽阔。")
        
        # ========== 新增：使用说明 ==========
        with st.expander("💡 使用提示（点击展开）"):
            st.markdown("""
            **价格查询新功能 - 智能助手模式**
            
            现在支持对话式价格查询！无需一次性提供所有信息：
            
            1. 直接说"查一下钢筋的价格"
            2. 系统会引导你补充省份、城市等信息
            3. 支持快捷按钮快速选择
            4. 信息完整后自动查询
            
            也支持传统方式：直接说"从信息价查广东省深圳市钢筋的价格"
            """)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["context"])

    if st.session_state.get("kb_updating", False):
        st.info("知识库正在更新，请稍后再试。")
        st.chat_input("知识库更新中，暂不可提问", disabled=True)
    else:
        # ========== 处理对话按钮动作 ==========
        if st.session_state.dialogue_action == "execute":
            # 执行查询
            st.session_state.dialogue_action = None
            st.session_state.dialogue_message_added = False
            with st.chat_message("assistant"):
                execute_dialogue_and_show_result(api)
        
        elif st.session_state.dialogue_action == "continue":
            # 继续收集，重新显示当前状态
            st.session_state.dialogue_action = None
            st.session_state.dialogue_message_added = True  # 标记为已添加，避免重复添加消息
            # 重新渲染当前对话状态
            with st.chat_message("assistant") as assistant_container:
                result = st.session_state.get("current_dialogue_result", {})
                if result:
                    # 构建响应文本
                    response_text = ""
                    
                    # 显示渠道推断信息
                    channel_info = result.get("channel_info", {})
                    if channel_info.get("inferred"):
                        response_text += f"💡 {channel_info.get('reason', '')}\n\n"
                    
                    response_text += result.get("response_text", "")
                    
                    # 显示当前收集的实体
                    entities = result.get("entities", {})
                    if entities:
                        response_text += "\n\n📋 **已收集信息**：\n"
                        field_names = {
                            "materialName": "材料名称",
                            "province": "省份",
                            "city": "城市",
                            "materialModelSpec": "规格型号",
                            "brand": "品牌"
                        }
                        for field, label in field_names.items():
                            value = entities.get(field)
                            if value:
                                response_text += f"- ✅ {label}：{value}\n"
                    
                    assistant_container.markdown(response_text, unsafe_allow_html=True)
                    
                    # 显示快捷选项按钮
                    quick_options = result.get("quick_options", [])
                    can_query = result.get("can_query", False)
                    status = result.get("status")
                    
                    if quick_options:
                        assistant_container.markdown("---")
                        assistant_container.markdown("**快捷选项：**")
                        cols = assistant_container.columns(min(len(quick_options), 4))
                        for idx, option in enumerate(quick_options):
                            with cols[idx]:
                                btn_key = f"dlg_opt_{result.get('session_id', 'new')}_{idx}_continue"
                                if st.button(option["text"], key=btn_key, use_container_width=True):
                                    if option.get("action") == "query_now":
                                        st.session_state.dialogue_action = "execute"
                                        st.rerun()
                                    elif option.get("action") == "continue_collect":
                                        st.session_state.dialogue_action = "continue"
                                        st.rerun()
                                    elif option.get("value"):
                                        st.session_state.dialogue_quick_value = option["value"]
                                        st.rerun()
                    
                    # 如果可以查询且状态为ready，显示执行按钮
                    if can_query and status == "ready":
                        assistant_container.markdown("---")
                        btn_key = f"execute_{result.get('session_id', 'new')}_continue"
                        if assistant_container.button("🔍 立即查询", key=btn_key, type="primary", use_container_width=True):
                            st.session_state.dialogue_action = "execute"
                            st.rerun()
        
        # 检查是否有快捷值需要处理
        elif st.session_state.dialogue_quick_value:
            quick_value = st.session_state.dialogue_quick_value
            st.session_state.dialogue_quick_value = None
            st.session_state.dialogue_message_added = False
            # 将快捷值作为用户输入处理
            prompt = quick_value
            st.session_state.messages.append({"role": "user", "context": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            assistant_container = st.chat_message("assistant")
            with assistant_container:
                assistant_container.markdown("<p style='color: grey;'>处理中...</p>", unsafe_allow_html=True)
                handle_dialogue_step(api, prompt, assistant_container)
        
        elif prompt := st.chat_input("请输入您的问题:" if not st.session_state.dialogue_session_id else "请补充信息（或点击上方按钮）："):
            st.session_state.messages.append({"role": "user", "context": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            assistant_container = st.chat_message("assistant")
            with assistant_container:
                assistant_container.markdown("<p style='color: grey;'>模型思考中，请稍候...</p>", unsafe_allow_html=True)
                full_text = ""
                meta = None

                try:
                    # ========== 修改：优先检查是否处于对话模式 ==========
                    if st.session_state.dialogue_status in ["collecting", "ready"]:
                        # 继续对话式查询
                        handle_dialogue_step(api, prompt, assistant_container)
                    else:
                        # 正常意图识别流程
                        try:
                            # 第一步：意图识别
                            intent_resp = requests.post(
                                f"{api.base_url}/api/v1/query/intent",
                                json={"question": prompt},
                                timeout=10
                            )
                            intent_resp.raise_for_status()
                            intent = intent_resp.json().get("intent", "knowledge_qa")
                        except Exception as e:
                            assistant_container.error(f"意图识别调用失败{e}")
                            intent = "knowledge_qa"  # 默认知识问答

                        if intent == "dangerous_sql":
                            st.info("❌ 用户该意图存在一定风险，请重新提问")
                            return
                        
                        elif intent == "price_recommendation":
                            # ========== 修改：使用对话式查询替代直接流式查询 ==========
                            # 启动对话式收集流程
                            st.session_state.dialogue_message_added = False
                            handle_dialogue_step(api, prompt, assistant_container)
                        
                        elif intent == "knowledge_qa":
                            # 知识问答保持原有逻辑
                            try:
                                endpoint = f"{api.base_url}/api/v1/query/stream"
                                r = requests.post(
                                    endpoint,
                                    json={"question": prompt},
                                    stream=True,
                                    timeout=120,
                                )
                                r.raise_for_status()
                            except Exception as e:
                                assistant_container.error(f"请求接口调用失败{e}")
                                return

                            buffer = ""
                            for chunk in r.iter_lines(decode_unicode=True):
                                if not chunk:
                                    continue
                                if chunk.strip() == "[END]":
                                    continue
                                if chunk.strip().startswith("{") and chunk.strip().endswith("}"):
                                    try:
                                        meta = json.loads(chunk.strip())
                                        if meta and "text" in meta:
                                            final_text = meta["text"].replace("\n", "  \n")
                                            assistant_container.markdown(final_text, unsafe_allow_html=True)
                                            st.session_state.messages.append({"role": "assistant", "context": final_text})
                                    except Exception as e:
                                        print("Meta JSON 解析失败:", e)
                                    continue
                                buffer += chunk
                                assistant_container.markdown(buffer + "▌", unsafe_allow_html=True)

                            if not (meta and "text" in meta):
                                final_text = buffer.replace("\n", "  \n")
                                assistant_container.markdown(final_text, unsafe_allow_html=True)
                                st.session_state.messages.append({"role": "assistant", "context": final_text})

                            # 处理参考文档
                            try:
                                if meta:
                                    st.session_state.last_docs = [
                                        {
                                            "filename": ctx.get("metadata", {}).get("source", f"文档片段 {i+1}"),
                                            "context": ctx.get("page_content", ""),
                                            "similarity_score": 1.0,
                                            "retriever_name": "RAG检索",
                                        }
                                        for i, ctx in enumerate(meta.get("contexts", []))
                                    ]
                            except Exception as e:
                                assistant_container.error(f"处理参考文档失败{e}")
                        
                        else:
                            # 其他意图，尝试使用流式接口
                            try:
                                endpoint = f"{api.base_url}/api/v1/query/stream"
                                r = requests.post(
                                    endpoint,
                                    json={"question": prompt},
                                    stream=True,
                                    timeout=120,
                                )
                                r.raise_for_status()
                                
                                buffer = ""
                                for chunk in r.iter_lines(decode_unicode=True):
                                    if not chunk:
                                        continue
                                    if chunk.strip() == "[END]":
                                        continue
                                    buffer += chunk
                                    assistant_container.markdown(buffer + "▌", unsafe_allow_html=True)
                                
                                final_text = buffer.replace("\n", "  \n")
                                assistant_container.markdown(final_text, unsafe_allow_html=True)
                                st.session_state.messages.append({"role": "assistant", "context": final_text})
                                
                            except Exception as e:
                                assistant_container.error(f"服务调用失败：{e}")
                                
                except Exception as e:
                    assistant_container.error(f"服务调用失败：{e}")

    # ---------- 弹出价格分布图 ----------
    if st.session_state.show_price_chart:
        show_price_chart(st.session_state._price_data_cache)
        st.session_state.show_price_chart = False

    # ---------- 在「参考文档」展示区域做兼容处理 ----------
    if st.session_state.get("last_docs"):
        st.markdown("---")
        st.subheader("📎 参考文档")
        idx = st.selectbox(
            "选择要查看的参考文档：",
            range(len(st.session_state.last_docs)),
            format_func=lambda i: f"{i+1}. {st.session_state.last_docs[i]['filename']}",
        )
        with st.expander("📄 文档内容"):
            doc = st.session_state.last_docs[idx]
            if doc["retriever_name"] == "价格推荐接口":
                st.markdown(doc["context"], unsafe_allow_html=True)
            else:
                st.markdown(
                    f"""
- **文件名**: `{doc['filename']}`  
- **相似度**: `{doc['similarity_score']}`  
- **检索方式**: `{doc['retriever_name']}`    
```
{doc['context']}
```
""")


if __name__ == "__main__":
    web()
