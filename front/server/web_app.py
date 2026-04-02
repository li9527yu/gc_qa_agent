# web_app.py
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
    
    # ---------- 初始化后台任务状态 ----------
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = False
    if "upload_message" not in st.session_state:
        st.session_state.upload_message = ""

    # ---------- 初始化服务状态检查 ----------
    if "service_status" not in st.session_state:
        st.session_state.service_status = None
    if "last_status_check" not in st.session_state:
        st.session_state.last_status_check = 0

    st.session_state.api = api

    # ---------- 服务状态检查函数 ----------
    def check_service_status():
        """检查后端服务状态"""
        try:
            resp = requests.get(f"{api.base_url}/api/v1/service/status", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

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
        # st.info("📝 注意：此版本已适配您的RAG服务")
        # st.info("🔗 服务地址：http://localhost:8001")
        st.markdown("---")
        st.subheader("📈 图表设置")
        st.session_state.data_threshold = st.number_input("设置数据量阈值（小于等于该值时不绘制图表）", min_value=0, value=10, step=1)
        st.subheader("📤 上传知识库文件")
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
            key=f"uploader_{st.session_state.upload_key}"
        )
        
        # 显示上传成功提示
        if st.session_state.upload_success:
            st.success(st.session_state.upload_message)
            # 重置状态
            st.session_state.upload_success = False
            st.session_state.upload_message = ""
        
        # 当有文件被选择时，显示文件选择列表和操作按钮
        if uploaded_files:
            st.markdown("**📎 待上传文件列表**（取消勾选可排除不需要的文件）")
            
            # 为每个文件创建复选框，默认全选
            selected_files_to_upload = []
            for idx, f in enumerate(uploaded_files):
                # 使用文件名+索引作为key，避免重名文件冲突
                checkbox_key = f"chk_{idx}_{f.name}"
                if st.checkbox(f"📄 {f.name} ({f.size / 1024:.1f} KB)", value=True, key=checkbox_key):
                    selected_files_to_upload.append(f)
            
            # 显示已选择数量
            st.caption(f"已选择 {len(selected_files_to_upload)} / {len(uploaded_files)} 个文件")
            
            # 操作按钮行
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                start_upload = st.button("⬆️ 开始上传", use_container_width=True, disabled=len(selected_files_to_upload) == 0)
            with col2:
                # 快速全选/取消全选
                if st.button("☐ 取消全选", use_container_width=True):
                    # 通过清空key来取消所有选择（下次渲染时checkbox恢复False）
                    for idx, f in enumerate(uploaded_files):
                        if f"chk_{idx}_{f.name}" in st.session_state:
                            del st.session_state[f"chk_{idx}_{f.name}"]
                    st.rerun()
            with col3:
                if st.button("❌ 清除全部", use_container_width=True):
                    st.session_state.upload_key += 1
                    st.rerun()
        else:
            start_upload = False
        
        if start_upload:
            # 使用独立的 spinner，不阻塞整个界面
            upload_placeholder = st.empty()
            upload_placeholder.info(f"📤 正在上传 {len(selected_files_to_upload)} 个文件...")
            try:
                files = [("files", (f.name, f.getvalue(), f.type)) for f in selected_files_to_upload]
                r = requests.post(f"{api.base_url}/api/v1/upload", files=files, timeout=300)
                r.raise_for_status()
                res = r.json()
                print(res)
                success = [x for x in res.get("results", []) if x.get("status") == "success"]
                failed = [x for x in res.get("results", []) if x.get("status") != "success"]
                
                if success:
                    # 设置后台任务状态
                    st.session_state.kb_updating = True
                    st.session_state.kb_update_task_id = res.get("task_id")
                    st.session_state.upload_success = True
                    st.session_state.upload_message = f"✅ 成功上传 {len(success)} 个文件，知识库后台更新中"
                    if failed:
                        st.session_state.upload_message += f"（{len(failed)} 个失败）"
                    
                    # 🔽 清空上传组件
                    st.session_state.upload_key += 1
                    st.rerun()
                else:
                    st.error(f"❌ 所有文件上传失败: {failed[0].get('detail', '未知错误') if failed else '未知错误'}")
                    upload_placeholder.empty()
            except Exception as e:
                st.error(f"文件上传失败：{e}")
                upload_placeholder.empty()

        # ===== 后台轮询：上传 & 删除 =====
        if st.session_state.get("kb_updating", False):
            task_id = st.session_state.get("kb_update_task_id")
            if not task_id:
                st.error("未获取任务ID")
                st.session_state.kb_updating = False
            else:
                # 使用 st.status 显示后台进度，不阻塞用户操作
                with st.status("🔄 知识库后台更新中...", expanded=False) as status:
                    try:
                        # 最多轮询 60 次 × 2秒 = 2分钟
                        for i in range(60):
                            resp = requests.get(f"{api.base_url}/api/v1/task_status", params={"task_id": task_id}, timeout=5)
                            if resp.status_code != 200:
                                status.update(label="❌ 查询任务状态失败", state="error")
                                st.session_state.kb_updating = False
                                break
                            status_json = resp.json()
                            if status_json.get("status") == "success":
                                status.update(label="✅ 知识库更新完成！", state="complete")
                                st.session_state.kb_updating = False
                                st.toast("🎉 知识库更新完成，可以开始提问啦！")
                                # 更新完成后自动刷新文件列表
                                try:
                                    r = requests.get(f"{api.base_url}/api/v1/files", timeout=10)
                                    r.raise_for_status()
                                    data = r.json()
                                    files = data.get("files", [])
                                    # 按修改时间倒序排列
                                    files.sort(key=lambda x: x.get("modified_time", ""), reverse=True)
                                    st.session_state.file_list = files
                                except Exception as e:
                                    st.warning(f"自动刷新文件列表失败: {e}")
                                time.sleep(1.5)
                                st.rerun()
                                break
                            elif "failed" in status_json.get("status", ""):
                                status.update(label="❌ 知识库更新失败", state="error")
                                st.session_state.kb_updating = False
                                time.sleep(1.5)
                                st.rerun()
                                break
                            # 更新进度 - 优先使用服务状态中的详细进度
                            service_status = check_service_status()
                            if service_status and service_status.get("is_rebuilding"):
                                progress_info = service_status.get("rebuild_progress", {})
                                percent = progress_info.get("percent", min((i + 1) / 30 * 100, 95))
                                message = progress_info.get("message", "更新中...")
                                status.write(f"⏳ {message} ({percent:.0f}%)")
                            else:
                                progress = min((i + 1) / 30, 0.95)
                                status.write(f"⏳ 更新进度: {progress*100:.0f}%")
                            time.sleep(2)
                        else:
                            status.update(label="⏱️ 更新超时，请稍后手动刷新文件列表", state="error")
                            st.session_state.kb_updating = False
                    except Exception as e:
                        status.update(label=f"❌ 更新出错: {e}", state="error")
                        st.session_state.kb_updating = False

        st.markdown("---")
        st.subheader("📁 文件管理")

        if "file_list" not in st.session_state:
            st.session_state.file_list = []
        if "selected_files" not in st.session_state:
            st.session_state.selected_files = []

        # 自动加载文件列表（如果为空）
        if not st.session_state.file_list:
            try:
                r = requests.get(f"{api.base_url}/api/v1/files", timeout=10)
                r.raise_for_status()
                data = r.json()
                files = data.get("files", [])
                # 按修改时间倒序排列，最新上传的在最前面
                files.sort(key=lambda x: x.get("modified_time", ""), reverse=True)
                st.session_state.file_list = files
            except Exception:
                pass  # 静默失败，不阻塞界面
        
        if st.button("🔄 刷新文件列表", key="refresh_files"):
            with st.spinner("正在获取文件列表..."):
                try:
                    r = requests.get(f"{api.base_url}/api/v1/files", timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    files = data.get("files", [])
                    # 按修改时间倒序排列，最新上传的在最前面
                    files.sort(key=lambda x: x.get("modified_time", ""), reverse=True)
                    st.session_state.file_list = files
                    st.success(f"✅ 已刷新，共 {len(files)} 个文件")
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

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["context"])

    # 检查服务状态（每5秒刷新一次）
    current_time = time.time()
    if current_time - st.session_state.last_status_check > 5:
        st.session_state.service_status = check_service_status()
        st.session_state.last_status_check = current_time
    
    service_status = st.session_state.service_status
    is_rebuilding = service_status and service_status.get("is_rebuilding", False)
    
    # 如果正在重建，显示警告提示
    if is_rebuilding:
        progress = service_status.get("rebuild_progress", {})
        st.warning(
            f"🔄 **知识库正在更新** ({progress.get('message', '处理中...')}) "
            f"- 进度 {progress.get('percent', 0):.0f}%\n\n"
            f"💡 知识问答功能暂不可用，但您仍可使用 **价格推荐** 功能。"
        )
    
    # 聊天输入始终可用（价格推荐不受重建影响）
    if prompt := st.chat_input("请输入您的问题:" if not is_rebuilding else "知识库更新中，价格推荐仍可用:"):
            st.session_state.messages.append({"role": "user", "context": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("<p style='color: grey;'>模型思考中，请稍候...</p>", unsafe_allow_html=True)
                full_text = ""
                meta = None

                try:
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
                        placeholder.error(f"意图识别调用失败{e}")


                    if intent =="dangerous_sql":
                        st.info("❌ 用户该意图存在一定风险，请重新提问")
                        return
                    
                    # 检查知识库重建状态，仅对知识问答进行限制
                    if intent == "knowledge_qa" and is_rebuilding:
                        progress = service_status.get("rebuild_progress", {}) if service_status else {}
                        placeholder.warning(
                            f"⚠️ **知识库正在更新中，知识问答功能暂不可用。**\n\n"
                            f"当前进度: {progress.get('message', '处理中...')} ({progress.get('percent', 0):.0f}%)\n\n"
                            f"💡 提示：您可以使用 **价格推荐** 功能查询材料价格，"
                            f"例如：\"查询铝合金门窗的信息价\""
                        )
                        # 记录一条助手消息到对话历史
                        error_msg = f"知识库正在更新中（{progress.get('percent', 0):.0f}%），请稍后再试知识问答。"
                        st.session_state.messages.append({"role": "assistant", "context": error_msg})
                        return
                    
                    # 价格推荐在重建期间仍然可用，显示轻微提示
                    if intent == "price_recommendation" and is_rebuilding:
                        st.toast("💡 知识库正在更新，价格推荐功能不受影响", icon="ℹ️")
                    
                    try:
                        # 第二步：选择接口
                        endpoint=f"{api.base_url}/api/v1/query/stream"
                        if intent == "price_recommendation":
                            endpoint=f"{api.base_url}/api/v1/query/price"
                        elif intent == "knowledge_qa":
                            endpoint=f"{api.base_url}/api/v1/query/stream"
                        else:
                            endpoint=f"{api.base_url}/api/v1/query/question"

                        r = requests.post(
                            endpoint,
                            json={"question": prompt},
                            stream=True,
                            timeout=120,
                        )
                        r.raise_for_status()
                    except Exception as e:
                        placeholder.error(f"请求接口调用失败{e}")
                        return
                    
                    # 流式读取响应
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
                                    # ✅ 用 text 字段渲染最终回答
                                    final_text = meta["text"].replace("\n", "  \n")
                                    placeholder.markdown(final_text, unsafe_allow_html=True)
                                    st.session_state.messages.append({"role": "assistant", "context": final_text})
                            except Exception as e:
                                print("Meta JSON 解析失败:", e)
                            continue
                        buffer += chunk
                        placeholder.markdown(buffer + "▌", unsafe_allow_html=True) 

                    # ✅ 如果没有 meta["text"]，用 buffer
                    if not (meta and "text" in meta):
                        final_text = buffer.replace("\n", "  \n")
                        placeholder.markdown(final_text, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "context": final_text})
                    
                    # 第三步：处理参考文档或价格原始数据（无论是否有 text 字段都执行）
                    try:
                        if meta:
                            if intent == "price_recommendation":
                                md_table   = meta.get("metadata", {}).get("md_table", "")
                                price_data = meta.get("price_data", [])
                                total_count = meta.get("metadata", {}).get("total_count", 0)
                                st.session_state.last_docs = [{
                                    "filename": "价格参考表",
                                    "context": md_table,
                                    "similarity_score": 1.0,
                                    "retriever_name": "价格推荐接口",
                                }]

                                if meta.get("channel", 0) == "unknown":
                                    placeholder.warning("用户未提及任何渠道关键词，请提供查询渠道关键词，例如：信息价、厂商报价等")
                                elif total_count==0:
                                    placeholder.warning("未返回任何价格数据，可能是数据库中无相关样本")
                                elif total_count <= st.session_state.data_threshold:
                                    placeholder.warning(f"数据库中样本数据量仅有{total_count}条小于等于{st.session_state.data_threshold}，不建议进行数据分析和价格推荐，以下是原始数据")
                                elif meta.get("success", False) == False:
                                    placeholder.warning(meta.get("metadata", {}).get("error_message", "未知错误"))
                                else:
                                    # ✅ 自动渲染图表
                                    with st.expander("📊 价格分布图（点击展开）", expanded=True):
                                        price_distribution_chart(price_data, bins=5)
                                    # ✅ 自动渲染散点图
                                    with st.expander("📈 价格散点图（点击展开）", expanded=True):
                                        price_scatter_chart(price_data)
                            else:
                                # ✅ 知识问答：保持原来的上下文文档
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
                        placeholder.error(f"渲染图表失败{e}")
                                

                except Exception as e:
                    placeholder.error(f"服务调用失败：{e}")

    # ---------- 弹出价格分布图 ----------
    if st.session_state.show_price_chart:
        show_price_chart(st.session_state._price_data_cache)
        st.session_state.show_price_chart = False   # 弹完即关

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
                # ✅ 直接渲染 markdown 表格
                st.markdown(doc["context"], unsafe_allow_html=True)
            else:
                # ✅ 原知识问答格式
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