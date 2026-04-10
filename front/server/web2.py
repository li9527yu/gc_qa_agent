# web2.py - 改进版 RAG 对话助手前端
import streamlit as st
import requests
import json
import re
import base64
from datetime import datetime
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# 导入图表组件
try:
    from price_chart_component import price_distribution_chart, price_scatter_chart
except ImportError:
    # 如果导入失败，提供空实现
    def price_distribution_chart(price_data, bins=5):
        st.write("价格分布图表（需要安装 streamlit_echarts）")
    def price_scatter_chart(price_data):
        st.write("价格散点图表（需要安装 streamlit_echarts）")


@dataclass
class ApiConfig:
    """API 配置"""
    base_url: str = "http://localhost:8001"
    timeout_short: int = 10      # 短时间请求超时（秒）
    timeout_long: int = 120      # 长时间请求超时（秒）
    max_upload_size_mb: int = 200


class RagApiClient:
    """RAG API 客户端封装"""
    
    def __init__(self, config: ApiConfig):
        self.config = config
    
    def _post(self, endpoint: str, json_data: Dict = None, files=None, 
              timeout: int = None, stream: bool = False) -> requests.Response:
        """发送 POST 请求"""
        url = f"{self.config.base_url}{endpoint}"
        timeout = timeout or self.config.timeout_short
        
        if files:
            return requests.post(url, files=files, timeout=timeout)
        return requests.post(url, json=json_data, timeout=timeout, stream=stream)
    
    def _get(self, endpoint: str, params: Dict = None, timeout: int = None) -> requests.Response:
        """发送 GET 请求"""
        url = f"{self.config.base_url}{endpoint}"
        timeout = timeout or self.config.timeout_short
        return requests.get(url, params=params, timeout=timeout)
    
    def _delete(self, endpoint: str, json_data: Dict = None, timeout: int = None) -> requests.Response:
        """发送 DELETE 请求"""
        url = f"{self.config.base_url}{endpoint}"
        timeout = timeout or self.config.timeout_short
        return requests.delete(url, json=json_data, timeout=timeout)
    
    # ========== 文件管理接口 ==========
    
    def upload_files(self, files: List) -> Dict[str, Any]:
        """批量上传文件"""
        try:
            file_tuples = [("files", (f.name, f, f.type)) for f in files]
            resp = self._post("/api/v1/upload", files=file_tuples, 
                            timeout=self.config.timeout_long)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def list_files(self) -> Dict[str, Any]:
        """获取文件列表"""
        try:
            resp = self._get("/api/v1/files")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e), "status": "error", "files": []}
    
    def delete_files(self, filenames: List[str]) -> Dict[str, Any]:
        """批量删除文件"""
        try:
            resp = self._delete("/api/v1/delete/batch", 
                              json_data={"filenames": filenames},
                              timeout=self.config.timeout_long)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        try:
            resp = self._get("/api/v1/task_status", params={"task_id": task_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    # ========== 查询接口 ==========
    
    def query_stream(self, question: str, num_docs: int = 10):
        """流式知识问答"""
        try:
            resp = self._post("/api/v1/query/stream",
                            json_data={"question": question, "num_docs": num_docs},
                            stream=True, timeout=self.config.timeout_long)
            resp.raise_for_status()
            return resp
        except Exception as e:
            raise e
    
    def query_price_stream(self, question: str):
        """流式价格查询"""
        try:
            resp = self._post("/api/v1/query/price",
                            json_data={"question": question},
                            stream=True, timeout=self.config.timeout_long)
            resp.raise_for_status()
            return resp
        except Exception as e:
            raise e
    
    def query_price_direct(self, datatype: str, material_list: List[Dict], 
                          question: str = ""):
        """直接价格查询（流式）"""
        try:
            resp = self._post("/api/v1/query/price/direct",
                            json_data={
                                "datatype": datatype,
                                "list": material_list,
                                "question": question
                            },
                            stream=True, timeout=self.config.timeout_long)
            resp.raise_for_status()
            return resp
        except Exception as e:
            raise e
    
    # ========== 对话式查询接口 ==========
    
    def dialogue_query(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """对话式价格查询（渐进式实体补全）"""
        try:
            resp = self._post("/api/v1/query/dialogue",
                            json_data={"user_input": user_input, "session_id": session_id},
                            timeout=self.config.timeout_short)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error_message": str(e)}
    
    def dialogue_execute(self, session_id: str) -> Dict[str, Any]:
        """执行对话会话中的查询"""
        try:
            resp = self._post("/api/v1/query/dialogue/execute",
                            json_data={"session_id": session_id},
                            timeout=self.config.timeout_long)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"success": False, "error_message": str(e)}
    
    def get_dialogue_session(self, session_id: str) -> Dict[str, Any]:
        """获取对话会话状态"""
        try:
            resp = self._get(f"/api/v1/query/dialogue/{session_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    def clear_dialogue_session(self, session_id: str) -> Dict[str, Any]:
        """清除对话会话"""
        try:
            resp = self._delete(f"/api/v1/query/dialogue/{session_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}


# ========== Streamlit UI 组件 ==========

def render_sidebar(api: RagApiClient) -> None:
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("---")
        
        # 图表设置
        st.subheader("📈 图表设置")
        st.session_state.data_threshold = st.number_input(
            "数据量阈值（小于等于该值时不绘制图表）",
            min_value=0, value=10, step=1,
            help="当价格数据条数小于等于此值时，不显示图表"
        )
        
        # 查询模式设置
        st.subheader("🔧 查询模式")
        
        # 功能模式选择（硬核分离两种功能）
        st.session_state.query_mode = st.radio(
            "选择功能",
            options=["knowledge_qa", "price_query"],
            format_func=lambda x: "🔍 知识问答" if x == "knowledge_qa" else "💰 价格查询",
            help="明确选择要使用的功能：知识问答基于文档语料库，价格查询基于材料价格数据库"
        )
        
        # 只有在价格查询模式下才显示对话式查询选项
        if st.session_state.query_mode == "price_query":
            st.session_state.use_dialogue_mode = st.toggle(
                "使用对话式价格查询",
                value=True,
                help="启用后，价格查询会采用渐进式实体补全模式"
            )
        
        # 对话状态显示
        if st.session_state.get("dialogue_session_id"):
            st.markdown("---")
            st.subheader("🔄 对话状态")
            st.info(f"会话ID: {st.session_state.dialogue_session_id[:8]}...")
            st.info(f"状态: {st.session_state.get('dialogue_status', '未知')}")
            if st.button("🚫 重置对话", use_container_width=True):
                reset_dialogue_state()
                st.rerun()
        
        st.markdown("---")
        
        # 文件上传
        st.subheader("📤 上传文件")
        
        # 自定义上传样式
        st.markdown("""
        <style>
        [data-testid="stFileUploaderDropzone"] div div::before {
            content: "点此上传文件";
            font-size: 16px;
            color: #555;
        }
        [data-testid="stFileUploaderDropzone"] div div span {
            display: none !important;
        }
        [data-testid="stFileUploader"] button {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "请选择要上传的文件（每文件大小限制：200MB；支持格式：TXT, PDF, MD）",
            type=["txt", "pdf", "md"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.upload_key}"
        )
        
        if uploaded_files and st.button("开始上传", use_container_width=True):
            with st.spinner("正在上传..."):
                result = api.upload_files(uploaded_files)
                
                if "error" in result:
                    st.error(f"上传失败：{result['error']}")
                elif result.get("status") == "success" or result.get("results"):
                    success_count = len([r for r in result.get("results", []) 
                                        if r.get("status") == "success"])
                    if success_count > 0:
                        st.success(f"✅ 成功上传 {success_count} 个文件，知识库正在更新")
                        st.session_state.kb_updating = True
                        st.session_state.kb_update_task_id = result.get("task_id")
                        st.session_state.upload_key += 1
                        st.rerun()
                    else:
                        st.error("所有文件上传失败")
                else:
                    st.error("上传响应异常")
        
        # 任务状态轮询
        if st.session_state.get("kb_updating"):
            with st.spinner("知识库正在更新..."):
                task_id = st.session_state.get("kb_update_task_id")
                if not task_id:
                    st.error("未获取任务ID")
                    st.session_state.kb_updating = False
                else:
                    progress_bar = st.progress(0)
                    for i in range(100):
                        status_result = api.get_task_status(task_id)
                        status = status_result.get("status", "unknown")
                        
                        if status == "success":
                            st.session_state.kb_updating = False
                            st.toast("✅ 数据库更新完成，可以开始提问啦！")
                            time.sleep(1.5)
                            st.rerun()
                            break
                        elif "failed" in status:
                            st.session_state.kb_updating = False
                            st.error(f"❌ 知识库更新失败: {status}")
                            time.sleep(1.5)
                            st.rerun()
                            break
                        
                        progress_bar.progress(min((i + 1) / 50, 0.95))
                        time.sleep(2)
                    else:
                        st.session_state.kb_updating = False
                        st.warning("⏱️ 更新超时，请稍后手动刷新")
        
        st.markdown("---")
        
        # 文件管理
        st.subheader("📁 文件管理")
        
        if st.button("🔄 刷新文件列表", key="refresh_files", use_container_width=True):
            with st.spinner("正在获取文件列表..."):
                result = api.list_files()
                if "error" not in result:
                    st.session_state.file_list = result.get("files", [])
                else:
                    st.error(f"获取文件列表失败: {result['error']}")
        
        if st.session_state.file_list:
            st.markdown(f"**共 {len(st.session_state.file_list)} 个文件**")
            
            for idx, file in enumerate(st.session_state.file_list):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        selected = st.checkbox(
                            file['filename'],
                            key=f"file_sel_{idx}",
                            value=file['filename'] in st.session_state.selected_files
                        )
                        if selected:
                            if file['filename'] not in st.session_state.selected_files:
                                st.session_state.selected_files.append(file['filename'])
                        else:
                            if file['filename'] in st.session_state.selected_files:
                                st.session_state.selected_files.remove(file['filename'])
                    
                    with col2:
                        st.caption(f"{file['size_mb']:.2f} MB")
                    
                    st.caption(f"修改时间: {datetime.fromisoformat(file['modified_time']).strftime('%Y-%m-%d %H:%M')}")
                    st.divider()
        else:
            st.info("暂无文件，请先上传或刷新")


@st.dialog("⚠️ 确认删除")
def confirm_delete_dialog(api: RagApiClient, filenames: List[str]):
    """删除确认对话框"""
    st.write(f"确定删除 **{len(filenames)}** 个文件吗？此操作不可恢复。")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("确认", type="primary", use_container_width=True):
            with st.spinner("正在删除..."):
                result = api.delete_files(filenames)
                
                if "error" not in result:
                    st.success(f"删除成功，知识库正在更新")
                    task_id = result.get("task_id")
                    st.session_state.kb_updating = True
                    st.session_state.kb_update_task_id = task_id
                    st.session_state.selected_files.clear()
                    st.session_state.file_list = []
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"删除失败: {result['error']}")
    
    with col2:
        if st.button("取消", use_container_width=True):
            st.rerun()


def render_floating_delete_button(api: RagApiClient) -> None:
    """渲染悬浮删除按钮"""
    if st.session_state.selected_files:
        st.markdown("""
        <style>
        .fab {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 9999;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="fab">', unsafe_allow_html=True)
            if st.button(f"🗑️ 删除 {len(st.session_state.selected_files)} 个文件", 
                        type="primary", key="fab_delete"):
                confirm_delete_dialog(api, st.session_state.selected_files)
            st.markdown("</div>", unsafe_allow_html=True)


def reset_dialogue_state() -> None:
    """重置对话状态"""
    st.session_state.dialogue_session_id = None
    st.session_state.dialogue_entities = {}
    st.session_state.dialogue_status = None
    st.session_state.dialogue_quick_value = None
    st.session_state.dialogue_action = None
    st.session_state.current_dialogue_result = None


def render_dialogue_buttons(result: Dict[str, Any]) -> None:
    """渲染对话式查询的按钮"""
    quick_options = result.get("quick_options", [])
    can_query = result.get("can_query", False)
    status = result.get("status")
    session_id = result.get("session_id", "")
    
    # 显示快捷选项
    if quick_options:
        st.markdown("---")
        st.markdown("**快捷选项：**")
        cols = st.columns(min(len(quick_options), 4))
        
        for idx, option in enumerate(quick_options):
            with cols[idx]:
                btn_key = f"dlg_opt_{session_id}_{idx}"
                if st.button(option["text"], key=btn_key, use_container_width=True):
                    if option.get("action") == "query_now":
                        st.session_state.dialogue_action = "execute"
                    elif option.get("action") == "continue_collect":
                        st.session_state.dialogue_action = "continue"
                    elif option.get("value"):
                        st.session_state.dialogue_quick_value = option["value"]
                    st.rerun()
    
    # 显示立即查询按钮
    if can_query and status == "ready":
        st.markdown("---")
        btn_key = f"execute_{session_id}"
        if st.button("🔍 立即查询", key=btn_key, type="primary", use_container_width=True):
            st.session_state.dialogue_action = "execute"
            st.rerun()


def execute_dialogue_query(api: RagApiClient) -> None:
    """执行对话式查询并显示结果"""
    with st.spinner("正在查询价格数据..."):
        result = api.dialogue_execute(st.session_state.dialogue_session_id)
        
        if result.get("success"):
            price_result = result.get("price_result", {})
            
            if price_result.get("success"):
                st.success("✅ 查询成功！")
                
                # 显示价格分析
                analysis = price_result.get("metadata", {}).get("price_analysis", {})
                if analysis:
                    with st.expander("📊 价格分析结果", expanded=True):
                        st.json(analysis)
                
                # 显示价格表格
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


def handle_dialogue_step(api: RagApiClient, user_input: str) -> None:
    """处理对话式查询的单步"""
    try:
        # 调用对话式查询接口
        result = api.dialogue_query(
            user_input=user_input,
            session_id=st.session_state.dialogue_session_id
        )
        
        if not result.get("success", True) and "error_message" in result:
            st.error(f"查询失败: {result.get('error_message')}")
            return
        
        # 更新会话状态
        st.session_state.dialogue_session_id = result.get("session_id")
        st.session_state.dialogue_entities = result.get("entities", {})
        st.session_state.dialogue_status = result.get("status")
        st.session_state.current_dialogue_result = result
        
        # 构建响应文本
        response_parts = []
        
        # 渠道推断信息
        channel_info = result.get("channel_info", {})
        if channel_info.get("inferred"):
            response_parts.append(f"💡 {channel_info.get('reason', '')}\n")
        
        response_parts.append(result.get("response_text", ""))
        
        # 已收集实体
        entities = result.get("entities", {})
        if entities:
            response_parts.append("\n\n📋 **已收集信息**：")
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
                    response_parts.append(f"\n- ✅ {label}：{value}")
        
        response_text = "\n".join(response_parts)
        st.markdown(response_text, unsafe_allow_html=True)
        
        # 渲染按钮
        render_dialogue_buttons(result)
        
        # 添加到消息历史
        if not st.session_state.get("dialogue_message_added"):
            st.session_state.messages.append({"role": "assistant", "context": response_text})
            st.session_state.dialogue_message_added = True
    
    except Exception as e:
        st.error(f"对话式查询失败：{e}")
        import traceback
        st.error(traceback.format_exc())
        reset_dialogue_state()


def process_streaming_response(resp: requests.Response, placeholder) -> Dict[str, Any]:
    """处理流式响应"""
    buffer = ""
    meta = None
    
    for chunk in resp.iter_lines(decode_unicode=True):
        if not chunk:
            continue
        if chunk.strip() == "[END]":
            continue
        
        # 尝试解析 JSON (meta 信息)
        if chunk.strip().startswith("{") and chunk.strip().endswith("}"):
            try:
                meta = json.loads(chunk.strip())
                if meta and "text" in meta:
                    final_text = meta["text"].replace("\n", "  \n")
                    placeholder.markdown(final_text, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "context": final_text})
                    return meta
            except json.JSONDecodeError:
                pass
        
        buffer += chunk
        placeholder.markdown(buffer + "▌", unsafe_allow_html=True)
    
    # 如果没有解析到 meta，使用 buffer
    if not (meta and "text" in meta):
        final_text = buffer.replace("\n", "  \n")
        placeholder.markdown(final_text, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "context": final_text})
    
    return meta or {}


def handle_knowledge_qa(api: RagApiClient, question: str, placeholder) -> None:
    """处理知识问答"""
    try:
        resp = api.query_stream(question)
        meta = process_streaming_response(resp, placeholder)
        
        # 处理参考文档
        if meta and "contexts" in meta:
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
        placeholder.error(f"知识问答请求失败：{e}")


def handle_price_query(api: RagApiClient, question: str, placeholder) -> None:
    """处理价格查询（传统流式模式）"""
    try:
        resp = api.query_price_stream(question)
        meta = process_streaming_response(resp, placeholder)
        
        if not meta:
            return
        
        # 检查错误情况
        if meta.get("channel") == "unknown":
            placeholder.warning("⚠️ 用户未提及任何渠道关键词，请提供查询渠道关键词，例如：信息价、厂商报价等")
            return
        
        metadata = meta.get("metadata", {})
        total_count = metadata.get("total_count", 0)
        
        if total_count == 0:
            placeholder.warning("⚠️ 未返回任何价格数据，可能是数据库中无相关样本")
            return
        elif not meta.get("success", True):
            error_msg = metadata.get("error_message", "查询失败")
            placeholder.warning(f"⚠️ {error_msg}")
            return
        
        # 保存参考文档
        st.session_state.last_docs = [{
            "filename": "价格参考表",
            "context": metadata.get("md_table", ""),
            "similarity_score": 1.0,
            "retriever_name": "价格推荐接口",
        }]
        
        # 显示价格分析
        price_analysis = metadata.get("price_analysis", {})
        if price_analysis:
            with st.expander("📊 价格分析结果", expanded=True):
                st.json(price_analysis)
        
        # 显示图表
        price_data = meta.get("price_data", [])
        if price_data and total_count > st.session_state.get("data_threshold", 10):
            with st.expander("📈 价格分布图", expanded=True):
                price_distribution_chart(price_data, bins=5)
            with st.expander("📍 价格散点图", expanded=True):
                price_scatter_chart(price_data)
        elif total_count <= st.session_state.get("data_threshold", 10):
            placeholder.info(f"ℹ️ 数据库中样本数据量仅有 {total_count} 条，小于等于阈值，显示原始数据")
    
    except Exception as e:
        placeholder.error(f"价格查询请求失败：{e}")


def handle_price_query_dialogue(api: RagApiClient, question: str) -> None:
    """处理价格查询（对话式模式）"""
    st.session_state.dialogue_message_added = False
    handle_dialogue_step(api, question)


def render_chat_interface(api: RagApiClient) -> None:
    """渲染聊天界面"""
    # 显示欢迎信息
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("## 👋 欢迎使用智诚 RAG 对话助手")
        st.info("🌿 很高兴在此与你相遇，愿我们一同在知识的枝叶间，探寻微光与辽阔。")
        
        with st.expander("💡 使用说明"):
            st.markdown("""
            **支持的功能：**
            
            1. **知识问答** - 基于上传的文档语料库进行问答
               - 示例："什么是工程造价？"
            
            2. **价格查询** - 查询材料价格信息
               - 对话式：直接说"查一下钢筋的价格"，系统会引导你补充信息
               - 直接式："从信息价查广东省深圳市钢筋的价格"
            
            3. **文件管理** - 在侧边栏上传、查看和删除知识库文件
            """)
    
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["context"])
    
    # 知识库更新中
    if st.session_state.get("kb_updating"):
        st.info("⏳ 知识库正在更新，请稍后再试。")
        st.chat_input("知识库更新中，暂不可提问", disabled=True)
        return
    
    # 处理对话式查询的动作
    if st.session_state.dialogue_action == "execute":
        st.session_state.dialogue_action = None
        st.session_state.dialogue_message_added = False
        with st.chat_message("assistant"):
            execute_dialogue_query(api)
        return
    
    if st.session_state.dialogue_action == "continue":
        st.session_state.dialogue_action = None
        st.session_state.dialogue_message_added = True
        # 重新渲染当前状态
        result = st.session_state.get("current_dialogue_result", {})
        if result:
            with st.chat_message("assistant"):
                _render_dialogue_content(result)
        return
    
    if st.session_state.dialogue_quick_value:
        quick_value = st.session_state.dialogue_quick_value
        st.session_state.dialogue_quick_value = None
        st.session_state.dialogue_message_added = False
        
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "context": quick_value})
        with st.chat_message("user"):
            st.markdown(quick_value)
        
        # 处理快捷值
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("<p style='color: grey;'>处理中...</p>", unsafe_allow_html=True)
            handle_dialogue_step(api, quick_value)
        return
    
    # 显示聊天输入框
    placeholder_text = "请输入您的问题..."
    if st.session_state.get("dialogue_session_id"):
        placeholder_text = "请补充信息（或点击上方按钮）..."
    
    if prompt := st.chat_input(placeholder_text):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "context": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 处理助手响应
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("<p style='color: grey;'>🤔 模型思考中，请稍候...</p>", 
                               unsafe_allow_html=True)
            
            try:
                # 检查是否处于对话模式（仅在价格查询模式下）
                if st.session_state.query_mode == "price_query" and st.session_state.dialogue_status in ["collecting", "ready"]:
                    handle_dialogue_step(api, prompt)
                    return
                
                # 根据用户选择的模式直接分流（去除意图识别）
                if st.session_state.query_mode == "knowledge_qa":
                    handle_knowledge_qa(api, prompt, placeholder)
                else:  # price_query
                    # 根据设置选择查询模式
                    if st.session_state.get("use_dialogue_mode", True):
                        handle_price_query_dialogue(api, prompt)
                    else:
                        handle_price_query(api, prompt, placeholder)
            
            except Exception as e:
                placeholder.error(f"服务调用失败：{e}")


def _render_dialogue_content(result: Dict[str, Any]) -> None:
    """渲染对话式查询的内容（在 chat_message 上下文中调用）"""
    # 构建响应文本
    response_parts = []
    
    channel_info = result.get("channel_info", {})
    if channel_info.get("inferred"):
        response_parts.append(f"💡 {channel_info.get('reason', '')}\n")
    
    response_parts.append(result.get("response_text", ""))
    
    entities = result.get("entities", {})
    if entities:
        response_parts.append("\n\n📋 **已收集信息**：")
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
                response_parts.append(f"\n- ✅ {label}：{value}")
    
    response_text = "\n".join(response_parts)
    st.markdown(response_text, unsafe_allow_html=True)
    
    # 渲染按钮
    render_dialogue_buttons(result)


def render_reference_docs() -> None:
    """渲染参考文档区域"""
    if not st.session_state.get("last_docs"):
        return
    
    st.markdown("---")
    st.subheader("📎 参考文档")
    
    idx = st.selectbox(
        "选择要查看的参考文档：",
        range(len(st.session_state.last_docs)),
        format_func=lambda i: f"{i+1}. {st.session_state.last_docs[i]['filename']}",
    )
    
    with st.expander("📄 文档内容", expanded=True):
        doc = st.session_state.last_docs[idx]
        if doc["retriever_name"] == "价格推荐接口":
            st.markdown(doc["context"], unsafe_allow_html=True)
        else:
            st.markdown(f"""
- **文件名**: `{doc['filename']}`  
- **相似度**: `{doc['similarity_score']}`  
- **检索方式**: `{doc['retriever_name']}`    
```
{doc['context']}
```
""")


def init_session_state() -> None:
    """初始化 session state"""
    defaults = {
        "messages": [],
        "last_docs": [],
        "file_list": [],
        "selected_files": [],
        "kb_updating": False,
        "kb_update_task_id": None,
        "upload_key": 0,
        "data_threshold": 10,
        "use_dialogue_mode": True,
        "dialogue_session_id": None,
        "dialogue_entities": {},
        "dialogue_status": None,
        "dialogue_quick_value": None,
        "dialogue_action": None,
        "current_dialogue_result": None,
        "dialogue_message_added": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """主函数"""
    # 页面配置
    st.set_page_config(
        page_title="智诚 RAG 对话助手",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化
    init_session_state()
    
    # 创建 API 客户端
    config = ApiConfig()
    api = RagApiClient(config)
    
    # 渲染侧边栏
    render_sidebar(api)
    
    # 渲染悬浮删除按钮
    render_floating_delete_button(api)
    
    # 渲染聊天界面
    render_chat_interface(api)
    
    # 渲染参考文档
    render_reference_docs()


if __name__ == "__main__":
    main()
