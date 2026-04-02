"""
前端集成示例代码
================

本文件展示如何在前端（Streamlit）中实现：
1. 上传文件不阻塞界面
2. 轮询任务状态
3. 检查知识库重建状态并提示用户
4. 在重建期间禁用知识问答功能，允许价格推荐

关键 API 端点：
- POST /api/v1/upload - 上传文件
- GET /api/v1/task_status?task_id=xxx - 查询任务状态
- GET /api/v1/service/status - 查询服务状态（新增）
- POST /api/v1/query/stream - 知识问答（重建期间会返回错误）
- POST /api/v1/query/price - 价格推荐（始终可用）
"""

import streamlit as st
import requests
import time

API_BASE_URL = "http://localhost:8001"


def upload_files_non_blocking(files):
    """
    上传文件 - 非阻塞版本
    上传后立即返回，不等待知识库重建完成
    """
    response = requests.post(
        f"{API_BASE_URL}/api/v1/upload",
        files=[("files", (f.name, f, f.type)) for f in files]
    )
    return response.json()


def check_service_status():
    """
    检查服务状态
    返回知识库是否正在重建等信息
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/service/status", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def check_task_status(task_id):
    """检查后台任务状态"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/task_status?task_id={task_id}",
            timeout=5
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ==================== Streamlit 页面示例 ====================

def render_sidebar():
    """侧边栏：文件上传和知识库状态"""
    with st.sidebar:
        st.header("📁 文件管理")
        
        # 文件上传区域
        uploaded_files = st.file_uploader(
            "上传文件",
            accept_multiple_files=True,
            type=["txt", "pdf", "md"]
        )
        
        if uploaded_files and st.button("开始上传"):
            # 使用轻量提示代替 spinner
            upload_status = st.empty()
            upload_status.info("📤 正在上传文件...")
            
            # 上传文件（非阻塞）
            result = upload_files_non_blocking(uploaded_files)
            
            if "task_id" in result:
                task_id = result["task_id"]
                st.session_state["current_task_id"] = task_id
                st.session_state["upload_results"] = result.get("results", [])
                upload_status.success(f"✅ 上传成功！任务ID: {task_id[:8]}...")
            else:
                upload_status.error("❌ 上传失败")
        
        # 知识库更新状态显示
        if "current_task_id" in st.session_state:
            st.divider()
            st.subheader("🔄 知识库状态")
            
            # 使用 st.status 组件折叠显示
            with st.status("正在检查知识库状态...", expanded=False) as status:
                service_status = check_service_status()
                
                if service_status.get("is_rebuilding"):
                    progress = service_status.get("rebuild_progress", {})
                    status.update(
                        label=f"🔄 知识库更新中 - {progress.get('message', '')}",
                        state="running",
                        expanded=True
                    )
                    st.progress(progress.get("percent", 0) / 100)
                    st.caption(f"进度: {progress.get('stage', '')}")
                else:
                    status.update(
                        label="✅ 知识库就绪",
                        state="complete"
                    )


def render_chat_interface():
    """主界面：聊天对话"""
    st.title("💬 Easy-RAG 智能问答")
    
    # 检查服务状态
    service_status = check_service_status()
    is_rebuilding = service_status.get("is_rebuilding", False)
    
    # 显示知识库状态警告
    if is_rebuilding:
        progress = service_status.get("rebuild_progress", {})
        st.warning(
            f"⚠️ **知识库正在更新中** ({progress.get('message', '')})\n\n"
            f"知识问答功能暂不可用，但您仍然可以使用 **价格推荐** 功能。",
            icon="🔄"
        )
    
    # 功能选择
    col1, col2 = st.columns(2)
    
    with col1:
        # 知识问答按钮（重建期间禁用）
        qa_disabled = is_rebuilding
        if st.button(
            "📚 知识问答",
            disabled=qa_disabled,
            help="基于上传文档回答问题" + ("（知识库更新中，暂不可用）" if qa_disabled else "")
        ):
            st.session_state["chat_mode"] = "knowledge"
    
    with col2:
        # 价格推荐按钮（始终可用）
        if st.button(
            "💰 价格推荐",
            help="查询材料价格信息（始终可用）"
        ):
            st.session_state["chat_mode"] = "price"
    
    # 显示当前模式
    chat_mode = st.session_state.get("chat_mode", "knowledge")
    mode_label = "📚 知识问答" if chat_mode == "knowledge" else "💰 价格推荐"
    st.caption(f"当前模式: {mode_label}")
    
    # 输入框
    user_input = st.text_input(
        "请输入您的问题",
        placeholder="例如：什么是工程造价？" if chat_mode == "knowledge" else "例如：查询铝合金门窗的价格"
    )
    
    if st.button("发送") and user_input:
        if chat_mode == "knowledge":
            handle_knowledge_qa(user_input)
        else:
            handle_price_query(user_input)


def handle_knowledge_qa(question):
    """处理知识问答"""
    # 先检查服务状态
    service_status = check_service_status()
    
    if service_status.get("is_rebuilding"):
        st.error("❌ 知识库正在更新中，请稍后再试知识问答功能。")
        st.info("💡 提示：您可以使用 **价格推荐** 功能查询材料价格。")
        return
    
    # 调用知识问答接口
    with st.spinner("正在思考..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/query/stream",
                json={"question": question, "num_docs": 5},
                stream=True,
                timeout=60
            )
            
            # 检查是否返回错误
            if response.status_code == 200:
                # 处理流式响应...
                answer_container = st.empty()
                full_answer = ""
                
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        if line == "[END]":
                            continue
                        full_answer += line + "\n"
                        answer_container.markdown(full_answer)
                
                st.success("✅ 回答完成")
            else:
                error_data = response.json()
                if error_data.get("error_code") == "KNOWLEDGE_BASE_UPDATING":
                    st.error("❌ 知识库正在更新中，请稍后再试。")
                else:
                    st.error(f"❌ 请求失败: {error_data.get('error_message', '未知错误')}")
                    
        except Exception as e:
            st.error(f"❌ 请求异常: {str(e)}")


def handle_price_query(question):
    """处理价格推荐（始终可用）"""
    with st.spinner("正在查询价格..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/query/price",
                json={"question": question},
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                # 处理流式响应
                answer_container = st.empty()
                full_answer = ""
                
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        if line == "[END]":
                            continue
                        full_answer += line + "\n"
                        answer_container.markdown(full_answer)
                
                st.success("✅ 查询完成")
            else:
                st.error(f"❌ 请求失败: {response.text}")
                
        except Exception as e:
            st.error(f"❌ 请求异常: {str(e)}")


def auto_refresh_service_status():
    """
    自动刷新服务状态（可选）
    可以在侧边栏显示实时状态
    """
    if "last_status_check" not in st.session_state:
        st.session_state["last_status_check"] = 0
    
    current_time = time.time()
    # 每 5 秒检查一次状态
    if current_time - st.session_state["last_status_check"] > 5:
        service_status = check_service_status()
        st.session_state["service_status"] = service_status
        st.session_state["last_status_check"] = current_time


# ==================== 完整的 Streamlit 应用 ====================

def main():
    """主应用入口"""
    st.set_page_config(
        page_title="Easy-RAG 智能问答",
        page_icon="🤖",
        layout="wide"
    )
    
    # 初始化 session state
    if "chat_mode" not in st.session_state:
        st.session_state["chat_mode"] = "knowledge"
    
    # 自动刷新状态
    auto_refresh_service_status()
    
    # 渲染界面
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()


# ==================== 简化的 JavaScript/React 示例 ====================
"""
// 如果使用 React，可以参考以下代码结构：

import React, { useState, useEffect } from 'react';

function ChatApp() {
    const [serviceStatus, setServiceStatus] = useState(null);
    const [chatMode, setChatMode] = useState('knowledge');
    
    // 定期检查服务状态
    useEffect(() => {
        const checkStatus = async () => {
            const response = await fetch('/api/v1/service/status');
            const data = await response.json();
            setServiceStatus(data);
        };
        
        checkStatus();
        const interval = setInterval(checkStatus, 5000); // 每5秒检查一次
        
        return () => clearInterval(interval);
    }, []);
    
    const isRebuilding = serviceStatus?.is_rebuilding || false;
    
    return (
        <div>
            {/* 状态提示栏 */}
            {isRebuilding && (
                <div className="alert alert-warning">
                    <span>🔄 知识库正在更新中: {serviceStatus?.rebuild_progress?.message}</span>
                    <progress value={serviceStatus?.rebuild_progress?.percent} max="100" />
                    <p>知识问答功能暂不可用，您仍可使用价格推荐功能。</p>
                </div>
            )}
            
            {/* 功能选择 */}
            <div className="mode-selector">
                <button 
                    onClick={() => setChatMode('knowledge')}
                    disabled={isRebuilding}
                    className={chatMode === 'knowledge' ? 'active' : ''}
                >
                    📚 知识问答 {isRebuilding && '(更新中)'}
                </button>
                <button 
                    onClick={() => setChatMode('price')}
                    className={chatMode === 'price' ? 'active' : ''}
                >
                    💰 价格推荐
                </button>
            </div>
            
            {/* 聊天界面 */}
            <ChatInterface mode={chatMode} />
        </div>
    );
}
"""
