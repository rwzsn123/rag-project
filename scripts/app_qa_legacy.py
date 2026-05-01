"""智能问答独立页面 - LangGraph Agent 版本"""
import streamlit as st
import time
import uuid
try:
    from scripts._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from rag_project import agent as agent_service
from rag_project import tools as tools_module

st.title("智能客服")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

# ===== 侧边栏：重置功能 =====
with st.sidebar:
    st.header("⚙️ 设置")
    if st.button("🔄 刷新知识库", use_container_width=True):
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        st.session_state["agent_service"] = agent_service.AgentService()
        st.success("知识库已刷新！")
        time.sleep(1)
        st.rerun()
    if st.button("🗑️ 重置对话", use_container_width=True):
        st.session_state["message"] = [{"role": "assistant", "content": "你好，我是智能客服，有什么可以帮助你的吗？"}]
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        st.session_state["agent_service"] = agent_service.AgentService()
        st.success("对话已重置！")
        time.sleep(1)
        st.rerun()

st.divider()
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，我是智能客服，有什么可以帮助你的吗？"}]
if "agent_service" not in st.session_state:
    st.session_state["agent_service"] = agent_service.AgentService()
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    with st.spinner("🤖 Agent 正在思考中..."):
        session_id = st.session_state["session_id"]
        agent_svc = st.session_state["agent_service"]
        res_stream = agent_svc.stream(prompt, session_id)
        full_response = st.chat_message("assistant").write_stream(res_stream)
        st.session_state["message"].append({"role": "assistant", "content": full_response})
