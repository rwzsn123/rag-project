"""智能问答页面 - LangGraph Agent 版本，支持多会话"""
import streamlit as st
import time
import uuid
import agent as agent_service
import generation_manager
from file_history import FileChatMessageHistory, delete_history, list_history_sessions
import re


st.title("💬 智能问答")


WELCOME_TEXT = "你好，我是智能客服，有什么可以帮助你的吗？"


def welcome_message():
    return {"role": "assistant", "content": WELCOME_TEXT}


def content_to_text(content):
    if isinstance(content, str):
        return content
    return str(content)


def load_session_messages(session_id):
    """从持久化历史中加载页面消息，保持 UI 和 Agent 上下文一致"""
    history = FileChatMessageHistory(session_id)
    messages = []
    for msg in history.messages:
        if msg.type == "human":
            role = "user"
        elif msg.type == "ai":
            role = "assistant"
        else:
            continue
        kwargs = getattr(msg, "additional_kwargs", {}) or {}
        messages.append({
            "role": role,
            "content": content_to_text(msg.content),
            "job_id": kwargs.get("job_id"),
            "job_status": kwargs.get("status"),
        })
    return messages or [welcome_message()]


# ===== 初始化 =====
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
if "agent_service" not in st.session_state:
    st.session_state["agent_service"] = agent_service.AgentService()
if "message" not in st.session_state:
    st.session_state["message"] = load_session_messages(st.session_state["session_id"])
if st.session_state.get("knowledge_base_dirty"):
    st.session_state["agent_service"].refresh_knowledge_base()
    st.session_state["knowledge_base_dirty"] = False


def switch_session(new_session_id):
    """切换到指定会话，并加载该会话自己的历史记录"""
    st.session_state["session_id"] = new_session_id
    st.session_state["message"] = load_session_messages(new_session_id)


def get_session_list():
    """获取所有历史会话 ID 列表"""
    return list_history_sessions()


# ===== 侧边栏 =====
with st.sidebar:
    st.header("⚙️ 设置")
    if generation_manager.has_active_jobs():
        st.caption("有回复正在后台生成，可继续操作其他会话。")
    if st.button("🔄 刷新知识库", use_container_width=True):
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        st.session_state["agent_service"].refresh_knowledge_base()
        st.success("知识库已刷新！")
        time.sleep(1)
        st.rerun()
    if st.button("🗑️ 重置对话", use_container_width=True):
        generation_manager.cancel_session_jobs(st.session_state["session_id"])
        agent_svc = st.session_state["agent_service"]
        agent_svc.clear_history(st.session_state["session_id"])
        st.session_state["message"] = [welcome_message()]
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        st.success("对话已重置！")
        time.sleep(1)
        st.rerun()

    st.divider()
    st.header("💬 会话管理")
    st.caption(f"当前会话：`{st.session_state['session_id']}`")

    if st.button("➕ 新建会话", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state["session_id"] = new_id
        st.session_state["message"] = [welcome_message()]
        st.success(f"已创建新会话：{new_id}")
        time.sleep(1)
        st.rerun()

    # 历史会话列表
    sessions = get_session_list()
    if sessions:
        st.subheader("📋 历史会话")
        for sid in sessions:
            is_current = sid == st.session_state["session_id"]
            col1, col2 = st.columns([3, 1])
            with col1:
                label = f"● {sid}（当前）" if is_current else f"○ {sid}"
                if st.button(label, key=f"switch_{sid}", use_container_width=True, disabled=is_current):
                    switch_session(sid)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{sid}", use_container_width=True, disabled=is_current):
                    generation_manager.cancel_session_jobs(sid)
                    delete_history(sid)
                    st.success(f"已删除会话 {sid}")
                    time.sleep(1)
                    st.rerun()

# ===== 聊天主体 =====


def extract_cited_sources(response_text, all_sources):
    """从 LLM 回答中解析 [来源N] 标记，只返回实际引用的来源"""
    if not all_sources:
        return []
    cited_ids = set(int(n) for n in re.findall(r'来源(\d+)', response_text))
    if cited_ids:
        return [all_sources[i] for i in sorted(cited_ids) if i in all_sources]
    else:
        return list(all_sources.values())


def render_chat_messages():
    st.divider()
    current_messages = load_session_messages(st.session_state["session_id"])
    st.session_state["message"] = current_messages
    for message in current_messages:
        content = message["content"]
        if message.get("job_status") in {"queued", "running"}:
            content = content or "正在生成回复..."
            if message.get("job_status") == "running":
                content = f"{content}▌"
        st.chat_message(message["role"]).write(content)
        if message.get("sources"):
            with st.expander(f"📎 引用来源（{len(message['sources'])} 条）"):
                for src in message["sources"]:
                    st.markdown(f"**📄 {src['source']}** &nbsp; ⏰ {src['time']}")
                    st.caption(src["preview"] + "...")
                    st.divider()


if hasattr(st, "fragment") and generation_manager.has_active_jobs(st.session_state["session_id"]):
    render_chat_messages = st.fragment(run_every="1s")(render_chat_messages)
render_chat_messages()

prompt = st.chat_input()
if prompt:
    session_id = st.session_state["session_id"]
    agent_svc = st.session_state["agent_service"]
    generation_manager.submit_prompt(agent_svc, session_id, prompt)
    st.rerun()
