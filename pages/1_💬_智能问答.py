"""智能问答页面 - LangGraph Agent 版本，支持多会话"""
import streamlit as st
import time
import uuid
import agent as agent_service
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
        messages.append({"role": role, "content": content_to_text(msg.content)})
    return messages or [welcome_message()]


# ===== 初始化 =====
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
if "agent_service" not in st.session_state:
    st.session_state["agent_service"] = agent_service.AgentService()
if "message" not in st.session_state:
    st.session_state["message"] = load_session_messages(st.session_state["session_id"])
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
if st.session_state.get("knowledge_base_dirty"):
    st.session_state["agent_service"].refresh_knowledge_base()
    st.session_state["knowledge_base_dirty"] = False
if st.session_state.get("last_generation_error"):
    st.error(st.session_state.pop("last_generation_error"))


def switch_session(new_session_id):
    """切换到指定会话，并加载该会话自己的历史记录"""
    st.session_state["session_id"] = new_session_id
    st.session_state["message"] = load_session_messages(new_session_id)


def get_session_list():
    """获取所有历史会话 ID 列表"""
    return list_history_sessions()


# ===== 侧边栏 =====
with st.sidebar:
    is_generating = st.session_state.get("is_generating", False)
    st.header("⚙️ 设置")
    if is_generating:
        st.caption("Agent 正在回复，完成后可切换会话。")
    if st.button("🔄 刷新知识库", use_container_width=True, disabled=is_generating):
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        st.session_state["agent_service"].refresh_knowledge_base()
        st.success("知识库已刷新！")
        time.sleep(1)
        st.rerun()
    if st.button("🗑️ 重置对话", use_container_width=True, disabled=is_generating):
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

    if st.button("➕ 新建会话", use_container_width=True, disabled=is_generating):
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
                if st.button(label, key=f"switch_{sid}", use_container_width=True, disabled=is_current or is_generating):
                    switch_session(sid)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{sid}", use_container_width=True, disabled=is_current or is_generating):
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


st.divider()
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
    if message.get("sources"):
        with st.expander(f"📎 引用来源（{len(message['sources'])} 条）"):
            for src in message["sources"]:
                st.markdown(f"**📄 {src['source']}** &nbsp; ⏰ {src['time']}")
                st.caption(src["preview"] + "...")
                st.divider()

prompt = st.chat_input(disabled=st.session_state.get("is_generating", False))
if prompt and not st.session_state.get("is_generating", False):
    session_id = st.session_state["session_id"]
    agent_svc = st.session_state["agent_service"]
    agent_svc.add_user_message(session_id, prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    st.session_state["pending_prompt"] = prompt
    st.session_state["pending_session_id"] = session_id
    st.session_state["is_generating"] = True
    st.rerun()

pending_prompt = st.session_state.get("pending_prompt")
if pending_prompt:
    pending_session_id = st.session_state.get("pending_session_id", st.session_state["session_id"])
    with st.spinner("🤖 Agent 正在思考中..."):
        agent_svc = st.session_state["agent_service"]
        try:
            # 使用 Agent 流式输出
            res_stream = agent_svc.stream(pending_prompt, pending_session_id, persist_user=False)
            full_response = st.chat_message("assistant").write_stream(res_stream) or ""
            # 获取引用来源
            all_sources = agent_svc.last_sources
            sources = extract_cited_sources(full_response, all_sources)
            if full_response:
                st.session_state["message"].append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })
        except Exception as exc:
            st.session_state["last_generation_error"] = f"回复生成失败：{exc}"
        finally:
            st.session_state["is_generating"] = False
            st.session_state.pop("pending_prompt", None)
            st.session_state.pop("pending_session_id", None)
            st.rerun()
