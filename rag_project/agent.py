"""LangGraph Agent 客服系统 — 用 ReAct Agent 替代固定 RAG Chain"""
from datetime import datetime
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage
from rag_project import config
from rag_project.file_history import FileChatMessageHistory
from rag_project.tools import search_knowledge, search_web
from rag_project import tools as tools_module

def build_system_prompt():
    """动态生成 System Prompt，注入当前日期时间"""
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M（%A）")
    return (
        f"当前时间：{now}\n\n"
        "你是一个智能客服助手。你有两个工具可以使用：\n"
        "1. search_knowledge：检索本地知识库中的参考资料\n"
        "2. search_web：联网搜索互联网上的最新信息\n\n"
        "请遵循以下规则：\n"
        "1. 当用户提出与公司产品、业务相关的专业问题时，优先调用 search_knowledge 检索知识库。\n"
        "2. 当用户询问实时信息（天气、新闻、最新动态）或知识库中没有相关内容时，调用 search_web 联网搜索。\n"
        "3. 如果 search_knowledge 返回的结果不够相关，可以再尝试 search_web 补充信息。\n"
        "4. 当用户只是打招呼、闲聊时，直接回答即可，不需要调用任何工具。\n"
        "5. 如果调用了工具并获得了参考资料，请基于资料简洁专业地回答，并在末尾注明引用来源（如 [来源1]）。\n"
        '6. 【严格禁止编造】不要编造搜索结果中没有的日期、数据或事实。如果搜索结果中没有明确的日期，就说"据近期报道"而不是编造一个具体日期。\n'
        "7. 使用 search_web 后，请在回答末尾附上关键来源的原始链接，方便用户验证。格式如：🔗 参考链接：[标题](链接)\n"
    )


class AgentService:
    """LangGraph Agent 服务，提供与原 RagService 兼容的调用接口"""

    def __init__(self):
        self.chat_model = ChatTongyi(model_name=config.chat_model_name, streaming=True)
        self.tools = [search_knowledge, search_web]
        self.agent = create_react_agent(
            model=self.chat_model,
            tools=self.tools,
            prompt=build_system_prompt(),
        )

    @property
    def last_sources(self) -> dict:
        """获取最近一次检索的来源信息（来自 tools 模块）"""
        return tools_module.last_sources

    def _get_config(self, session_id: str) -> dict:
        """生成 LangGraph 运行配置"""
        return {"configurable": {"thread_id": session_id}}

    def _get_history(self, session_id: str) -> FileChatMessageHistory:
        return FileChatMessageHistory(session_id, config.chat_history_dir)

    @staticmethod
    def _is_unfinished_ai_message(message) -> bool:
        kwargs = getattr(message, "additional_kwargs", {}) or {}
        return bool(kwargs.get("job_id")) and kwargs.get("status") != "done"

    def _get_agent_messages(self, session_id: str):
        """过滤后台生成中的 AI 占位消息，避免把空回复传给模型"""
        history = self._get_history(session_id)
        return [message for message in history.messages if not self._is_unfinished_ai_message(message)]

    @staticmethod
    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        return str(content)

    def clear_history(self, session_id: str) -> None:
        """清空指定会话的持久化消息历史"""
        self._get_history(session_id).clear()

    def add_user_message(self, session_id: str, user_input: str) -> None:
        """立即保存用户消息，避免页面重跑或切换时丢失输入"""
        self._get_history(session_id).add_messages([HumanMessage(content=user_input)])

    def add_user_message_with_ai_placeholder(self, session_id: str, user_input: str, job_id: str) -> None:
        """保存用户消息，并在其后创建可被后台任务持续更新的 AI 占位消息"""
        self._get_history(session_id).add_messages([
            HumanMessage(content=user_input),
            AIMessage(content="", additional_kwargs={"job_id": job_id, "status": "queued"}),
        ])

    def update_ai_message(self, session_id: str, job_id: str, content=None, status=None, error=None) -> None:
        self._get_history(session_id).update_ai_message(job_id, content=content, status=status, error=error)

    def refresh_knowledge_base(self) -> None:
        """知识库更新后刷新工具层的向量检索服务"""
        tools_module.refresh_vector_service()

    def invoke(self, user_input: str, session_id: str, persist_user: bool = True) -> str:
        """同步调用 Agent，返回最终回答文本"""
        tools_module.last_sources = {}
        history = self._get_history(session_id)
        human_message = HumanMessage(content=user_input)
        messages = self._get_agent_messages(session_id)
        if persist_user:
            messages.append(human_message)
            history.add_messages([human_message])
        result = self.agent.invoke(
            {"messages": messages},
            config=self._get_config(session_id),
        )
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        answer = self._content_to_text(ai_messages[-1].content) if ai_messages else ""
        if answer:
            history.add_messages([AIMessage(content=answer)])
        return answer

    def stream(self, user_input: str, session_id: str, persist_user: bool = True, persist_ai: bool = True):
        """流式调用 Agent，yield 文本 token 片段（供 Streamlit write_stream 使用）"""
        tools_module.last_sources = {}
        history = self._get_history(session_id)
        human_message = HumanMessage(content=user_input)
        messages = self._get_agent_messages(session_id)
        if persist_user:
            messages.append(human_message)
            history.add_messages([human_message])

        streamed_text = ""
        for chunk, metadata in self.agent.stream(
            {"messages": messages},
            config=self._get_config(session_id),
            stream_mode="messages",
        ):
            if metadata.get("langgraph_node") != "agent":
                continue
            if getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None):
                continue

            text = self._content_to_text(getattr(chunk, "content", ""))
            if not text:
                continue

            if text.startswith(streamed_text):
                delta = text[len(streamed_text):]
                streamed_text = text
            else:
                delta = text
                streamed_text += text

            if delta:
                yield delta

        if streamed_text and persist_ai:
            history.add_messages([AIMessage(content=streamed_text)])


if __name__ == "__main__":
    service = AgentService()
    # 测试直接回答
    print("=== 测试闲聊 ===")
    res = service.invoke("你好", "test-session")
    print(res)
    print()
    # 测试知识库检索
    print("=== 测试知识库问题 ===")
    res = service.invoke("outlook邮箱没法用", "test-session")
    print(res)
    print("来源:", service.last_sources)
