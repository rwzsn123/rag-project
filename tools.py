"""Agent 工具定义模块：提供知识库检索和联网搜索工具供 LangGraph Agent 调用"""
from langchain_core.tools import tool
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config

def _build_vector_service():
    return VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name))


# 全局向量检索服务实例（供工具函数和外部获取来源使用）
_vector_service = _build_vector_service()

# 保存最近一次检索的来源信息，供 UI 展示引用
last_sources: dict = {}


def refresh_vector_service():
    """知识库更新后重建向量检索服务，避免复用旧连接或缓存"""
    global _vector_service
    _vector_service = _build_vector_service()


@tool
def search_knowledge(query: str) -> str:
    """搜索知识库，根据用户问题检索相关的参考资料。
    当用户提出需要查阅知识库才能回答的专业问题时，使用此工具。
    对于简单的打招呼、闲聊或常识性问题，不需要调用此工具。

    Args:
        query: 用户的问题或搜索关键词
    """
    global last_sources
    retriever = _vector_service.get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        last_sources = {}
        return "未找到相关参考资料，请根据你的知识直接回答。"

    last_sources = {}
    formatted_str = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知")
        create_time = doc.metadata.get("create_time", "未知")
        preview = doc.page_content[:150].replace("\n", " ")
        last_sources[i + 1] = {
            "source": source,
            "time": create_time,
            "preview": preview,
        }
        formatted_str += f"[来源{i + 1}: {source}] {doc.page_content}\n\n"

    return formatted_str


@tool
def search_web(query: str) -> str:
    """联网搜索，使用搜索引擎查找互联网上的最新信息。
    当知识库中没有相关内容、用户询问实时信息（如天气、新闻、最新技术动态）、
    或知识库检索结果不够满意时，使用此工具。

    Args:
        query: 搜索关键词
    """
    from duckduckgo_search import DDGS
    global last_sources
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, backend="lite", max_results=5))
        if not results:
            last_sources = {}
            return "联网搜索未找到相关结果。"
        last_sources = {}
        formatted = "以下是联网搜索到的真实结果，请基于这些内容回答用户：\n\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            last_sources[i] = {
                "source": f"🌐 {title}",
                "time": href,
                "preview": body[:150],
            }
            formatted += f"[来源{i}] {title}\n{body}\n链接: {href}\n\n"
        return formatted
    except Exception as e:
        return f"联网搜索出错: {e}"
