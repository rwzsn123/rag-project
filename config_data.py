import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_int_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


app_password = os.getenv("RAG_APP_PASSWORD", "rag123")

md5_path = os.getenv("RAG_MD5_PATH", os.path.join(_BASE_DIR, "md5.text"))
persist_directory = os.getenv("RAG_CHROMA_DIR", os.path.join(_BASE_DIR, "chroma_db"))
chat_history_dir = os.getenv("RAG_CHAT_HISTORY_DIR", os.path.join(_BASE_DIR, "chat_history"))
collection_name = os.getenv("RAG_COLLECTION_NAME", "rag")

chunk_size = _get_int_env("RAG_CHUNK_SIZE", 800)
chunk_overlap = _get_int_env("RAG_CHUNK_OVERLAP", 150)
separators = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "，", ",", " "]
max_split = _get_int_env("RAG_MAX_SPLIT", 500)
similarity_threshold = _get_int_env("RAG_RETRIEVER_TOP_K", 5)
embedding_model_name = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-v4")
chat_model_name = os.getenv("RAG_CHAT_MODEL", "qwen3.6-max-preview")

session_config = {
    "configurable": {
        "session_id": "123456",
    }
}


def get_session_config(session_id):
    """根据 session_id 动态生成会话配置"""
    return {
        "configurable": {
            "session_id": session_id,
        }
    }
