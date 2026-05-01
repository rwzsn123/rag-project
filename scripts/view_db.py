"""查看 ChromaDB 知识库中的所有文档"""
try:
    from scripts._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from rag_project import config

db = Chroma(
    collection_name=config.collection_name,
    embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
    persist_directory=config.persist_directory,
)

data = db.get()
print(f"=== 知识库内容 ===")
print(f"总文档数: {len(data['ids'])}\n")

for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
    source = meta.get("source", "未知")
    time = meta.get("create_time", "未知")
    preview = doc[:100].replace("\n", " ")
    print(f"[{i+1}] 来源: {source}")
    print(f"    时间: {time}")
    print(f"    内容: {preview}...")
    print()
