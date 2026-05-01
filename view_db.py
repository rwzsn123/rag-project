"""查看 ChromaDB 知识库中的所有文档"""
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config

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
