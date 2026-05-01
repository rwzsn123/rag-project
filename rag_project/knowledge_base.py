import os
import hashlib
from langchain_chroma import Chroma
from rag_project import config
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def _ensure_md5_file():
    os.makedirs(os.path.dirname(config.md5_path), exist_ok=True)
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()


def check_md5(md5_str):
 #检查传入的md5字符串是否被处理过
    _ensure_md5_file()
    with open(config.md5_path, 'r', encoding='utf-8') as f:
        return any(line.strip() == md5_str for line in f)


def save_md5(md5_str):
#将传入的md5字符串,记录到文件保存
    _ensure_md5_file()
    with open(config.md5_path,'a',encoding="utf-8") as f:
        f.write(md5_str+"\n")


def remove_md5(md5_str):
    """从md5记录文件中移除指定的md5字符串"""
    if not os.path.exists(config.md5_path):
        return
    with open(config.md5_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(config.md5_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip() != md5_str:
                f.write(line)



def get_string_md5(input_str,encoding='utf-8'):
    #将传入的字符串转为md5字符串
    str_bytes =input_str.encode(encoding=encoding)
    md5_obj=hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex


class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory,exist_ok=True)
        self.chroma =Chroma(
            collection_name=config.collection_name,#向量库集合名称
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,#向量库存储路径
        )
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,#文本块的大小
            chunk_overlap=config.chunk_overlap,#文本块的重叠部分
            separators=config.separators,#分割符号列表
            length_function=len,
        )
    def upload_by_str(self,data,filename):
        md5_hex=get_string_md5(data)
        if check_md5(md5_hex):
            return"[跳过]内容已经存在知识库里"
        if len(data)>config.max_split:
            knowledge_chunks=self.spliter.split_text(data)
        else: 
            knowledge_chunks = [data]  
        metadata={
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"system",
            "md5":md5_hex,
        }
        self.chroma.add_texts(  #内容加载到向量库
             knowledge_chunks,
             metadatas=[metadata for _ in knowledge_chunks],
        )
        save_md5(md5_hex)
        return "[成功]内容已经成功加载到知识库"

    def delete_by_source(self, source_name):
        """按来源文件名删除文档，同时清理MD5记录"""
        data = self.chroma.get(where={"source": source_name})
        if data["ids"]:
            # 清理对应的MD5记录
            md5_set = set()
            for meta in data["metadatas"]:
                md5_val = meta.get("md5")
                if md5_val:
                    md5_set.add(md5_val)
            for md5_val in md5_set:
                remove_md5(md5_val)
            self.chroma.delete(ids=data["ids"])
            return f"[成功] 已删除来源为 '{source_name}' 的 {len(data['ids'])} 条文档"
        return f"[跳过] 未找到来源为 '{source_name}' 的文档"

    def delete_by_ids(self, ids):
        """按 ID 列表删除文档，同时清理MD5记录"""
        data = self.chroma.get(ids=ids)
        md5_set = set()
        for meta in data["metadatas"]:
            md5_val = meta.get("md5")
            if md5_val:
                md5_set.add(md5_val)
        for md5_val in md5_set:
            remove_md5(md5_val)
        self.chroma.delete(ids=ids)
        return f"[成功] 已删除 {len(ids)} 条文档"

    def delete_all(self):
        """清空整个知识库，同时清空MD5记录"""
        all_ids = self.chroma.get()["ids"]
        if all_ids:
            self.chroma.delete(ids=all_ids)
            # 清空md5记录文件
            with open(config.md5_path, 'w', encoding='utf-8') as f:
                f.write('')
            return f"[成功] 已清空知识库，共删除 {len(all_ids)} 条文档"
        return "[跳过] 知识库已为空"

    def update_document(self, doc_id, new_text=None, new_metadata=None):
        """更新指定 ID 的文档内容或元数据"""
        kwargs = {"ids": [doc_id]}
        if new_text:
            kwargs["documents"] = [new_text]
        if new_metadata:
            kwargs["metadatas"] = [new_metadata]
        self.chroma._collection.update(**kwargs)
        return f"[成功] 已更新文档 {doc_id}"
if __name__ == '__main__':
    kb=KnowledgeBaseService()
    r=kb.upload_by_str("测试","测试.txt")
    print(r)


knowledgeBaseService = KnowledgeBaseService
