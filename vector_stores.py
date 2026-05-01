from langchain_chroma import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
import config_data as config
class VectorStoreService(object):
    def __init__(self,embedding):
       self.embedding = embedding
       self.vector_store=Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold}) #k表示返回的向量数量
if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever=VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name)).get_retriever()
    res=retriever.invoke("无法正常使用outlook")
    print(res[1].page_content)