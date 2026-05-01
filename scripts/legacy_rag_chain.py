try:
    from scripts._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory
from rag_project.vector_store import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
from rag_project import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from rag_project.file_history import get_history, FileChatMessageHistory
class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(DashScopeEmbeddings(model=config.embedding_model_name))
        self.last_sources = []  # 保存最近一次检索的来源
        self.prompt_template=ChatPromptTemplate.from_messages(
            [
            ("system","你是一个客服助手，以我提供的已知参考资料为主，简洁并专业的回答客户的问题。如果引用了参考资料，请在回答末尾注明引用来源（文件名）。参考资料：{context}"),
            ("system","并且我提供客户的对话历史记录,如下:"),
            MessagesPlaceholder("history"),
            ("user","请回答客户的提问：{input}"),
           ]
            )
        self.chat_model=ChatTongyi(model_name=config.chat_model_name)
        self.chain=self.get_chain()

    def format_document(self,docs):
        if not docs:
            self.last_sources = {}
            return "无相关参考资料"
        # 按索引保存所有来源，供 UI 根据 LLM 实际引用过滤
        self.last_sources = {}
        formatted_str=""
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知")
            create_time = doc.metadata.get("create_time", "未知")
            preview = doc.page_content[:150].replace("\n", " ")
            self.last_sources[i+1] = {
                "source": source,
                "time": create_time,
                "preview": preview,
            }
            formatted_str+=f"[来源{i+1}: {source}] {doc.page_content}\n\n"
        return formatted_str 
    def print_prompt(self,prompt):
        print("*"*50)
        print(prompt.to_string())
        print("*"*50)
        return prompt

    def get_chain(self):
        retriever = self.vector_service.get_retriever()
        def format_for_retriever(value):
            return value["input"]
        def format_for_prompt(value):
            new_value={}
            new_value["input"]=value["input"]["input"]
            new_value["history"]=value["input"]["history"]
            new_value["context"]=value["context"]
            return new_value
        chain=(
            {
                "input":RunnablePassthrough(),   #retriver要求字符串输入
                "context":RunnableLambda(format_for_retriever)|retriever|self.format_document
            }|RunnableLambda(format_for_prompt)|self.prompt_template|self.print_prompt|self.chat_model|StrOutputParser()
        )
        conversation_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain
if __name__ == "__main__":
    #session_id配置
    session_config={
        "configurable":{
            "session_id":"123456",
        }
    }
    service=RagService()
    res=service.chain.invoke({"input":"outlook邮箱没法用"},session_config)
    print(res)
