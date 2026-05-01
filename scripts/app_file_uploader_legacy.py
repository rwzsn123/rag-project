import streamlit as st
#当web页面发生变化时，代码就会重跑一遍，无法保存一些文件
#from streamlit.runtime.uploaded_file_manager import UploadedFileRec
try:
    from scripts._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from rag_project.knowledge_base import KnowledgeBaseService
from rag_project.file_parser import parse_file
from collections import Counter
import time
st.title("知识库更新服务")
uploaded_file = st.file_uploader(
    "请上传文件（支持 TXT、PDF、DOCX、MD）",
    type=['txt', 'pdf', 'docx', 'md'],
    accept_multiple_files=False,
)

if "service" not in st.session_state:
    st.session_state.service = KnowledgeBaseService()
if uploaded_file is not None:
    file_name = uploaded_file.name
    file_type=uploaded_file.type
    file_size=uploaded_file.size/1024
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式:{file_type}|大小:{file_size:.2f}KB")
    with st.spinner("正在解析并上传..."):
        try:
            text, filename = parse_file(uploaded_file)
            time.sleep(1)
            result=st.session_state.service.upload_by_str(text, filename)
            st.write(result)
        except Exception as e:
            st.error(f"文件解析失败：{e}")

# ===== 侧边栏：已上传文件列表 =====
with st.sidebar:
    st.header("📂 已上传文件")
    data = st.session_state.service.chroma.get()
    total = len(data["ids"])
    st.metric("总文档数", total)
    st.divider()

    if total > 0:
        # 按来源分组统计
        sources = [meta.get("source", "未知") for meta in data["metadatas"]]
        source_counts = Counter(sources)
        # 获取每个来源的最新上传时间
        source_times = {}
        for meta in data["metadatas"]:
            src = meta.get("source", "未知")
            t = meta.get("create_time", "未知")
            if src not in source_times or t > source_times[src]:
                source_times[src] = t

        for source, count in source_counts.items():
            with st.expander(f"📄 {source}（{count} 条）"):
                st.write(f"⏰ 上传时间：{source_times[source]}")
                if st.button(f"🗑️ 删除", key=f"del_{source}"):
                    result = st.session_state.service.delete_by_source(source)
                    st.success(result)
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("知识库为空，请上传文件。")
