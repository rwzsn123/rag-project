"""RAG 智能客服系统 - 多页面应用入口"""
import hmac
import streamlit as st
from rag_project import config

st.set_page_config(
    page_title="RAG 智能客服系统",
    page_icon="🤖",
    layout="wide",
)

def get_app_password():
    """优先读取 Streamlit secrets，其次读取环境变量配置。"""
    try:
        return st.secrets.get("APP_PASSWORD") or config.app_password
    except Exception:
        return config.app_password


def check_password():
    """简单密码验证，未通过则阻止访问"""
    if st.session_state.get("authenticated"):
        return True
    st.title("🔐 RAG 智能客服系统")
    st.divider()
    password = st.text_input("请输入访问密码", type="password", placeholder="输入密码后按回车")
    app_password = get_app_password()
    if password:
        if hmac.compare_digest(password, app_password):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ 密码错误，请重试")
    else:
        st.info("请输入密码以访问系统")
    return False

if not check_password():
    st.stop()

# ===== 页面导航（登录后才可见）=====
qa_page = st.Page("pages/1_💬_智能问答.py", title="智能问答", icon="💬", default=True)
kb_page = st.Page("pages/2_📂_知识库管理.py", title="知识库管理", icon="📂")

pg = st.navigation([qa_page, kb_page])
pg.run()
