# RAG 智能客服系统

基于 Streamlit、LangChain/LangGraph、ChromaDB 和通义千问模型构建的本地知识库问答应用。系统支持上传文档构建知识库，用户可以在聊天页面基于知识库内容提问；当本地知识库无法覆盖问题时，Agent 可以调用联网搜索工具补充信息。

## 功能

- 支持 TXT、MD、PDF、DOCX 文档上传
- 使用 ChromaDB 持久化本地向量知识库
- 基于 LangGraph ReAct Agent 进行工具调用
- 支持本地知识库检索和联网搜索
- 支持多会话聊天历史
- 支持知识库文件列表展示和删除
- 支持访问密码保护

## 环境要求

- Python 3.12+
- uv
- DashScope API Key

## 安装依赖

```bash
uv sync
```

## 配置

运行前需要配置 DashScope API Key：

```bash
set DASHSCOPE_API_KEY=your_api_key
```

PowerShell 推荐使用：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

可选配置：

| 变量名 | 说明 | 默认值 |
| --- | --- | --- |
| `RAG_APP_PASSWORD` | 应用访问密码 | `rag123` |
| `RAG_CHAT_MODEL` | 聊天模型名称 | `qwen3.6-max-preview` |
| `RAG_EMBEDDING_MODEL` | Embedding 模型名称 | `text-embedding-v4` |
| `RAG_RETRIEVER_TOP_K` | 检索返回文档数量 | `5` |
| `RAG_CHROMA_DIR` | 向量库目录 | `chroma_db` |
| `RAG_CHAT_HISTORY_DIR` | 会话历史目录 | `chat_history` |

也可以使用 Streamlit secrets 配置访问密码：

```toml
# .streamlit/secrets.toml
APP_PASSWORD = "your_password"
```

## 启动

```bash
uv run streamlit run app.py
```

浏览器打开 Streamlit 输出的本地地址，输入访问密码后即可使用。

## 使用方式

1. 进入“知识库管理”页面上传文档。
2. 上传成功后，系统会写入本地 ChromaDB 向量库。
3. 回到“智能问答”页面提问。
4. 如需更新知识库，可上传新文件或删除已有文件。

## 本地数据

以下内容属于本地运行数据，不会提交到 GitHub：

- `chroma_db/`：ChromaDB 向量库数据
- `chat_history/`：多会话聊天历史
- `md5.text`：已上传内容去重记录
- `__pycache__/`：Python 缓存
- `.streamlit/secrets.toml`：本地密钥配置

## 项目结构

```text
.
├── app.py                         # Streamlit 多页面入口
├── rag_project/                   # 核心业务包
│   ├── agent.py                   # LangGraph Agent 服务
│   ├── config.py                  # 项目配置
│   ├── file_history.py            # 会话历史持久化
│   ├── file_parser.py             # 上传文件解析
│   ├── generation_manager.py      # 后台生成任务管理
│   ├── knowledge_base.py          # 知识库写入、删除、更新
│   ├── tools.py                   # 知识库检索和联网搜索工具
│   └── vector_store.py            # ChromaDB 检索封装
├── pages/
│   ├── 1_💬_智能问答.py
│   └── 2_📂_知识库管理.py
├── scripts/                       # 手动调试和旧版入口脚本
│   ├── view_db.py
│   ├── test_agent_manual.py
│   └── *_legacy.py
└── pyproject.toml
```

## 注意事项

- 不要把 API Key、访问密码、聊天记录或向量库数据提交到 GitHub。
- 如果知识库更新后问答未命中新内容，可以在问答页点击“刷新知识库”。
- 联网搜索依赖当前网络环境，网络不可用时只会返回本地知识库结果。
