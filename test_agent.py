"""测试 Agent 联网搜索"""
from agent import AgentService

svc = AgentService()
result = svc.agent.invoke(
    {"messages": [("user", "今天有什么科技新闻")]},
    config={"configurable": {"thread_id": "debug-3"}}
)
for m in result["messages"]:
    print(f"=== {m.type} ===")
    if hasattr(m, "tool_calls") and m.tool_calls:
        names = [tc["name"] for tc in m.tool_calls]
        print(f"TOOL_CALLS: {names}")
    if m.type == "tool":
        print(f"TOOL_RESULT (first 300): {str(m.content)[:300]}")
    elif m.content:
        print(f"CONTENT (first 300): {str(m.content)[:300]}")
    print()
