"""Step 2: turn the fetcher into a LangChain tool and let the model use it.

Run:  python step2_tool.py
"""
from langchain.agents import create_agent
from llm import get_llm
from tools import fetch_page

# ---- Part A: call the tool ourselves, like a normal function ----
print("== Part A: calling the tool directly ==")
for url in ["https://example.com", "https://adagecapital.com", "https://4lnb.com"]:
    out = fetch_page.invoke({"url": url})
    print(url)
    print("   ", {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v) for k, v in out.items()})

# ---- Part B: what the model is told about the tool ----
print("\n== Part B: what the LLM sees ==")
print("name:       ", fetch_page.name)
print("arguments:  ", fetch_page.args)
print("description:", fetch_page.description)

# ---- Part C: the model decides to call the tool (nothing is executed yet) ----
print("\n== Part C: the model chooses the tool ==")
llm = get_llm()
llm_with_tools = llm.bind_tools([fetch_page])
reply = llm_with_tools.invoke("Get the information from https://example.com")
print("tool_calls:", reply.tool_calls)

# ---- Part D: an agent runs the whole loop (ask -> call tool -> read result -> answer) ----
print("\n== Part D: agent ==")
agent = create_agent(
    llm,
    tools=[fetch_page],
    system_prompt="You are a company research assistant. Use fetch_page to read a website, "
                  "then answer in two sentences. If the page cannot be fetched, say why.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "What is https://example.com about?"}]})
print(result["messages"][-1].content)