"""
Step 3b: LangGraph with a CONDITIONAL edge.

New idea compared to 3a: after "classify", a small ROUTER function looks at the
state and returns the NAME of the next node. So the graph can choose a path.

                       +--> fetch     (lookup_company and we have a URL)
                       |
START -> classify -----+--> need_url  (lookup_company but no URL)
                       |
                       +--> saved     (ask_saved; real answer comes in Step 5, RAG)
                       |
                       +--> chitchat  (small talk, answered by the LLM)

Every end node writes its reply into state["answer"].
"""
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from intent import classify_intent
from llm import get_llm
from tools import fetch_page


# ---------- 1. State ----------
class AssistantState(TypedDict, total=False):
    user_text: str            # input
    intent: str               # set by classify_node
    company: Optional[str]    # set by classify_node
    url: Optional[str]        # set by classify_node
    page: dict                # set by fetch_node
    answer: str               # set by every end node


# ---------- 2. Nodes ----------
def classify_node(state: AssistantState) -> dict:
    result = classify_intent(state["user_text"])
    # classify_intent may return a Pydantic object or a dict; handle both.
    data = result if isinstance(result, dict) else result.model_dump()
    return {
        "intent": data["intent"],
        "company": data.get("company"),
        "url": data.get("url"),
    }


def fetch_node(state: AssistantState) -> dict:
    page = fetch_page.invoke({"url": state["url"]})
    if page["ok"]:
        answer = f"Fetched {page['final_url']} (title: {page['title']})."
    else:
        answer = f"Could not fetch the page: {page['reason']}."
    return {"page": page, "answer": answer}


def need_url_node(state: AssistantState) -> dict:
    company = state.get("company") or "that company"
    return {"answer": f"I understood you mean {company}, but I need its website URL. Please send the URL."}


def saved_node(state: AssistantState) -> dict:
    return {"answer": "Questions about saved companies need RAG (Step 5). Not built yet."}


def chitchat_node(state: AssistantState) -> dict:
    reply = get_llm().invoke(state["user_text"])
    return {"answer": reply.content}


# ---------- 3. Router (this is what the conditional edge calls) ----------
def route_after_classify(state: AssistantState) -> str:
    """Return the name of the next node. Plain Python, no LLM: the LLM already did its job in classify."""
    if state["intent"] == "lookup_company":
        return "fetch" if state.get("url") else "need_url"
    if state["intent"] == "ask_saved":
        return "saved"
    return "chitchat"


# ---------- 4. Build the graph ----------
builder = StateGraph(AssistantState)
builder.add_node("classify", classify_node)
builder.add_node("fetch", fetch_node)
builder.add_node("need_url", need_url_node)
builder.add_node("saved", saved_node)
builder.add_node("chitchat", chitchat_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route_after_classify,
    {   # router's return value -> node name
        "fetch": "fetch",
        "need_url": "need_url",
        "saved": "saved",
        "chitchat": "chitchat",
    },
)
for end_node in ("fetch", "need_url", "saved", "chitchat"):
    builder.add_edge(end_node, END)

graph = builder.compile()


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())

    messages = [
        "Tell me about https://example.com",                 # -> fetch
        "What does Stripe do?",                              # -> need_url
        "What do we know about the companies we saved?",     # -> saved
        "hello, how are you?",                               # -> chitchat
    ]
    for text in messages:
        print("=" * 70)
        print("INPUT :", text)
        final_state = graph.invoke({"user_text": text})
        print("INTENT:", final_state["intent"], "| URL:", final_state.get("url"))
        print("ANSWER:", final_state["answer"])