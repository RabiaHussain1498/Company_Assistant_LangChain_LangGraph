"""
Company Assistant graph (Steps 3, 4, 5 and 6a).

Step 3:  state, nodes, conditional edge after "classify".
Step 4:  retry loop after "fetch" (a conditional edge that points BACKWARD).
Step 5:  "saved" node answers questions with RAG (Chroma search + LLM).
Step 6a: HUMAN REVIEW. "extract" no longer saves. It only PROPOSES a name and summary
         (plus automatic warnings). If state["review"] is True the graph PAUSES in the
         "review" node (interrupt) until a human approves, edits or rejects. Only then
         does "save" write to Chroma.

                       +--> fetch --+--> extract --+--> review --+--> save --> done
                       |     ^      |              |  (pause)    |
                       |     +------+--> retry     +-------------+--> rejected
                       |            +--> give_up   (no review: extract -> save directly)
START -> classify -----+
                       +--> need_url  (lookup_company, no URL, and the company is not saved yet)
                       +--> saved     (ask_saved, or a lookup of an already saved company)
                       +--> chitchat  (small talk, answered by the LLM)

Every end node writes its reply into state["answer"].

Two compiled versions of the SAME graph are exported:
  graph         - no memory, never pauses (state["review"] is ignored by design of the tests)
  review_graph  - has a checkpointer, so it CAN pause and be resumed (needs a thread_id)
"""
# Load settings from a .env file (LLM_MODEL, CRAWLER_USER_AGENT, LANGSMITH_*, ...).
# This must run BEFORE the project imports below, because some of them read environment
# variables when they are imported. Variables already set in the terminal win over .env.
from dotenv import load_dotenv

load_dotenv()

import re
import time
from typing import Optional, TypedDict
from urllib.parse import urlparse

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from intent import classify_intent
from llm import get_llm
from store import is_saved, save_company, search_companies
from tools import fetch_page

MAX_ATTEMPTS = 3
# Only these failures can be temporary. login_required / parked_domain / empty_page
# will look the same on a second try, so retrying them is a waste of time.
RETRYABLE_REASONS = {"network_error", "http_error"}


# ---------- 1. State and schema ----------
class AssistantState(TypedDict, total=False):
    user_text: str            # input
    review: bool              # input: True = pause for a human before saving
    intent: str               # set by classify_node
    company: Optional[str]    # set by classify_node
    url: Optional[str]        # set by classify_node
    page: dict                # set by fetch_node (overwritten on every attempt)
    attempts: int             # set by fetch_node: how many fetches we have tried
    proposed_name: str        # set by extract_node (a PROPOSAL, not saved yet)
    proposed_summary: str     # set by extract_node
    warnings: list            # set by extract_node: automatic checks that failed
    review_action: str        # set by review_node: approve | edit | reject
    saved_name: str           # set by save_node
    saved_summary: str        # set by save_node
    answer: str               # set by every end node


class CompanyInfo(BaseModel):
    """What we want the LLM to pull out of a company web page."""
    name: str = Field(description="The company's name, as written on the page.")
    summary: str = Field(description="One or two sentences: what the company does and for whom. "
                                     "Use only facts that are on the page.")
    services: list[str] = Field(description="Main products or services named on the page. "
                                            "Empty list if the page names none.")


EXTRACT_PROMPT = (
    "Read the web page text below and extract information about the company.\n"
    "Rules: use ONLY facts that appear in the text. Do not guess or add outside knowledge. "
    "Keep the summary to one or two short sentences."
)


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
    attempt = state.get("attempts", 0) + 1
    if attempt > 1:
        time.sleep(2 * (attempt - 1))          # small backoff before a retry: 2s, 4s, ...
    page = fetch_page.invoke({"url": state["url"]})
    print(f"   [fetch attempt {attempt}] ok={page['ok']} reason={page.get('reason')}")
    return {"page": page, "attempts": attempt}


def _extract_info(page: dict) -> CompanyInfo:
    """The LLM call, kept in its own function so tests can replace it."""
    llm = get_llm().with_structured_output(CompanyInfo)
    return llm.invoke(f"{EXTRACT_PROMPT}\n\nPage title: {page.get('title')}\n\nPage text:\n{page['text']}")


def _base_host(url: Optional[str]) -> str:
    """'https://www.shop.example.com/x' -> 'example.com' (naive: last two labels)."""
    host = urlparse(url or "").netloc.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return ".".join(host.split(".")[-2:])


def _check_extraction(state: AssistantState, info: CompanyInfo) -> list:
    """Cheap automatic checks (plain code). They do not decide anything; they only give the
    human a reason to look closer."""
    warnings = []
    page = state["page"]
    haystack = f"{page.get('title') or ''} {page.get('text') or ''}".lower()
    tokens = re.sub(r"[^a-z0-9 ]", "", info.name.lower()).split()
    if tokens and tokens[0] not in haystack:
        warnings.append(f"the name '{info.name}' does not appear in the page text")
    if state.get("url") and _base_host(state["url"]) != _base_host(page.get("final_url")):
        warnings.append(f"the page redirected to a different domain: "
                        f"{_base_host(state['url'])} -> {_base_host(page.get('final_url'))}")
    return warnings


def extract_node(state: AssistantState) -> dict:
    """Propose a name and summary. NOTHING is saved here."""
    page = state["page"]
    try:
        info = _extract_info(page)
    except Exception as e:                      # a small model can fail to produce valid output
        print(f"   [extract failed: {type(e).__name__}] using title + start of the text instead")
        info = CompanyInfo(name=state.get("company") or page.get("title") or "unknown",
                           summary=page["text"][:300], services=[])
    summary = info.summary
    if info.services:
        summary += " Services: " + ", ".join(info.services) + "."
    return {"proposed_name": info.name, "proposed_summary": summary,
            "warnings": _check_extraction(state, info)}


def review_node(state: AssistantState) -> dict:
    """Pause here. interrupt() hands the payload to whoever called the graph and waits.
    When the caller resumes with Command(resume=decision), interrupt() returns that decision.
    NOTE: on resume this whole node runs again from the top, so it must contain no
    side effects before the interrupt (no LLM calls, no saving)."""
    decision = interrupt({
        "question": "Save this company?",
        "name": state["proposed_name"],
        "summary": state["proposed_summary"],
        "url": state["page"]["final_url"],
        "page_title": state["page"].get("title"),
        "source_excerpt": state["page"]["text"][:500],
        "warnings": state.get("warnings", []),
    })
    # decision looks like {"action": "approve" | "edit" | "reject", "name": ..., "summary": ...}
    action = decision.get("action", "reject")
    update = {"review_action": action}
    if action == "edit":
        update["proposed_name"] = decision.get("name") or state["proposed_name"]
        update["proposed_summary"] = decision.get("summary") or state["proposed_summary"]
    return update


def save_node(state: AssistantState) -> dict:
    save_company(state["proposed_name"], state["page"]["final_url"], state["proposed_summary"])
    return {"saved_name": state["proposed_name"], "saved_summary": state["proposed_summary"]}


def done_node(state: AssistantState) -> dict:
    page = state["page"]
    return {"answer": f"Fetched {page['final_url']} after {state['attempts']} attempt(s). "
                      f"Saved as '{state['saved_name']}': {state['saved_summary']}"}


def rejected_node(state: AssistantState) -> dict:
    return {"answer": f"Not saved: the reviewer rejected '{state['proposed_name']}'."}


def give_up_node(state: AssistantState) -> dict:
    page = state["page"]
    return {"answer": f"Could not fetch the page after {state['attempts']} attempt(s). "
                      f"Reason: {page.get('reason')}."}


def need_url_node(state: AssistantState) -> dict:
    company = state.get("company") or "that company"
    return {"answer": f"I understood you mean {company}, but I need its website URL. Please send the URL."}


ANSWER_PROMPT = (
    "You answer questions about companies the user has saved.\n"
    "Use ONLY the saved companies listed below. If they do not contain the answer, say that "
    "you have no saved company that matches. Mention the company name and its URL.\n\n"
    "Saved companies:\n{context}\n\n"
    "Question: {question}"
)


def saved_node(state: AssistantState) -> dict:
    """RAG: Retrieve the closest saved companies, then let the LLM Generate an answer from them."""
    hits = search_companies(state["user_text"], k=3)          # [(Document, distance), ...]
    if not hits:
        return {"answer": "You have not saved any company yet. Send me a company URL first."}
    context = "\n".join(f"- {doc.page_content} (URL: {doc.metadata['url']})" for doc, _ in hits)
    reply = get_llm().invoke(ANSWER_PROMPT.format(context=context, question=state["user_text"]))
    return {"answer": reply.content}


def chitchat_node(state: AssistantState) -> dict:
    reply = get_llm().invoke(state["user_text"])
    return {"answer": reply.content}


# ---------- 3. Routers (what the conditional edges call) ----------
def route_after_classify(state: AssistantState) -> str:
    if state["intent"] == "lookup_company":
        if state.get("url"):
            return "fetch"
        if state.get("company") and is_saved(state["company"]):
            return "saved"          # already saved: answer from the database, no URL needed
        return "need_url"
    if state["intent"] == "ask_saved":
        return "saved"
    return "chitchat"


def route_after_fetch(state: AssistantState) -> str:
    page = state["page"]
    if page["ok"]:
        return "ok"
    if page.get("reason") in RETRYABLE_REASONS and state["attempts"] < MAX_ATTEMPTS:
        return "retry"
    return "give_up"


def route_after_extract(state: AssistantState) -> str:
    return "review" if state.get("review") else "save"


def route_after_review(state: AssistantState) -> str:
    return "rejected" if state["review_action"] == "reject" else "save"


# ---------- 4. Build the graph ----------
builder = StateGraph(AssistantState)
for name, fn in [
    ("classify", classify_node), ("fetch", fetch_node), ("extract", extract_node),
    ("review", review_node), ("save", save_node), ("done", done_node),
    ("rejected", rejected_node), ("give_up", give_up_node), ("need_url", need_url_node),
    ("saved", saved_node), ("chitchat", chitchat_node),
]:
    builder.add_node(name, fn)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify", route_after_classify,
    {"fetch": "fetch", "need_url": "need_url", "saved": "saved", "chitchat": "chitchat"},
)
builder.add_conditional_edges(
    "fetch", route_after_fetch,
    {"ok": "extract", "retry": "fetch", "give_up": "give_up"},   # "retry" points BACK to fetch
)
builder.add_conditional_edges(
    "extract", route_after_extract,
    {"review": "review", "save": "save"},
)
builder.add_conditional_edges(
    "review", route_after_review,
    {"save": "save", "rejected": "rejected"},
)
builder.add_edge("save", "done")
for end_node in ("done", "rejected", "give_up", "need_url", "saved", "chitchat"):
    builder.add_edge(end_node, END)

graph = builder.compile()                                    # simple: no memory, never pauses
review_graph = builder.compile(checkpointer=InMemorySaver())  # can pause (interrupt) and resume


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())

    messages = [
        "Tell me about https://example.com",                       # ok -> extract -> saved in Chroma
        "Tell me about https://adagecapital.com",                  # login_required -> give up at once
        "Tell me about https://this-site-does-not-exist-xyz123.com",   # network_error -> retried 3 times
    ]
    for text in messages:
        print("=" * 70)
        print("INPUT :", text)
        final_state = graph.invoke({"user_text": text})
        print("ANSWER:", final_state["answer"])