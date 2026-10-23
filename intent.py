"""Final intent classifier for the project (the result of step 1).

Usage:
    from intent import classify_intent
    result = classify_intent("Tell me about https://www.hl.com")
    result.intent, result.company, result.url
"""
import re

from langchain_core.prompts import ChatPromptTemplate

from llm import get_llm
from schemas import Intent

# Note: no curly braces inside this text, because {text} below is a template variable.
SYSTEM = """You classify a user's message for a company research assistant.

Choose exactly one intent:
- lookup_company: the user wants information about ONE specific company or website
  (a company name or a URL is given).
- ask_saved: the user asks a question about companies that were ALREADY saved earlier
  (words like "saved", "stored", "which of", "list", "compare").
- chitchat: greetings, thanks, or anything that is not about companies.

Rules:
- Fill company only with a real company name. Never put a category such as "banks" in company.
- Fill url only if a URL appears in the message.

Examples:
Message: What does Goldman Sachs do?
intent=lookup_company, company=Goldman Sachs, url=none
Message: Pull the information from https://www.example.com
intent=lookup_company, company=none, url=https://www.example.com
Message: Which of the saved companies are in Texas?
intent=ask_saved, company=none, url=none
Message: Thanks, that was helpful!
intent=chitchat, company=none, url=none
"""

_URL_RE = re.compile(r"https?://[^\s]+|\bwww\.[^\s]+", re.I)
_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{text}")])
_chain = None


def _get_chain():
    """Build the chain on first use, so importing this module does not load a model."""
    global _chain
    if _chain is None:
        _chain = _prompt | get_llm().with_structured_output(Intent)
    return _chain


def _clean(value):
    """Models sometimes answer with the text 'none' instead of a real null; turn that into None."""
    if value is None:
        return None
    value = value.strip()
    return None if value.lower() in ("", "none", "null", "n/a") else value


def classify_intent(text: str) -> Intent:
    """LLM decides the intent; plain code extracts any URL (code is more reliable at that)."""
    result = _get_chain().invoke({"text": text})
    result.company = _clean(result.company)
    result.url = _clean(result.url)
    match = _URL_RE.search(text)
    if match:
        result.url = match.group(0).rstrip(".,);")
        result.intent = "lookup_company"
    return result