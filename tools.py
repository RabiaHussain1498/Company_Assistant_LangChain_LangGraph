"""Tools the assistant can use. A tool is a normal Python function the LLM is allowed to call."""
import re

from langchain_core.tools import tool
from cookie_patch import CookieAwareFetcher

_fetcher = None


def _get_fetcher():
    """One shared fetcher for the whole program (it keeps cookies per website)."""
    global _fetcher
    if _fetcher is None:
        # Browser fallback is off for now, so Playwright is not required yet.
        _fetcher = CookieAwareFetcher(delay=1.0, max_workers=2, use_browser_fallback=False)
    return _fetcher


def _visible_text(html, limit=3000):
    """Strip scripts, styles and tags so the LLM only seep readable text."""
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split())[:limit]


@tool
def fetch_page(url: str) -> dict:
    """Download a website page and return its readable text.

    Use this when the user wants information about a company or website and a URL is known.
    Returns ok (True or False), final_url (where the page really ended up), title and text
    (the first 3000 characters of readable text). If ok is False, reason says why
    (for example login_required, parked_domain, http_error or network_error).
    """
    result = _get_fetcher().fetch(url)
    if not result.ok:
        return {
            "ok": False,
            "reason": result.reason,
            "detail": result.detail,
            "final_url": result.final_url,
        }
    return {
        "ok": True,
        "final_url": result.final_url,
        "title": result.title,
        "text": _visible_text(result.html),
    }