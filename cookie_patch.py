"""
cookie_patch.py - cookie / JS-challenge aware page fetcher.

Strategy (cheapest first):
  tier 0  direct GET with a per-domain session            (no warm-up, no extra request)
  tier 1  warm-up homepage visit to collect cookies, retry (only if tier 0 looked gated)
  tier 2  headless browser to run JS and collect cookies   (only if tiers 0/1 looked gated,
                                                            or on a JS-challenge 403/503)
Network errors get one retry with backoff before being reported.
"""
import atexit
import os
import re
import email.utils
import time
import threading
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from urllib import robotparser
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.exceptions import TooManyRedirects, RequestException

# The user agent is read from the CRAWLER_USER_AGENT environment variable, e.g.
#   export CRAWLER_USER_AGENT="CompanyAssistantLearning/0.1 (contact: you@email.com)"
# If it is not set, a placeholder is used.
UA = os.environ.get(
    "CRAWLER_USER_AGENT",
    "MyCrawler/1.0 (+https://yoursite.com/bot; you@email.com)",
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Text that only shows up on cookie-gate / interstitial pages.
COOKIE_GATE_PATTERNS = re.compile(
    r"enable cookies|cookies (are|must be) (required|enabled)|cookie[_-]?check|"
    r"document\.cookie\s*=|setcookie|"
    r"your browser (does not|doesn'?t) (support|accept) cookies",
    re.I,
)

# Bot-challenge pages (Cloudflare, Incapsula/Imperva, generic JS challenges).
CHALLENGE_PATTERNS = re.compile(
    r"just a moment|checking your browser|cf-browser-verification|cf-chl|"
    r"_incapsula_resource|incapsula incident|request unsuccessful|"
    r"enable javascript and cookies|attention required|ddos protection by|"
    r"please enable js",
    re.I,
)

# Pages that return HTTP 200 but are NOT real site content.
LOGIN_PATTERN = re.compile(r"<input[^>]+type\s*=\s*[\"']?password", re.I)
LOGIN_TITLE_PATTERN = re.compile(
    r"log\s?-?in\b|sign\s?-?in\b|client (portal|access)|investor (portal|access|login)|"
    r"secure (access|login)|member (login|access)|\bportal\b", re.I)
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>")
_TAG_RE = re.compile(r"<[^>]+>")
PARKED_TITLE_PATTERNS = re.compile(
    r"godaddy|domain (is )?(for sale|parked|available|expired)|buy this domain|"
    r"this domain|parked|sedo|hugedomains|afternic|namecheap|domain registration",
    re.I,
)
PARKED_BODY_PATTERNS = re.compile(
    r"this domain (is|may be) (for sale|parked|registered)|domain (is )?for sale|"
    r"buy this domain|the domain .{0,60} (has expired|is parked)|"
    r"parked (free|domain|by)|related searches|"
    r"godaddy\.com/(domainsearch|domains)|sedoparking|parkingcrew|hugedomains|afternic\.com",
    re.I,
)
PARKED_HOSTS = ("godaddy.com", "sedoparking.com", "sedo.com", "hugedomains.com", "afternic.com",
                "dan.com", "parkingcrew.net", "bodis.com", "namecheap.com", "above.com",
                "domainsponsor.com", "parklogic.com")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
CONTENT_CHECK_MAX = 60000   # parked / login pages are small; don't scan huge real pages

GATE_MAX_BODY = 6000      # gate / challenge pages are small
EMPTY_BODY = 300          # below this a page is basically empty
BROWSER_REASONS = ("cookie_gate", "redirect_loop", "js_challenge", "empty_page")


@dataclass
class FetchResult:
    url: str
    ok: bool
    status: Optional[int] = None
    html: Optional[str] = None
    method: Optional[str] = None   # "session" | "warmup" | "browser"
    reason: Optional[str] = None   # None | cookie_gate | js_challenge | empty_page |
                                   # redirect_loop | http_error | network_error | robots_disallowed
    detail: Optional[str] = None   # real exception text / extra info, for debugging
    final_url: Optional[str] = None  # URL after redirects (may be a different domain!)
    title: Optional[str] = None      # <title> of the page, handy for auditing results
    # reason can also be: login_required | parked_domain | browser_error


def page_title(html):
    m = TITLE_RE.search(html or "")
    return " ".join(m.group(1).split())[:120] if m else None


def looks_like_login_wall(html):
    """A password box alone is NOT enough: most bank homepages have an online-banking
    login widget. It is a login *wall* only if the page is basically just the login:
    the title says login/portal, or the page has very little text and very few links."""
    if not LOGIN_PATTERN.search(html):
        return False
    if LOGIN_TITLE_PATTERN.search(page_title(html) or ""):
        return True
    text = " ".join(_TAG_RE.sub(" ", _SCRIPT_STYLE_RE.sub(" ", html)).split())
    links = len(re.findall(r"<a\s", html, re.I))
    return len(text) < 1500 and links < 15


def content_problem(html, final_url=None):
    """Page loaded fine (HTTP 200, decent size) but isn't real content.
    Returns 'parked_domain' | 'login_required' | None."""
    html = html or ""
    host = (urlparse(final_url).netloc if final_url else "").lower().split(":")[0]
    if host and any(host == h or host.endswith("." + h) for h in PARKED_HOSTS):
        return "parked_domain"
    if len(html) > CONTENT_CHECK_MAX:
        return None
    title = page_title(html) or ""
    if PARKED_TITLE_PATTERNS.search(title) or PARKED_BODY_PATTERNS.search(html):
        return "parked_domain"
    if looks_like_login_wall(html):
        return "login_required"
    return None


def base_domain(host: str) -> str:
    """Naive registrable domain (last two labels) so sibling subdomains share one polite delay."""
    host = host.split(":")[0].lower()
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


class CookieAwareFetcher:
    def __init__(self, delay=2.0, timeout=(10, 20), use_browser_fallback=True,
                 max_workers=8, retries=1, retry_backoff=3.0,
                 browser_on_blocked=True, respect_robots=False,
                 rate_limit_retries=1, max_retry_after=20):
        self.delay = delay
        self.timeout = timeout
        self.use_browser_fallback = use_browser_fallback
        self.max_workers = max_workers
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.browser_on_blocked = browser_on_blocked   # also try browser on 403/503 JS-challenge pages
        self.respect_robots = respect_robots
        self.rate_limit_retries = rate_limit_retries   # extra tries after an HTTP 429
        self.max_retry_after = max_retry_after         # never wait longer than this (seconds)

        self._sessions = {}
        self._last_hit = {}
        self._domain_locks = {}
        self._robots = {}
        self._global_lock = threading.Lock()

        # Playwright's sync API must stay on the thread that created it, so every
        # browser call is routed through this single dedicated thread and the
        # browser is launched once and reused.
        self._browser_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pw")
        self._pw = None
        self._browser = None
        self._closed = False
        atexit.register(self.close)

    # ---------- helpers ----------
    def _host(self, url):
        return urlparse(url).netloc

    def _lock_for(self, host):
        key = base_domain(host)
        with self._global_lock:
            return self._domain_locks.setdefault(key, threading.RLock())

    def _session_for(self, url):
        h = self._host(url)
        with self._global_lock:
            if h not in self._sessions:
                s = requests.Session()
                s.headers.update(HEADERS)
                s.max_redirects = 10
                self._sessions[h] = s
            return self._sessions[h]

    def _wait_politely(self, host):
        key = base_domain(host)
        with self._global_lock:
            elapsed = time.time() - self._last_hit.get(key, 0)
            wait = max(0.0, self.delay - elapsed)
            # reserve the slot now so concurrent callers queue up behind us
            self._last_hit[key] = time.time() + wait
        if wait:
            time.sleep(wait)

    def _robots_allows(self, url):
        if not self.respect_robots:
            return True
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        with self._global_lock:
            rp = self._robots.get(origin)
        if rp is None:
            rp = robotparser.RobotFileParser()
            try:
                r = requests.get(origin + "/robots.txt", headers=HEADERS, timeout=self.timeout)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except RequestException:
                rp.parse([])          # unreachable robots.txt -> allow
            with self._global_lock:
                self._robots[origin] = rp
        return rp.can_fetch(UA, url)

    # ---------- classification ----------
    def classify(self, resp):
        """Return None if the response looks like a real page, else a reason string."""
        body = resp.text or ""
        small = len(body) < GATE_MAX_BODY
        if resp.status_code >= 400:
            if small and CHALLENGE_PATTERNS.search(body):
                return "js_challenge"
            return "http_error"
        if small and CHALLENGE_PATTERNS.search(body):
            return "js_challenge"
        if small and COOKIE_GATE_PATTERNS.search(body):
            return "cookie_gate"
        if len(body) < EMPTY_BODY:
            return "empty_page"
        return content_problem(body, getattr(resp, "url", None))

    # kept for backward compatibility with the old test script
    def _looks_like_cookie_gate(self, resp):
        return self.classify(resp) in ("cookie_gate", "js_challenge", "empty_page")

    # ---------- low level GET with retry ----------
    def _retry_after(self, resp):
        """Seconds to wait after a 429, from the Retry-After header (seconds or HTTP date)."""
        val = (resp.headers.get("Retry-After") or "").strip()
        wait = 5.0
        if val.isdigit():
            wait = float(val)
        elif val:
            try:
                dt = email.utils.parsedate_to_datetime(val)
                wait = max(0.0, dt.timestamp() - time.time())
            except (TypeError, ValueError):
                pass
        return min(max(wait, 1.0), self.max_retry_after)

    def _get(self, s, url, host):
        last_exc = None
        attempt = 0
        rate_limited = 0
        while attempt <= self.retries:
            self._wait_politely(host)
            try:
                r = s.get(url, timeout=self.timeout, allow_redirects=True)
            except TooManyRedirects as e:
                return None, ("redirect_loop", str(e))
            except RequestException as e:
                last_exc = e
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                attempt += 1
                continue
            if r.status_code == 429 and rate_limited < self.rate_limit_retries:
                rate_limited += 1
                time.sleep(self._retry_after(r))     # "slow down" -> wait, then try again
                continue
            return r, None
        return None, ("network_error", f"{type(last_exc).__name__}: {last_exc}")

    # ---------- tier 0 / 1: session ----------
    def _fetch_with_session(self, url, warm_up):
        host = self._host(url)
        s = self._session_for(url)

        if warm_up and not s.cookies:
            root = f"{urlparse(url).scheme}://{host}/"
            # a failed warm-up must never fail the real request
            try:
                self._wait_politely(host)
                s.get(root, timeout=self.timeout, allow_redirects=True)
            except RequestException:
                pass

        r, err = self._get(s, url, host)
        if err:
            return FetchResult(url, False, reason=err[0], detail=err[1])

        reason = self.classify(r)
        title = page_title(r.text)
        if reason is None:
            return FetchResult(url, True, r.status_code, r.text,
                               method="warmup" if warm_up else "session",
                               final_url=r.url, title=title)
        return FetchResult(url, False, r.status_code, reason=reason,
                           detail=f"status={r.status_code} body_len={len(r.text or '')}",
                           final_url=r.url, title=title)

    # ---------- tier 2: browser ----------
    def _ensure_browser(self):            # runs on the playwright thread only
        if self._browser is None:
            from playwright.sync_api import sync_playwright
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch()
        return self._browser

    def _browser_job(self, url):          # runs on the playwright thread only
        host = self._host(url)
        root = f"{urlparse(url).scheme}://{host}/"
        ctx = self._ensure_browser().new_context(user_agent=UA)
        try:
            page = ctx.new_page()
            try:
                page.goto(root, timeout=30000)
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass                       # home page failure must never block the target URL
            resp = page.goto(url, timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            html = page.content()
            status = resp.status if resp else None
            cookies = ctx.cookies()
            return html, status, cookies, page.url
        finally:
            ctx.close()

    def _fetch_with_browser(self, url):
        try:
            html, status, cookies, final_url = self._browser_thread.submit(self._browser_job, url).result(timeout=120)
        except Exception as e:
            # report the *real* problem instead of pretending it was a cookie gate
            return FetchResult(url, False, reason="browser_error", detail=f"{type(e).__name__}: {e}")

        s = self._session_for(url)
        for c in cookies:
            s.cookies.set(c["name"], c["value"], domain=c["domain"], path=c["path"])

        title = page_title(html)
        extra = dict(method="browser", final_url=final_url, title=title)
        if status and status >= 400:
            return FetchResult(url, False, status, reason="http_error",
                               detail="browser also got an HTTP error", **extra)
        if CHALLENGE_PATTERNS.search(html) and len(html) < GATE_MAX_BODY:
            return FetchResult(url, False, status, reason="js_challenge", **extra)
        if len(html) < EMPTY_BODY:
            return FetchResult(url, False, status, reason="empty_page", **extra)
        problem = content_problem(html, final_url)
        if problem:
            return FetchResult(url, False, status, reason=problem,
                               detail=f"landed on {final_url}", **extra)
        return FetchResult(url, True, status or 200, html, **extra)

    # ---------- public API ----------
    def fetch(self, url):
        host = self._host(url)
        with self._lock_for(host):
            if not self._robots_allows(url):
                return FetchResult(url, False, reason="robots_disallowed")

            result = self._fetch_with_session(url, warm_up=False)        # tier 0
            if result.ok:
                return result

            gated = result.reason in ("cookie_gate", "empty_page", "redirect_loop", "js_challenge")
            if gated:
                result = self._fetch_with_session(url, warm_up=True)     # tier 1
                if result.ok:
                    return result

            if not self.use_browser_fallback:
                return result

            want_browser = result.reason in BROWSER_REASONS
            if (not want_browser and self.browser_on_blocked and result.reason == "http_error"
                    and result.status in (403, 503)):
                want_browser = True                                      # JS challenge w/o recognisable body
            if want_browser:
                return self._fetch_with_browser(url)                     # tier 2
            return result

    def fetch_many(self, urls):
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(self.fetch, urls))

    def close(self):
        if self._closed:
            return
        self._closed = True

        def _shutdown():
            try:
                if self._browser:
                    self._browser.close()
                if self._pw:
                    self._pw.stop()
            except Exception:
                pass
        try:
            self._browser_thread.submit(_shutdown).result(timeout=20)
        except Exception:
            pass
        self._browser_thread.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()