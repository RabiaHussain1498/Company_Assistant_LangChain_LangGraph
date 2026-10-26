"""
Step 4 test: does the retry loop behave correctly?

Real websites cannot fail on demand, so we replace the fetch_page tool inside
step3_graph with a FAKE one that fails in a way we control. The graph code is
not changed at all; only the tool it calls is swapped.

Since Step 5b the graph also extracts and saves company info after a good fetch.
This test is about retries only, so those two steps are replaced by do-nothing stubs
(no LLM call, nothing written to the database).
"""
import step3_graph


class FakeFetch:
    """Looks like a LangChain tool (has .invoke) but returns scripted results."""

    def __init__(self, results):
        self.results = list(results)   # one result per call, in order

    def invoke(self, args):
        return self.results.pop(0)


OK = {"ok": True, "final_url": "https://example.com/", "title": "Example Domain", "text": "..."}
NETWORK_ERROR = {"ok": False, "reason": "network_error", "detail": "fake timeout"}
LOGIN = {"ok": False, "reason": "login_required", "detail": "fake login wall"}

# stubs for Step 5b (not what this test is about)
step3_graph._extract_info = lambda page: step3_graph.CompanyInfo(name="Example", summary="stub", services=[])
step3_graph.save_company = lambda *args, **kwargs: None

CASES = [
    # (name, scripted results, expected attempts)
    ("fails twice, then works      -> done after 3 attempts", [NETWORK_ERROR, NETWORK_ERROR, OK], 3),
    ("login wall (permanent)       -> give up after 1 attempt", [LOGIN], 1),
    ("always fails (network error) -> give up after 3 attempts", [NETWORK_ERROR] * 3, 3),
]

passed = 0
for name, results, expected_attempts in CASES:
    print("=" * 70)
    print("CASE:", name)
    step3_graph.fetch_page = FakeFetch(results)          # swap the tool
    state = step3_graph.graph.invoke({"user_text": "Tell me about https://example.com"})
    print("ANSWER:", state["answer"])
    good = state["attempts"] == expected_attempts
    print("attempts =", state["attempts"], "| expected =", expected_attempts, "->", "PASS" if good else "FAIL")
    passed += good

print("=" * 70)
print(f"{passed}/{len(CASES)} cases passed")