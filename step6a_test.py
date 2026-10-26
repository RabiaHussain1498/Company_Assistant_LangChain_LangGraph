"""
Step 6a test: the graph PAUSES for a human, then continues after the human decides.

- fetch is FAKE (scripted pages), LLM extraction and Chroma are REAL
- a separate database folder (chroma_test_review) is used.  Delete it afterwards:
      rm -rf chroma_test_review

Each case: run until the pause -> look at what the human would see -> resume with a decision.
"""
import os

os.environ["CHROMA_DIR"] = "chroma_test_review"      # must be set BEFORE importing store / step3_graph

from langgraph.types import Command

import step3_graph
from store import is_saved, search_companies


class FakeFetch:
    def __init__(self, results):
        self.results = list(results)

    def invoke(self, args):
        return self.results.pop(0)


STRIPE = {"ok": True, "final_url": "https://stripe.com/", "title": "Stripe | Financial infrastructure",
          "text": "Stripe is a financial infrastructure platform for businesses. Millions of companies use "
                  "Stripe to accept payments, send payouts and manage subscriptions. "
                  "Products: Payments, Billing, Connect, Radar, Stripe Tax."}
SHOPIFY = {"ok": True, "final_url": "https://www.shopify.com/", "title": "Shopify | Start and grow your business",
           "text": "Shopify helps anyone start, run and grow an online business. Build your store, manage "
                   "products, orders, inventory and shipping from one dashboard. "
                   "Products: Online Store, Shopify POS, Shopify Payments."}
NOTION = {"ok": True, "final_url": "https://www.notion.so/", "title": "Notion | Your connected workspace",
          "text": "Notion is a workspace for notes, documents, wikis and project management for teams."}
# the user asked for acme-payments.com but the site redirected to a different domain
REDIRECTED = {"ok": True, "final_url": "https://cheap-deals.example.net/", "title": "BigDeals Marketplace",
              "text": "Welcome to BigDeals Marketplace. Buy cheap electronics, phones and accessories "
                      "from thousands of sellers. Free shipping on all orders."}

CASES = [
    # (label, user text, fake page, the human's decision)
    ("A: human APPROVES", "Tell me about https://stripe.com", STRIPE,
     {"action": "approve"}),
    ("B: human EDITS the summary", "Tell me about https://www.shopify.com", SHOPIFY,
     {"action": "edit", "summary": "EDITED BY HUMAN: e-commerce platform for building online stores."}),
    ("C: human REJECTS", "Tell me about https://www.notion.so", NOTION,
     {"action": "reject"}),
    ("D: redirect to another domain -> warning -> human REJECTS", "Tell me about https://acme-payments.com", REDIRECTED,
     {"action": "reject"}),
]

for i, (label, text, page, decision) in enumerate(CASES):
    print("=" * 70)
    print("CASE", label)
    step3_graph.fetch_page = FakeFetch([page])
    config = {"configurable": {"thread_id": f"case-{i}"}}      # one thread = one conversation / one pause

    step3_graph.review_graph.invoke({"user_text": text, "review": True}, config)   # runs until the pause
    snapshot = step3_graph.review_graph.get_state(config)
    print("PAUSED. Next node to run:", snapshot.next)
    shown = snapshot.tasks[0].interrupts[0].value             # what the human is shown
    print("  HUMAN SEES  name    :", shown["name"])
    print("              summary :", shown["summary"])
    print("              url     :", shown["url"])
    print("              warnings:", shown["warnings"] or "none")
    print("HUMAN DECIDES:", decision)

    final = step3_graph.review_graph.invoke(Command(resume=decision), config)      # continue from the pause
    print("ANSWER:", final["answer"])

print("=" * 70)
print("Check the database:")
for name in ["Stripe", "Shopify", "Notion", "BigDeals Marketplace"]:
    print(f"  {name:22s} saved? {is_saved(name)}")
doc, _ = search_companies("online store", k=1)[0]
print("  Shopify text in the database:", doc.page_content)
print("Expected: Stripe True, Shopify True (with the EDITED text), Notion False, BigDeals False")