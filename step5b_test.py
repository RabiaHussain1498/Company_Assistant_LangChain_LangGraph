"""
Step 5b test: fetch -> extract -> save -> search, end to end.

- The fetch tool is FAKE (two scripted pages), so the test does not depend on the internet.
- The LLM extraction and the Chroma database are REAL.
- A separate database folder (chroma_test) is used, so your real data is not touched.
  Delete it afterwards with:  rm -rf chroma_test
"""
import os

os.environ["CHROMA_DIR"] = "chroma_test"      # must be set BEFORE importing store / step3_graph

import step3_graph
from store import count_companies, search_companies


class FakeFetch:
    def __init__(self, results):
        self.results = list(results)

    def invoke(self, args):
        return self.results.pop(0)


STRIPE_PAGE = {
    "ok": True, "final_url": "https://stripe.com/", "title": "Stripe | Financial infrastructure",
    "text": ("Stripe is a financial infrastructure platform for businesses. Millions of companies "
             "use Stripe to accept payments, send payouts, automate financial processes and "
             "manage subscriptions. Products: Payments, Billing, Connect, Radar fraud prevention, "
             "Stripe Tax. Developers integrate with our APIs in minutes."),
}
SHOPIFY_PAGE = {
    "ok": True, "final_url": "https://www.shopify.com/", "title": "Shopify | Start and grow your business",
    "text": ("Shopify helps anyone start, run and grow an online business. Build your store with "
             "ready-made themes, sell on social media and marketplaces, manage products, orders, "
             "inventory and shipping from one dashboard. Products: Online Store, Shopify POS, "
             "Shopify Payments, Shopify Capital."),
}

step3_graph.fetch_page = FakeFetch([STRIPE_PAGE, SHOPIFY_PAGE])

for text in ["Tell me about https://stripe.com", "Tell me about https://www.shopify.com"]:
    print("=" * 70)
    print("INPUT :", text)
    state = step3_graph.graph.invoke({"user_text": text})
    print("ANSWER:", state["answer"])

print("=" * 70)
print("Companies in chroma_test:", count_companies(), "(should be 2)")

for q in ["Who can I use to accept card payments?", "I want to build an online store"]:
    print("-" * 70)
    print("QUESTION:", q)
    for doc, distance in search_companies(q, k=2):
        print(f"  distance={distance:.3f}  {doc.metadata['name']}  {doc.metadata['url']}")