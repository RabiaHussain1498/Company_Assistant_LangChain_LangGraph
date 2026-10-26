"""
Step 5a: save a few companies and find them by meaning. No graph, no LLM chat yet.

Things to notice when you run it:
  - the question does NOT have to share words with the company text
  - the distance is smaller for the better match
  - running the file twice does not create duplicates (same id = update)
"""
from store import count_companies, save_company, search_companies

SAMPLES = [
    ("Stripe", "https://stripe.com",
     "Online payment processing platform. Businesses use its APIs to accept card payments, "
     "send payouts and manage subscriptions."),
    ("Shopify", "https://www.shopify.com",
     "E-commerce platform that lets merchants build an online store, sell products "
     "and manage orders, inventory and shipping."),
    ("Notion", "https://www.notion.so",
     "Workspace app for notes, documents, wikis and project management for teams."),
]

QUESTIONS = [
    "Which company helps me take credit card payments?",   # should match Stripe
    "I want to open an online shop",                       # should match Shopify
    "Where can my team keep documents and plan projects?", # should match Notion
]

print("Saving companies...")
for name, url, summary in SAMPLES:
    save_company(name, url, summary)
print("Companies in the database:", count_companies(), "(should be 3, even if you run this twice)")

for q in QUESTIONS:
    print("=" * 70)
    print("QUESTION:", q)
    for doc, distance in search_companies(q, k=3):
        print(f"  distance={distance:.3f}  {doc.metadata['name']:8s} {doc.metadata['url']}")