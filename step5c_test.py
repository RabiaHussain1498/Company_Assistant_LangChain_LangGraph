"""
Step 5c test: the "saved" node answers questions about saved companies (full RAG).

Companies are saved directly with save_company (no fetching), into the separate
chroma_test folder. Retrieval, embeddings and the answering LLM are all REAL.
The last question is about something we did NOT save: the answer should say so.
"""
import os

os.environ["CHROMA_DIR"] = "chroma_test"      # must be set BEFORE importing store / step3_graph

import step3_graph
from store import save_company

save_company("Stripe", "https://stripe.com/",
             "Stripe is a financial infrastructure platform for businesses that offers payments, "
             "billing and subscription management. Services: Payments, Billing, Connect, Radar, Stripe Tax.")
save_company("Shopify", "https://www.shopify.com/",
             "Shopify helps businesses start, run and grow an online store and manage products, orders, "
             "inventory and shipping. Services: Online Store, Shopify POS, Shopify Payments, Shopify Capital.")

QUESTIONS = [
    ("Which of my saved companies can handle card payments?",
     "answer from saved data (Stripe; Shopify Payments is also in the saved text)"),
    ("What do we know about Shopify?",
     "company is already saved -> answer from saved data, NOT 'I need its URL'"),
    ("Which saved company offers legal advice for lawyers?",
     "no match -> says nothing saved fits"),
    ("What does Airbnb do?",
     "Airbnb is NOT saved -> asks for its URL"),
]

for q, expected in QUESTIONS:
    print("=" * 70)
    print("QUESTION:", q)
    print("EXPECT  :", expected)
    state = step3_graph.graph.invoke({"user_text": q})
    print("INTENT  :", state["intent"], "| company:", state.get("company"))
    print("ANSWER  :", state["answer"])