"""Step 1c: test the final classifier on messages that are NOT in the prompt examples.

Run:  python step1c_holdout.py
"""
from intent import classify_intent

# (message, expected intent). None of these appear in the prompt examples.
HOLDOUT = [
    ("Can you research Blackstone for me?", "lookup_company"),
    ("I want details on Visa Inc.", "lookup_company"),
    ("Open https://www.fidelity.com and tell me what they offer", "lookup_company"),
    ("Who runs Citadel Securities?", "lookup_company"),
    ("Show me everything I saved about insurance firms", "ask_saved"),
    ("How many companies have I stored so far?", "ask_saved"),
    ("Which saved company is the largest?", "ask_saved"),
    ("Is Stripe in my saved list?", "ask_saved"),
    ("Hey there!", "chitchat"),
    ("Can you explain what a vector store is?", "chitchat"),
    ("Great, see you tomorrow", "chitchat"),
    ("What can you help me with?", "chitchat"),
]

correct = 0
for text, expected in HOLDOUT:
    got = classify_intent(text)
    ok = got.intent == expected
    correct += ok
    mark = "ok   " if ok else "WRONG"
    print(f"{mark} {text!r}\n        expected={expected}  got={got.intent}  company={got.company}  url={got.url}")

print(f"\nHoldout accuracy: {correct}/{len(HOLDOUT)}")