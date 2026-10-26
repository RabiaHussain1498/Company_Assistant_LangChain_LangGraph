"""
store.py - save companies and find them again by meaning (Chroma + Ollama embeddings).

How it works, in one paragraph:
  An EMBEDDING MODEL turns a text into a list of numbers (a vector) so that texts with
  similar meaning get similar vectors. Chroma stores one vector per company on disk.
  To answer a question we embed the question the same way and ask Chroma for the
  closest company vectors. That lookup is the "R" (retrieval) in RAG.
"""
import os
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DB_DIR = os.environ.get("CHROMA_DIR", "chroma_db")   # folder created next to your code


def _store() -> Chroma:
    return Chroma(
        collection_name="companies",
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=DB_DIR,
    )


def _slug(name: str) -> str:
    """'Stripe, Inc.' -> 'stripe-inc'. Used as the id, so saving the same company
    twice UPDATES it instead of creating a duplicate."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def save_company(name: str, url: str, summary: str) -> None:
    doc = Document(
        page_content=f"{name}: {summary}",   # this text is what gets embedded
        metadata={"name": name, "url": url},  # extra info stored next to it
    )
    _store().add_documents([doc], ids=[_slug(name)])


def search_companies(question: str, k: int = 3) -> list[tuple[Document, float]]:
    """Return up to k (document, distance) pairs. Smaller distance = closer in meaning."""
    return _store().similarity_search_with_score(question, k=k)


def is_saved(name: str) -> bool:
    """True if a company with this name (same id) is already in the database."""
    return bool(_store().get(ids=[_slug(name)])["ids"])


def count_companies() -> int:
    return len(_store().get()["ids"])