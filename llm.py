"""Single place where the chat model is chosen.

The rest of the code never needs to know which model or provider is used.
"""
import os

from langchain.chat_models import init_chat_model

def get_llm():
    """Return a chat model based on the LLM_MODEL environment variable.

    Format: provider:model

    Examples:
        ollama:qwen2.5:3b            local and free (run `ollama pull qwen2.5:3b` first)
        openai:gpt-4o-mini           needs OPENAI_API_KEY
        anthropic:claude-sonnet-5-5  needs ANTHROPIC_API_KEY
    """
    spec = os.environ.get("LLM_MODEL", "ollama:qwen2.5:3b")
    provider, model = spec.split(":", 1)
    return init_chat_model(model, model_provider=provider, temperature=0)