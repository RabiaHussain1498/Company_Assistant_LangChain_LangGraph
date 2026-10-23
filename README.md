# Company Assistant (LangChain + LangGraph)

A step-by-step learning project: a multimodal assistant that takes a company name or URL (text, voice, or image), fetches the website, extracts structured information, and answers questions about saved companies.

## Progress

- [x] Step 1: chain, structured output, intent classification (`intent.py`; 12/12 on the held-out test)
- [x] Step 2: website fetcher as a LangChain tool (`tools.py`, `step2_tool.py`; the model calls `fetch_page` by itself)
- [x] Step 3: LangGraph: state, nodes, conditional edges (`step3_graph.py`; 4 routes work)
- [ ] Step 4: retry loop for failed fetches
- [ ] Step 5: RAG over saved companies
- [ ] Step 6: human review (interrupt) and Gradio interface
- [ ] Later: audio (speech-to-text) and image (vision) input

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Choose a model

The model is chosen with the `LLM_MODEL` environment variable (`provider:model`):

```bash
# local and free (install Ollama first)
ollama pull qwen2.5:3b
export LLM_MODEL="ollama:qwen2.5:3b"

# or a cloud model (needs an API key in your environment, never in the code)
export LLM_MODEL="anthropic:claude-sonnet-5-5"
export ANTHROPIC_API_KEY="..."
```

## Fetcher user agent

The fetcher (`cookie_patch.py`) reads its user agent from an environment variable:

```bash
export CRAWLER_USER_AGENT="CompanyAssistantLearning/0.1 (contact: your-email)"
```

## Run

```bash
python step1_intent.py     # first chain and structured output
python step1c_holdout.py   # test the final classifier (intent.py) on unseen messages
python step2_tool.py       # fetch_page tool: direct call, tool schema, tool call, agent
```