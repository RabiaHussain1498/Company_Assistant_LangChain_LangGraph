# Company Assistant (LangChain + LangGraph)

A step-by-step learning project: a multimodal assistant that takes a company name or URL (text, voice, or image), fetches the website, extracts structured information, and answers questions about saved companies.

## Progress

- [x] Step 1: chain, structured output, intent classification (`intent.py`)
- [x] Step 2: website fetcher as a LangChain tool (`tools.py`, `step2_tool.py`)
- [x] Step 3: LangGraph: state, nodes, conditional edges (`step3_graph.py`)
- [x] Step 4: retry loop for failed fetches (`step4_test.py`)
- [x] Step 5: RAG over saved companies
  - [x] 5a: Chroma + Ollama embeddings, save and search (`step5a_store.py`)
  - [x] 5b: extract company info with the LLM and save it after a fetch (`step5b_test.py`)
  - [x] 5c: `saved` node answers questions from the retrieved companies; a lookup for an already saved company is routed to it (`step5c_test.py`)
- [x] Step 6: human review (interrupt) and Gradio interface
  - [x] 6a: pause before saving, human approves / edits / rejects (`step6a_test.py`)
  - [x] 6b: Gradio interface (`app.py`)


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

# or a cloud model (needs an API key in your environment)
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
python step1b_eval.py      # compare prompt versions on a labeled test set
python step1c_holdout.py   # test the final classifier (intent.py) on unseen messages
python step2_tool.py       # fetch_page tool: direct call, tool schema, tool call, agent
python step3_graph.py      # LangGraph: routing + retry loop (real sites)
python step4_test.py       # retry loop tested with a fake fetch tool
python step5a_store.py     # save sample companies in Chroma and search by meaning
python step5b_test.py      # fetch (fake) -> extract (LLM) -> save -> search
python step5c_test.py      # ask questions about saved companies (full RAG)
python step6a_test.py      # graph pauses for a human (interrupt) and resumes
python app.py              # Gradio web interface (open http://127.0.0.1:7860)
```