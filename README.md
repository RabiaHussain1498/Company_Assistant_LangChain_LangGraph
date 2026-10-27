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

## Environment variables

Create a `.env` file in the project root (it is already in `.gitignore`):

```env
# --- Model ---
LLM_MODEL=ollama:qwen2.5:3b

# --- Fetcher ---
CRAWLER_USER_AGENT="CompanyAssistantLearning/0.1 (contact: your-email)"

# --- LangSmith tracing (optional but recommended) ---
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=company-assistant
```

`load_dotenv()` is called at the top of `step3_graph.py` (before any other import), so all scripts that import from it — including `app.py` — will pick up the file automatically.

## Choose a model

The model is chosen with the `LLM_MODEL` variable (`provider:model`):

```bash
# local and free (install Ollama first)
ollama pull qwen2.5:3b
# then set in .env:  LLM_MODEL=ollama:qwen2.5:3b

# or a cloud model (needs the matching API key)
# LLM_MODEL=anthropic:claude-3-5-sonnet-20241022
# ANTHROPIC_API_KEY=...
```

## LangSmith tracing

Sign up at [smith.langchain.com](https://smith.langchain.com), create a project, and copy your API key into `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=your-project-name
```

Every run of `app.py` (or any step script) will then send traces to your LangSmith dashboard automatically.

## Run

```bash
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