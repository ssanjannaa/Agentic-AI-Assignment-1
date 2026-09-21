# Applied Agentic AI (LLM & RAG) -- Coding Assignment 1

All four required tasks, each in its own folder, sharing one LLM wrapper.

```
agentic_ai_assignment1/
├── README.md
├── requirements.txt
├── .env.example
├── common/
│   └── llm_client.py          # shared LLM wrapper (Ollama / OpenAI / Gemini / offline)
├── task1_llm_workflow/        # 1. basic LLM workflow (5 marks)
│   └── llm_workflow.py
├── task2_prompt_chaining/     # 2. summary -> key points -> questions (5 marks)
│   └── prompt_chaining.py
├── task3_agentic_ai/          # 3. plan + execute agent (5 marks)
│   └── simple_agent.py
└── task4_rag_qa/              # 4. RAG-based Q&A over PDF/TXT (5 marks)
    ├── rag_qa.py
    └── sample_docs/company_policy.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and configure ONE of: Ollama, OpenAI, or Gemini (see below)
```

### Choosing a backend (the assignment allows OpenAI / Gemini / Ollama)

`common/llm_client.py` auto-detects which backend to use, in this order:

1. **Ollama** (local, free, no API key) -- install [Ollama](https://ollama.com),
   run `ollama pull llama3`, then just run any script. No `.env` changes needed.
2. **OpenAI** -- set `OPENAI_API_KEY` in `.env`.
3. **Gemini** -- set `GEMINI_API_KEY` in `.env`.
4. **Offline fallback** -- if none of the above are available, every script
   automatically switches to a clearly labeled `[OFFLINE FALLBACK MODE ...]`
   that uses simple extractive text processing instead of a real LLM. This
   means the full pipeline (plan -> execute, retrieve -> answer, chain
   step 1 -> 2 -> 3) can still be run and inspected end-to-end for grading
   even with zero setup.

You can also force a specific backend in code: `LLMClient(backend="openai")`.

---

## Task 1 -- LLM Workflow (5 Marks)

**File:** `task1_llm_workflow/llm_workflow.py`

Accepts user input (CLI flag, interactive prompt, or a chat loop) and
returns an LLM-generated response.

```bash
python task1_llm_workflow/llm_workflow.py --query "Explain quantum computing in 2 sentences"
python task1_llm_workflow/llm_workflow.py --interactive
```

---

## Task 2 -- Prompt Chaining (5 Marks)

**File:** `task2_prompt_chaining/prompt_chaining.py`

A 3-step chained workflow where each step's output feeds the next:

1. **Summary** of the topic.
2. **Key points** extracted *from that summary*.
3. **Three questions** generated *from those key points*.

```bash
python task2_prompt_chaining/prompt_chaining.py --topic "The impact of climate change on agriculture"
```

---

## Task 3 -- Agentic AI (5 Marks)

**File:** `task3_agentic_ai/simple_agent.py`

A simple agent implementing the classic **plan -> execute -> display** loop:

1. Accepts a task in plain English.
2. **Plans**: asks the LLM to break the task into 3-5 concrete steps.
3. **Executes**: carries out each step in order, passing prior steps'
   outputs forward as context.
4. **Displays** the full plan, every intermediate step's output, and the
   final result.

```bash
python task3_agentic_ai/simple_agent.py --task "Write a short marketing email for a new eco-friendly water bottle"
```

---

## Task 4 -- RAG-Based Question Answering (5 Marks)

**File:** `task4_rag_qa/rag_qa.py`

A basic RAG pipeline over a PDF or TXT file:

1. Load and chunk the document.
2. Index chunks with TF-IDF (scikit-learn -- no external embedding API needed).
3. Retrieve the most relevant chunk(s) for a query via cosine similarity.
4. Ask the LLM to answer using only that retrieved context.

```bash
python task4_rag_qa/rag_qa.py --doc task4_rag_qa/sample_docs/company_policy.txt \
    --query "What is the refund policy after 90 days?"

# or interactively:
python task4_rag_qa/rag_qa.py --doc task4_rag_qa/sample_docs/company_policy.txt --interactive
```

Swap in your own PDF/TXT by pointing `--doc` at any file.

---

## Design notes

* **One shared LLM wrapper** (`common/llm_client.py`) avoids duplicating
  API glue code across all four tasks, and lets you switch backends
  (Ollama / OpenAI / Gemini) by only changing environment variables.
* **No hard dependency on a paid API** -- every task degrades gracefully to
  a clearly labeled offline mode so grading isn't blocked by missing
  credentials, no local Ollama install, or no internet access.
* Every script is runnable from the CLI with `--help` for the full list of
  options.
