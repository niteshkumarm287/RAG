# RAG v3 — Threshold, citations & a proper pipeline (skeleton to complete)

> This version ships as a **guided skeleton**. The structure, types, and function
> signatures are provided; the core logic is left as `TODO`/`NotImplementedError`
> for you to implement. The goal is to upskill your Python, not just run code.

## What v3 adds over v2

| Concept | Why it matters |
|---|---|
| **Retrieval distance threshold** | Off-topic queries return "I don't know" *before* calling the LLM — saves cost and prevents hallucination. |
| **Citations with distances** | Every answer shows which chunk it came from and how close the match was, so you can *see* retrieval working. |
| **`RAGPipeline` class** | Heavy models/clients load once and are reused — the shape you'll later wrap in a web server. |
| **Config module** | All tunables (`chunk size`, `k`, `max_distance`, models) live in one place. |
| **Dataclasses** | `Chunk` / `RetrievedChunk` replace raw dicts — typed, self-documenting data. |
| **argparse CLIs** | One-shot (`--question`) and interactive modes; tune `-k` and `--max-distance` from the terminal. |

## Structure

```
v3/
├── ingest.py          # CLI: load → chunk → embed → persist   (has TODOs)
├── query.py           # CLI: load store → answer questions    (complete)
├── rag/
│   ├── config.py      # all tunables                          (complete)
│   ├── models.py      # Chunk / RetrievedChunk dataclasses    (1 TODO)
│   ├── chunking.py    # token chunker                         (complete)
│   ├── store.py       # FAISS store + THRESHOLD search        (TODOs)
│   └── pipeline.py    # RAGPipeline: retrieve + generate      (TODOs)
├── data/              # source docs
├── faiss_store/       # generated index (gitignored)
└── requirements.txt
```

## Your tasks (in recommended order)

1. **`rag/models.py`** → `Chunk.from_dict` (easiest; warm-up).
2. **`rag/store.py`** → `build`, `load`, and **`search`** (the threshold — the heart of v3).
3. **`rag/pipeline.py`** → `embed_query`, `build_context`, **`answer`**.
4. **`ingest.py`** → the file-walking + chunking loop.

Each `TODO` has step-by-step hints inline. Work top-to-bottom; run as you go.

## Setup & run

```bash
cd v3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"

python ingest.py
python query.py                       # interactive
python query.py -q "How are paragraphs separated?"
python query.py -q "good morning" --max-distance 1.2   # should say "I don't know"
```

## How to know your threshold works

Ask something **off-topic** (e.g. "good morning"). With the threshold implemented,
`search()` returns an empty list and `answer()` replies "I don't know" **without
calling the LLM**. Lower `--max-distance` to make it stricter, raise it to be more
permissive — tuning this properly is exactly what **v4 (evaluation)** will teach.
