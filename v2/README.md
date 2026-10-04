# RAG v2 — Persistent RAG with chunking, PDF/Markdown ingestion

An upgrade over `v1`. Instead of a hardcoded list of sentences, v2 ingests real
documents (PDF / Markdown / text), splits them into token-based chunks, and persists
a FAISS index + metadata to disk. Querying loads that index and answers questions
with Google Gemini, grounded strictly in the retrieved context.

## Pipeline

```
ingest.py                                   query.py
─────────                                   ────────
load files (PDF/MD/TXT)                     load persisted index + metadata
      │                                           │
      ▼                                           ▼
token chunking (500 tok, 50 overlap)        embed the user question
      │                                           │
      ▼                                           ▼
embed chunks (MiniLM)                        FAISS top-k search
      │                                           │
      ▼                                           ▼
build + persist FAISS index                 build context + Gemini answer
(faiss_store/index.faiss + metadata.json)   (grounded, temperature=0)
```

## Requirements

- Python 3.10+
- A Google Gemini API key

## Setup

```bash
cd v2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
```

Get a key from https://aistudio.google.com/apikey

## Usage

1. Drop your documents into `data/` (PDF, `.md`, or `.txt`).
2. Build the index:

   ```bash
   python ingest.py
   ```

3. Ask questions interactively:

   ```bash
   python query.py
   ```

   ```
   Ask a question (or type 'exit' to quit): How are paragraphs separated?
   💡 Response: Paragraphs are separated by a blank line.
   📚 Sources referenced:
    - sample-markdown.md
   ```

## Project structure

```
v2/
├── ingest.py          # load → chunk → embed → persist FAISS index
├── query.py           # load index → retrieve → generate grounded answer
├── data/              # your source documents (input)
├── faiss_store/       # generated index + metadata (gitignored, regenerable)
├── requirements.txt
└── README.md
```

## Key concepts in this version

- **Chunking** — documents are split into ~500-token windows with 50-token overlap
  (`tiktoken` `cl100k_base`), so no single chunk is too large and context isn't lost
  at boundaries.
- **Persistence** — the index is built once by `ingest.py` and reused by `query.py`,
  instead of re-embedding on every run.
- **Grounding** — Gemini is instructed to answer only from retrieved context and to
  say "I don't know" otherwise, with `temperature=0` for deterministic answers.

## Known limitations (addressed in later versions)

- No **retrieval distance threshold** yet — all top-k chunks are passed to the LLM
  even if they are weak matches. Only the LLM (not retrieval) rejects off-topic queries.
- No incremental ingestion — every run re-embeds all files.
- No evaluation harness to measure retrieval/answer quality.
