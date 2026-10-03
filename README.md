# RAG v1 — Minimal Retrieval-Augmented Generation

A minimal, single-file RAG (Retrieval-Augmented Generation) demo. It embeds a small
set of documents, retrieves the most relevant ones for a question using a FAISS
vector index, and generates a grounded answer with Google Gemini.

## How it works

```
question ──► embed query ──► FAISS similarity search ──► top-k documents
                                                               │
                                                               ▼
                                          build prompt (context + question)
                                                               │
                                                               ▼
                                              Gemini generate_content ──► answer
```

1. **Embed** the document corpus with `sentence-transformers` (`all-MiniLM-L6-v2`).
2. **Index** the embeddings in an in-memory FAISS L2 index.
3. **Retrieve** the top-`k` most similar documents for a query.
4. **Generate** an answer with Gemini, constrained to only the retrieved context.

## Requirements

- Python 3.10+
- A Google Gemini API key

## Setup

```bash
# from this directory
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

The Gemini client reads your API key from the environment:

```bash
export GEMINI_API_KEY="your-api-key"   # or GOOGLE_API_KEY
```

Get a key from https://aistudio.google.com/apikey

## Usage

```bash
python main.py
```

Expected output:

```
Shipping takes 3-5 business days for domestic orders.
```

## Customizing

- **Documents**: edit the `documents` list in `main.py`.
- **Retrieval depth**: change `k` in `retrieve(query, k=2)`.
- **Embedding model**: swap the `SentenceTransformer(...)` model name.
- **LLM**: change the `model='gemini-2.5-flash'` argument.

## Project structure

```
v1/
├── main.py            # the full RAG pipeline
├── requirements.txt   # runtime dependencies
├── .gitignore
└── README.md
```

## Notes

- The FAISS index is built in memory on every run; there is no persistence yet.
- This is a learning/demo version (`v1`); it is not production-hardened.
