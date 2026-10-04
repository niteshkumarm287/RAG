"""Central configuration for the RAG pipeline.

Keeping all tunables in one place is a production habit: you change behaviour
here instead of hunting through the code. Later you could load these from
environment variables or a YAML file.
"""
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent  # the v3/ folder
DATA_DIR = BASE_DIR / "data"
STORE_DIR = BASE_DIR / "faiss_store"
INDEX_FILE = STORE_DIR / "index.faiss"
METADATA_FILE = STORE_DIR / "metadata.json"

# --- Chunking ---
CHUNK_SIZE = 500        # tokens per chunk
CHUNK_OVERLAP = 50      # tokens shared between neighbouring chunks
TOKENIZER_ENCODING = "cl100k_base"

# --- Embeddings ---
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 3               # how many chunks to fetch before filtering

# The retrieval threshold. FAISS IndexFlatL2 returns an L2 *distance*:
# smaller = more similar. Any chunk whose distance is ABOVE this cutoff is
# considered irrelevant and dropped. Tune this empirically (that's what v4's
# eval harness is for). A good starting point for MiniLM + L2 is ~1.2.
MAX_DISTANCE = 1.2

# --- LLM ---
LLM_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0
