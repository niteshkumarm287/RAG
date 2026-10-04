import os
import faiss
import json
import numpy as np
from pypdf import PdfReader
import tiktoken
from sentence_transformers import SentenceTransformer

DATA_DIR = "./data"
DB_DIR = "./faiss_store"
INDEX_FILE = os.path.join(DB_DIR, "index.faiss")
METADATA_FILE = os.path.join(DB_DIR, "metadata.json")

def token_chunker(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_text = tokenizer.decode(tokens[start_idx:end_idx]).strip()
        if chunk_text:
            chunks.append(chunk_text)

        start_idx += (chunk_size - chunk_overlap)
        if start_idx >= len(tokens) or chunk_size <= chunk_overlap:
            break

    return chunks

def load_and_extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")


    return text

def run_ingestion():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created '{DATA_DIR}' folder. Drop your PDF/MD/TXT files there and re-run.")
        return

    all_chunks = []

    metadata = []

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"📄 Processing: {file}")

            raw_text = load_and_extract_text(file_path)
            if not raw_text.strip():
                continue

            file_chunks = token_chunker(raw_text, chunk_size=500, chunk_overlap=50)

            for chunk in file_chunks:
                all_chunks.append(chunk)
                metadata.append({"source": file, "text": chunk})

    if not all_chunks:
        print("❌ No text extracted. Ingestion aborted.")
        return

    print(f"🧱 Total chunks generated: {len(all_chunks)}")

    # Vectorize
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.array(model.encode(all_chunks)).astype("float32")

    # Build FAISS database
    dimension = embeddings.shape[-1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Persist everything to disk
    os.makedirs(DB_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"💾 Success! Database saved to '{DB_DIR}' directory.")

if __name__ == "__main__":
    run_ingestion()