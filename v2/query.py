import os
import sys
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

DB_DIR = "./faiss_store"
INDEX_FILE = os.path.join(DB_DIR, "index.faiss")
METADATA_FILE = os.path.join(DB_DIR, "metadata.json")

# =====================================================================
# CONFIGURATION & THRESHOLDS
# =====================================================================
# For L2 distance with all-MiniLM-L6-v2, 1.2 is a solid cutoff.
# Anything greater than 1.2 is considered unrelated junk.
MAX_DISTANCE = 1.2  

# =====================================================================
# 1. LOAD PERSISTED VECTOR STORE & METADATA
# =====================================================================
if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
    print("❌ Error: Vector index database not found. Please run 'ingest.py' first.")
    sys.exit(1)

print("💾 Loading index and metadata from disk...")
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load lightweight embedding model for mapping user query
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the GenAI Client
ai_client = genai.Client()

# =====================================================================
# 2. RETRIEVAL & GENERATION LOGIC WITH DISTANCE THRESHOLD
# =====================================================================
def answer_question(query: str, k: int = 2):
    # Vectorize the query phrase
    query_vector = np.array(embed_model.encode([query])).astype("float32")
    
    # Search FAISS index for top K nearest vector neighbors
    distances, indices = index.search(query_vector, k)
    
    # Gather matching source text ONLY if it passes our similarity threshold
    retrieved_contexts = []
    
    # FAISS returns a 2D array, so we must iterate over distances[0] and indices[0]
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(metadata):
            # Check if vector similarity is close enough
            if dist < MAX_DISTANCE:
                retrieved_contexts.append(metadata[idx])
            else:
                print(f"⚠️ Skipped irrelevant chunk (Distance: {dist:.4f} > Max: {MAX_DISTANCE})")
            
    # SHORT-CIRCUIT: If nothing passes the threshold, stop right here.
    if not retrieved_contexts:
        return "I'm sorry, I couldn't find any relevant documentation in my knowledge base to answer that.", []

    # Construct a clean string combining all verified matching document snippets
    context_str = "\n\n".join([f"[Source: {item['source']}]\n{item['text']}" for item in retrieved_contexts])
    
    # Hit Gemini LLM securely using system instructions
    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=query,  
        config=types.GenerateContentConfig(
            system_instruction=f"""You are a helpful assistant answering questions based strictly on the provided context. 
If the context doesn't contain the answer, say "I don't know." Do not make up answers.

Context:
{context_str}""",
            temperature=0.0  
        )
    )
    
    return response.text, retrieved_contexts

# =====================================================================
# 3. INTERACTIVE RUNTIME LOOP
# =====================================================================
if __name__ == "__main__":
    print("\n🚀 RAG system initialized with Retrieval Threshold filtering!")
    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ")
        if user_query.strip().lower() == 'exit':
            break
            
        if not user_query.strip():
            continue
            
        print("🔍 Searching database and thinking...")
        answer, sources = answer_question(user_query)
        
        print("\n💡 Response:")
        print(answer)
        
        if sources:
            print("\n📚 Sources referenced:")
            for src in sources:
                print(f" - {src['source']}")
