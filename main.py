from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai

# Preparing a document

documents = [
    "Our refund policy: 30 days, full refund with receipt.",
    "Shipping takes 3-5 business days for domestic orders.",
    "We accept Visa, Mastercard, and PayPal.",
    "Customer support: support@example.com or call 1-800-HELP"    
]

# create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Build FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Retrieval function
def retrieve(query, k=2):
    query_embedding = model.encode([query])
    distance, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Rag function
def rag_query(question):
    context = retrieve(question)

    prompt = f"""Answer the qustion based only on this context:

Context:
{chr(10).join(context)}

Question: {question}

Answer:"""

    # Generate response

    client = genai.Client()

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )

    return response.text

# Test it
print(rag_query("How long does shipping take?"))
# Output: "Shipping takes 3-5 business days for domestic orders."