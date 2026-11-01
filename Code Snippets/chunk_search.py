
import os
import json
import numpy as np

from openai import OpenAI
from dotenv import load_dotenv


# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBED_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")
VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")


# ==========================================
# 2. Get embedding of a text
# ==========================================

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )

    return response.data[0].embedding


# ==========================================
# 3. Calculate cosine similarity
# ==========================================

def cosine_similarity(A, B):
    dot_prod = np.dot(A, B)

    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_prod / (norm_a * norm_b)


# ==========================================
# 4. Load documents from knowledge base
# ==========================================

def load_documents(folder=KB_FOLDER):
    documents = []

    for filename in sorted(os.listdir(folder)):

        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8") as f:
            documents.append(
                {
                    "source": filename,
                    "text": f.read()
                }
            )

    return documents


# ==========================================
# 5. Split documents into chunks
# One paragraph = One chunk
# ==========================================

def chunk_document(document):
    chunks = []

    chunk_id = 0

    for part in document["text"].split("\n\n"):

        part = part.strip()

        if part:
            chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "text": part
                }
            )

            chunk_id += 1

    return chunks


# ==========================================
# 6. Create chunks from all documents
# ==========================================

def create_chunks():
    documents = load_documents()

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks


# ==========================================
# 7. Add embeddings to chunks
# ==========================================

def add_embeddings(chunks):

    for chunk in chunks:

        vector = get_embedding(chunk["text"])

        chunk["vector"] = vector

    return chunks


# ==========================================
# 8. Semantic search method
# ==========================================

def search(query, chunks, top_k=3):

    query_vector = get_embedding(query)

    results = []

    for chunk in chunks:

        score = cosine_similarity(
            query_vector,
            chunk["vector"]
        )

        results.append(
            {
                "score": float(score),
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "size": len(chunk["text"])
            }
        )

    # Sort by similarity score (highest first)
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Return top K chunks
    return results[:top_k]


# ==========================================
# 9. Testing method
# ==========================================

def test(user_query, chunks, top_k=3):

    print(f"\nUser Query: {user_query}\n")

    results = search(user_query, chunks, top_k)

    for r in results:

        print(
            f" {r['score']:.2f} - "
            f"{r['source']} - "
            f"{r['size']} characters"
        )

        print(f"    {r['text']}\n")


# ==========================================
# 10. Save vector store
# ==========================================

def save_vector_store(chunks):

    with open(VECTOR_STORE_FILE, "w", encoding="utf-8") as f:

        json.dump(chunks, f, indent=4)


# ==========================================
# 11. Load vector store
# ==========================================

def load_vector_store():

    with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as f:

        return json.load(f)


# ==========================================
# 12. Start testing
# ==========================================

chunks = []

if os.path.exists(VECTOR_STORE_FILE):

    # Load previously saved embeddings
    chunks = load_vector_store()

    print(
        f"Loaded {len(chunks)} chunks from the "
        f"vector store file {VECTOR_STORE_FILE}"
    )

else:

    # Create chunks from knowledge base
    chunks = create_chunks()

    # Generate embeddings
    chunks = add_embeddings(chunks)

    # Save embeddings for future use
    save_vector_store(chunks)

    print(
        f"Created and embedded {len(chunks)} chunks "
        f"and saved to the vector store file "
        f"{VECTOR_STORE_FILE}"
    )


# ==========================================
# 13. Test semantic search
# ==========================================

user_query = "What are the benefits of Maternity Leave?"

test(user_query, chunks, 3)
