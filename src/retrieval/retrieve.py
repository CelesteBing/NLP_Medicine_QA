import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_chunks.pkl"


# load data
def load_chunks():
    with open(DATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    return chunks


# build index database
def build_faiss_index(chunks):
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding chunks...")
    texts = [doc.page_content for doc in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return model, index, texts, chunks


# function
def retrieve(query: str, embed_model, index, chunks, k: int = 2, min_score: float = 0.45):
    query_vec = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k + 2)
    results = []
    for rank, idx in enumerate(indices[0]):
        score = float(scores[0][rank])
        doc = chunks[idx]
        if score < min_score:
            continue
        if len(doc.page_content.strip()) < 80:
            continue
        results.append(
            {
                "rank": len(results) + 1,
                "score": score,
                "doc": doc,
            }
        )
        if len(results) == k:
            break
    return results

# test
if __name__ == "__main__":
    chunks = load_chunks()
    model, index, texts, chunks = build_faiss_index(chunks)

    query = "What is the role of mitochondria in cell death?"

    results = retrieve(query, model, index, chunks, k=3)

    print("\nTop results:\n")
    for i, r in enumerate(results):
        print(f"Result {i+1}:")
        print(r.page_content[:600])
        print("-" * 50)