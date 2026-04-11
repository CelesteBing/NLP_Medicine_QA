import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_chunks.pkl"


def load_chunks():
    """
    Load chunked PubMedQA documents from pickle.
    """
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_PATH}. "
            "Please run src/retrieval/prepare_pubmedqa.py first."
        )

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return chunks


def simple_retrieve(question: str, top_k: int = 3):
    """
    A simple keyword-overlap retrieval method.
    This is a placeholder retrieval system before FAISS is integrated.
    """
    chunks = load_chunks()

    query_words = set(question.lower().split())
    scored = []

    for doc in chunks:
        text = doc.page_content.lower()
        score = sum(1 for word in query_words if word in text)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_docs = [doc for score, doc in scored[:top_k] if score > 0]

    if not top_docs:
        top_docs = [doc for _, doc in scored[:top_k]]

    return top_docs