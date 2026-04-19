import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pubmedqa_chunks.pkl"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "rag_test_results.json"

# Config
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

OPENAI_MODEL = "gpt-3.5-turbo"

# Load chunks
def load_chunks():
    with open(DATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    return chunks

# Build embedding model + FAISS
def build_faiss_index(chunks):
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Encoding chunks...")
    texts = [doc.page_content for doc in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # cosine
    faiss.normalize_L2(embeddings)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return embed_model, index

def keyword_overlap(query: str, text: str) -> int:
    query_words = set(query.lower().replace("?", "").split())
    text_words = set(text.lower().split())
    return len(query_words & text_words)

def retrieve(query: str, embed_model, index, chunks, k: int = 2, min_score: float = 0.35):
    query_vec = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k + 3)

    results = []
    for rank, idx in enumerate(indices[0]):
        score = float(scores[0][rank])
        doc = chunks[idx]

        if score < min_score:
            continue

        if len(doc.page_content.strip()) < 80:
            continue

        if keyword_overlap(query, doc.page_content) < 1:
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

# Prompt construction
def build_prompt(question: str, retrieved_results):
    context_blocks = []
    for item in retrieved_results:
        context_blocks.append(
            f"[Context {item['rank']} | score={item['score']:.4f}]\n{item['doc'].page_content}"
        )

    joined_context = "\n\n".join(context_blocks)

    prompt = f"""You are a biomedical question-answering assistant.

You must answer ONLY based on the retrieved context below.
Do not use outside knowledge.
If the retrieved context is insufficient, respond exactly with:
"I don't know based on the provided context."

Important rules:
- Use the higher-ranked contexts first.
- Ignore lower-ranked contexts if they are less relevant or inconsistent.
- Keep the answer concise and evidence-based.

Retrieved Context:
{joined_context}

Question:
{question}

Output format:
Direct answer:
Brief explanation:
"""
    return prompt

# GPT generation
def generate_with_gpt(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your environment.")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": "You are a careful biomedical QA assistant. Use only the provided context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.output_text.strip()

# End-to-end QA
def answer_question(question: str, embed_model, index, chunks):
    retrieved_results = retrieve(question, embed_model, index, chunks, k=TOP_K)
    prompt = build_prompt(question, retrieved_results)
    answer = generate_with_gpt(prompt)

    return {
        "question": question,
        "retrieved": [
            {
                "rank": item["rank"],
                "score": item["score"],
                "page_content": item["doc"].page_content,
                "metadata": item["doc"].metadata,
            }
            for item in retrieved_results
        ],
        "prompt": prompt,
        "answer": answer,
    }

# Batch test
def run_batch_test(embed_model, index, chunks):
    test_questions = [
        "What is the role of mitochondria in cell death?",
        "How does programmed cell death occur in plants?",
        "What is the function of chloroplasts during PCD?",
        "What is the role of TUNEL assay?",
        "What is cyclosporine A used for in this study?",
        "How do mitochondria behave during programmed cell death?",
        "What happens to nuclear DNA during mitochondrial stages?",
        "What are the categories of mitochondrial dynamics?",
        "What is the role of transvacuolar strands?",
        "Does the study suggest mitochondria are important in lace plant programmed cell death?",
    ]

    all_results = []

    for i, q in enumerate(test_questions, start=1):
        print("\n" + "=" * 90)
        print(f"Test {i}: {q}")

        result = answer_question(
            question=q,
            embed_model=embed_model,
            index=index,
            chunks=chunks,
        )

        print("\nTop retrieved chunks:")
        for item in result["retrieved"]:
            print(f"\n[Rank {item['rank']}] score={item['score']:.4f}")
            print(item["page_content"][:400])

        print("\nGenerated answer:")
        print(result["answer"])

        all_results.append(result)

    return all_results

# Save results
def save_results(results: List[Dict[str, Any]]):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to: {RESULTS_PATH}")

# Main
if __name__ == "__main__":
    chunks = load_chunks()
    embed_model, index = build_faiss_index(chunks)

    results = run_batch_test(
        embed_model=embed_model,
        index=index,
        chunks=chunks,
    )

    save_results(results)