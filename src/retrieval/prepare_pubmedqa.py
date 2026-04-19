import os
import pickle
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_context(example):
    context = example.get("context", "")

    if isinstance(context, str):
        return context

    if isinstance(context, list):
        return "\n".join([str(x) for x in context])

    if isinstance(context, dict):
        parts = []
        if "contexts" in context:
            val = context["contexts"]
            if isinstance(val, list):
                parts.extend([str(v) for v in val])
            else:
                parts.append(str(val))
        return "\n".join(parts)

    return str(context)


def build_documents(max_samples=500):
    print("Loading PubMedQA...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    print("\nFirst sample from dataset:")
    print(ds[0])

    documents = []

    for i, ex in enumerate(ds):
        if i >= max_samples:
            break

        context = normalize_context(ex)
        question = ex.get("question", "")
        answer = ex.get("final_decision", ex.get("long_answer", ""))

        full_text = f"Question: {question}. Context: {context}"

        doc = Document(
            page_content=full_text,
            metadata={
                "id": i,
                "question": question,
                "answer": answer
            }
        )

        documents.append(doc)

    print(f"\nBuilt {len(documents)} documents")
    return documents


def chunk_documents(docs):
    print("Chunking documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", ". ", "\n", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    print(f"Total chunks: {len(chunks)}")
    return chunks


def save_data(docs, chunks):
    docs_path = DATA_DIR / "pubmedqa_docs.pkl"
    chunks_path = DATA_DIR / "pubmedqa_chunks.pkl"

    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print("\nSaved files:")
    print(docs_path)
    print(chunks_path)


if __name__ == "__main__":
    docs = build_documents(max_samples=500)
    chunks = chunk_documents(docs)
    save_data(docs, chunks)

    print("\nSample chunk preview:")
    print(chunks[0].page_content[:300])