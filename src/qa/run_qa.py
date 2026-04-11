from src.retrieval.retrieve import simple_retrieve


def run_qa(input_path, question):
    """
    Run a simple QA pipeline:
    1. Retrieve top-k relevant chunks
    2. Return retrieved evidence
    3. Use a placeholder answer for now
    """
    retrieved_docs = simple_retrieve(question, top_k=3)

    evidence_list = []
    for doc in retrieved_docs:
        evidence_list.append({
            "content": doc.page_content[:500],
            "metadata": doc.metadata
        })

    return {
        "task": "qa",
        "input_file": input_path,
        "question": question,
        "answer": "Placeholder answer based on retrieved evidence.",
        "evidence": evidence_list
    }