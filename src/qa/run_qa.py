from src.retrieval.retrieve import simple_retrieve


def rag_answer(question: str):
    """
    Team-aligned RAG interface:
    Input: question (str)
    Output: Tuple[str, str] -> (answer, evidence)
    """
    retrieved_docs = simple_retrieve(question, top_k=3)

    evidence = "\n\n".join([doc.page_content[:300] for doc in retrieved_docs])
    answer = "Placeholder answer based on retrieved evidence."

    return answer, evidence


def run_qa(input_path, question):
    """
    CLI-facing wrapper:
    Input: input_path, question
    Output: dict
    """
    answer, evidence = rag_answer(question)

    return {
        "task": "qa",
        "input_file": input_path,
        "question": question,
        "answer": answer,
        "evidence": evidence
    }