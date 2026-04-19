from src.retrieval.rag_pipeline import (
    load_chunks,
    build_faiss_index,
    answer_question,
)


_chunks = load_chunks()
_embed_model, _index = build_faiss_index(_chunks)


def rag_answer(question: str):
    """
    Team-aligned RAG interface
    Input: question (str)
    Output: Tuple[str, str] -> (answer, evidence)
    """
    result = answer_question(
        question=question,
        embed_model=_embed_model,
        index=_index,
        chunks=_chunks,
    )

    answer = result["answer"]

    evidence = "\n\n".join(
        [
            f"[Rank {item['rank']}] score={item['score']:.4f}\n{item['page_content'][:400]}"
            for item in result["retrieved"]
        ]
    )

    return answer, evidence


def run_qa(input_path, question):
    answer, evidence = rag_answer(question)

    return {
        "task": "qa",
        "input_file": input_path,
        "question": question,
        "answer": answer,
        "evidence": evidence,
    }