import os
import sys
import pandas as pd
from rouge_score import rouge_scorer

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.qa.rag_pipeline import load_chunks, build_faiss_index, answer_question

#  Please enter your OpenAI API key here.
os.environ["OPENAI_API_KEY"] = "sk-xxxxx"


def calculate_metrics():
    csv_path = os.path.join(project_root, "data", "processed", "pubmedqa_testset_100.csv")
    print(f"The test set is being loaded: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"The test set cannot be found! Please confirm the path.")
        return

    # Before running the benchmark, initialize the model and vector library first (only load once to significantly improve the speed!)
    print("Initializing the RAG model and the FAISS vector library... (This may take about ten seconds or so)...")
    chunks = load_chunks()
    embed_model, index = build_faiss_index(chunks)
    print("System initialization completed!")

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    exact_match_count = 0
    rouge_l_scores = []
    bad_cases = []

    print(f"\nStarting the automated assessment (a total of {len(df)} data entries)...")

    for idx, row in df.iterrows():
        question = str(row['question'])
        standard_answer = str(row['standard_answer'])

        # calling interface
        try:
            result = answer_question(
                question=question,
                embed_model=embed_model,
                index=index,
                chunks=chunks
            )
            predicted_answer = result["answer"]
            # Extract the retrieved original text for display.
            evidence = "\n".join([item["page_content"] for item in result["retrieved"]])

        except Exception as e:
            print(f"Error occurred when generating question {idx + 1}: {e}")
            continue

        # EM score
        if predicted_answer.strip().lower() == standard_answer.strip().lower():
            exact_match_count += 1

        # ROUGE score
        scores = scorer.score(standard_answer, predicted_answer)
        rouge_l_f1 = scores['rougeL'].fmeasure
        rouge_l_scores.append(rouge_l_f1)

        # Collecting wrong answers
        if rouge_l_f1 < 0.1 and len(bad_cases) < 6:
            bad_cases.append({
                "question": question,
                "standard_answer": standard_answer,
                "predicted_answer": predicted_answer,
                "evidence": evidence
            })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} items...")

    final_em = (exact_match_count / len(df)) * 100
    final_rouge_l = (sum(rouge_l_scores) / len(rouge_l_scores)) * 100

    print("\n" + "=" * 50)
    print("RAG System QA Evaluation Report")
    print("=" * 50)
    print(f"Test data volume: {len(df)} entries")
    print(f"Exact Match: {final_em:.2f}%")
    print(f"ROUGE-L:   {final_rouge_l:.2f}%")
    print("=" * 50)

    if bad_cases:
        print("\nBad Cases Extracted:")
        for i, case in enumerate(bad_cases[:3]):
            print(f"\n【Bad Case {i + 1}】")
            print(f"Question: {case['question']}")
            print(f"Standard answer: {case['standard_answer']}")
            print(f"Model prediction: {case['predicted_answer']}")
            print(f"Search source: {case['evidence'][:150]}... \n" + "-" * 30)


if __name__ == "__main__":
    calculate_metrics()