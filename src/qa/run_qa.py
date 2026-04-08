def run_qa(input_path, question):
    return {
        "task": "qa",
        "input_file": input_path,
        "question": question,
        "answer": "This is a placeholder answer.",
        "evidence": "Sample retrieved text"
    }