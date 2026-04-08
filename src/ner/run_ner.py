def run_ner(input_path):
    return {
        "task": "ner",
        "input_file": input_path,
        "entities": [
            {"text": "aspirin", "type": "Chemical"},
            {"text": "cancer", "type": "Disease"}
        ]
    }