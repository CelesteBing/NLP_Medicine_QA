def run_ner(input_path):
    print("Running NER...")
    return {
        "entities": [
            {"text": "aspirin", "type": "Chemical"},
            {"text": "cancer", "type": "Disease"}
        ]
    }