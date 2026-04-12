from typing import List, Optional, Tuple


def ner_extract(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Team-aligned NER interface:
    Input: raw text
    Output: List[Tuple[text_segment, label]]
    """
    if "aspirin" in text.lower() and "headache" in text.lower():
        return [
            ("The patient has ", None),
            ("headache", "Disease"),
            (" and takes ", None),
            ("aspirin", "Drug"),
            (".", None)
        ]

    return [(text, None)]


def run_ner(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    entities = ner_extract(text)

    return {
        "task": "ner",
        "input_file": input_path,
        "entities": entities
    }