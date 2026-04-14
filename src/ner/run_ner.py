import re
from typing import List, Optional, Tuple


def ner_extract(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Return a list of (text_segment, label) while preserving spaces
    so Gradio HighlightedText can render correctly.
    """
    pieces = re.findall(r"\S+|\s+", text)

    results: List[Tuple[str, Optional[str]]] = []

    for piece in pieces:
        # spaces / tabs / newlines should stay unlabelled
        if piece.isspace():
            results.append((piece, None))
            continue

        token = piece.strip()

        # remove surrounding punctuation for matching only
        normalized = token.lower().strip(".,!?;:()[]{}\"'")

        if normalized in ["aspirin", "ibuprofen"]:
            results.append((piece, "Drug"))
        elif normalized in ["fever", "headache"]:
            results.append((piece, "Disease"))
        else:
            results.append((piece, None))

    return results



def run_ner(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    entities = ner_extract(text)

    return {
        "task": "ner",
        "input_file": input_path,
        "entities": entities
    }