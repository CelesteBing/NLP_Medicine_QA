from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "biobert"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

id2label = model.config.id2label


def _normalize_label(label: str) -> str:
    if label == "Chemical":
        return "Chemical"
    if label == "Disease":
        return "Disease"
    return label


def _merge_adjacent_segments(
    segments: List[Tuple[str, Optional[str]]]
) -> List[Tuple[str, Optional[str]]]:
    """
    Merge consecutive segments with the same label or consecutive plain-text pieces.
    This helps fix cases like:
    ('i', 'Chemical'), ('buprofen', 'Chemical') -> ('ibuprofen', 'Chemical')
    """
    if not segments:
        return segments

    merged: List[Tuple[str, Optional[str]]] = [segments[0]]

    for text, label in segments[1:]:
        prev_text, prev_label = merged[-1]

        if prev_label == label:
            merged[-1] = (prev_text + text, label)
        else:
            merged.append((text, label))

    return merged


def biobert_predict(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Input:
        raw text string
    Output:
        List[Tuple[text_segment, label]]
        Example:
        [
            ("The patient has ", None),
            ("fever", "Disease"),
            (" and takes ", None),
            ("ibuprofen", "Chemical"),
            (".", None)
        ]
    """
    if not text.strip():
        return [(text, None)]

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

    entities = []
    current_entity = None

    for pred_id, (start, end) in zip(predictions, offset_mapping):
        if start == 0 and end == 0:
            continue

        label = id2label[pred_id]

        if label == "O":
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        if label.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)

            current_entity = {
                "start": start,
                "end": end,
                "label": _normalize_label(label[2:])
            }

        elif label.startswith("I-"):
            entity_type = _normalize_label(label[2:])

            if current_entity is not None and current_entity["label"] == entity_type:
                # expand current entity
                current_entity["end"] = end
            else:
                if current_entity is not None:
                    entities.append(current_entity)

                current_entity = {
                    "start": start,
                    "end": end,
                    "label": entity_type
                }

    if current_entity is not None:
        entities.append(current_entity)

    results: List[Tuple[str, Optional[str]]] = []
    cursor = 0

    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]

        if cursor < start:
            results.append((text[cursor:start], None))

        results.append((text[start:end], label))
        cursor = end

    if cursor < len(text):
        results.append((text[cursor:], None))

    if not results:
        return [(text, None)]

    results = _merge_adjacent_segments(results)

    return results