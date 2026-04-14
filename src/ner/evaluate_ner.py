"""
Step 3: NER Evaluation Script

"""

import io
import json
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

try:
    from seqeval.metrics import (
        precision_score,
        recall_score,
        f1_score,
        classification_report,
    )
    from seqeval.scheme import IOB2
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print(
        "[Warning] seqeval not found -- falling back to built-in implementation.\n"
        "          Install with: pip install seqeval\n"
        "          Both implementations produce equivalent results; seqeval\n"
        "          additionally enforces strict BIO format validation.\n"
    )


# =============================================================================
# Metric computation
# =============================================================================

def compute_metrics_seqeval(all_gold: List[List[str]],
                             all_pred: List[List[str]]) -> Dict:
    """
    Compute P / R / F1 using seqeval.

    Args:
        all_gold: List of gold label sequences, one per document.
        all_pred: List of predicted label sequences, same structure.

    """
    report_str = classification_report(
        all_gold, all_pred,
        scheme=IOB2, output_dict=False, zero_division=0,
    )
    report_dict = classification_report(
        all_gold, all_pred,
        scheme=IOB2, output_dict=True, zero_division=0,
    )

    metrics = {}
    for label, vals in report_dict.items():
        if label == "accuracy":
            continue
        metrics[label] = {
            "precision": round(vals["precision"], 4),
            "recall":    round(vals["recall"], 4),
            "f1":        round(vals["f1-score"], 4),
            "support":   int(vals["support"]),
        }
    metrics["_seqeval_report"] = report_str
    return metrics


def compute_metrics_builtin(all_gold: List[List[str]],
                             all_pred: List[List[str]]) -> Dict:
    """
    Fallback implementation using the standard library only.
    Evaluation granularity: exact span match (both position and label must match the gold standard to count as a true positive).
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for gold_labels, pred_labels in zip(all_gold, all_pred):
        gold_spans = set(_bio_to_spans(gold_labels))
        pred_spans = set(_bio_to_spans(pred_labels))

        for span in pred_spans:
            if span in gold_spans:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1
        for span in gold_spans:
            if span not in pred_spans:
                fn[span[2]] += 1

    all_labels = sorted(set(list(tp) + list(fp) + list(fn)))
    metrics = {}
    total_tp = total_fp = total_fn = 0

    for label in all_labels:
        t, f_p, f_n = tp[label], fp[label], fn[label]
        total_tp += t; total_fp += f_p; total_fn += f_n
        p  = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r  = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[label] = {
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "support": t + f_n,
        }

    p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    metrics["micro avg"] = {
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(f1, 4), "support": total_tp + total_fn,
    }
    return metrics


def compute_metrics(all_gold: List[List[str]],
                    all_pred: List[List[str]]) -> Dict:
    """Use seqeval if available, otherwise fall back to built-in."""
    if SEQEVAL_AVAILABLE:
        return compute_metrics_seqeval(all_gold, all_pred)
    return compute_metrics_builtin(all_gold, all_pred)


# =============================================================================
# BIO utility
# =============================================================================

def _bio_to_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
    """
    Convert a BIO label sequence into a list of entity spans.
    """
    spans = []
    i = 0
    while i < len(labels):
        tag = labels[i]
        if tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{label}":
                j += 1
            spans.append((i, j, label))
            i = j
        else:
            i += 1
    return spans


# =============================================================================
# Error categorisation
# =============================================================================

ERROR_TYPE_DESC = {
    "Missing entity":  "Entity exists in the gold standard but was not predicted at all",
    "False positive":  "Predicted entity does not exist in the gold standard",
    "Boundary shift":  "Correct entity type but the span start or end is off by one or more tokens",
    "Label confusion": "Span position is correct but the entity type is wrong (Disease vs Chemical)",
    "Partial overlap": "Predicted span partially overlaps a gold span but does not fit any category above",
}

def categorize_errors(tokens:      List[str],
                      gold_labels: List[str],
                      pred_labels: List[str],
                      pmid:        str = "?") -> List[Dict]:
    """
    Categorise every prediction error for a single document.

    Classification priority (highest to lowest):
      1. Exact match (TP)       -- skip, not an error
      2. Same span, wrong label -- Label confusion
      3. Same label, span off by <= 4 tokens total -- Boundary shift
      4. Predicted span overlaps any gold span     -- Partial overlap
      5. No overlap at all                         -- False positive
      6. Gold span has no corresponding prediction -- Missing entity
    """
    gold_spans = _bio_to_spans(gold_labels)
    pred_spans = _bio_to_spans(pred_labels)
    gold_set   = set(gold_spans)
    pred_set   = set(pred_spans)
    errors     = []

    def overlaps(a_s, a_e, b_s, b_e):
        return not (a_e <= b_s or a_s >= b_e)

    def span_text(s, e):
        return " ".join(tokens[s:e])

    # -- Analyse each predicted span ------------------------------------------
    for ps, pe, pl in pred_spans:
        if (ps, pe, pl) in gold_set:
            continue  # True positive, skip

        pred_text = span_text(ps, pe)

        # Label confusion: exact position match, wrong label
        label_mismatch = [(gs, ge, gl) for gs, ge, gl in gold_spans
                          if gs == ps and ge == pe and gl != pl]
        if label_mismatch:
            _, _, gl = label_mismatch[0]
            errors.append({
                "pmid":   pmid,
                "type":   "Label confusion",
                "pred":   f'"{pred_text}" [{pl}]',
                "gold":   f'"{pred_text}" [{gl}]',
                "note":   f"Span [{ps}, {pe}) is correct but label predicted as {pl}, should be {gl}",
                "tokens": tokens,
                "pred_span": (ps, pe, pl),
                "gold_span": (ps, pe, gl),
            })
            continue

        # Boundary shift: same label, position off by <= 4 tokens total
        boundary_match = [(gs, ge, gl) for gs, ge, gl in gold_spans
                          if gl == pl
                          and abs(gs - ps) + abs(ge - pe) <= 4
                          and not (gs == ps and ge == pe)]
        if boundary_match:
            gs, ge, gl = boundary_match[0]
            left_d  = ps - gs
            right_d = pe - ge
            errors.append({
                "pmid":   pmid,
                "type":   "Boundary shift",
                "pred":   f'"{pred_text}" [{pl}]',
                "gold":   f'"{span_text(gs, ge)}" [{gl}]',
                "note":   (f"Predicted [{ps},{pe}), gold [{gs},{ge}); "
                           f"left offset {left_d:+d} token(s), right offset {right_d:+d} token(s)"),
                "tokens": tokens,
                "pred_span": (ps, pe, pl),
                "gold_span": (gs, ge, gl),
            })
            continue

        # Partial overlap: overlaps some gold span but not cleanly
        overlap_match = [(gs, ge, gl) for gs, ge, gl in gold_spans
                         if overlaps(ps, pe, gs, ge)]
        if overlap_match:
            gs, ge, gl = overlap_match[0]
            errors.append({
                "pmid":   pmid,
                "type":   "Partial overlap",
                "pred":   f'"{pred_text}" [{pl}]',
                "gold":   f'"{span_text(gs, ge)}" [{gl}]',
                "note":   f"Predicted [{ps},{pe}) partially overlaps gold [{gs},{ge}) but is not aligned",
                "tokens": tokens,
                "pred_span": (ps, pe, pl),
                "gold_span": (gs, ge, gl),
            })
            continue

        # False positive: no overlap with any gold span
        errors.append({
            "pmid":   pmid,
            "type":   "False positive",
            "pred":   f'"{pred_text}" [{pl}]',
            "gold":   "(not in gold standard)",
            "note":   f"Model predicted {pl} at [{ps},{pe}) but no entity exists there",
            "tokens": tokens,
            "pred_span": (ps, pe, pl),
            "gold_span": None,
        })

    # -- Find gold spans with no matching prediction ---------------------------
    for gs, ge, gl in gold_spans:
        if (gs, ge, gl) in pred_set:
            continue  # Correctly predicted
        # Skip if already captured as boundary shift / partial overlap above
        already_recorded = any(
            overlaps(ps, pe, gs, ge)
            for ps, pe, pl in pred_spans
            if (ps, pe, pl) not in gold_set
        )
        if not already_recorded:
            errors.append({
                "pmid":   pmid,
                "type":   "Missing entity",
                "pred":   "(not predicted)",
                "gold":   f'"{span_text(gs, ge)}" [{gl}]',
                "note":   f"{gl} entity at [{gs},{ge}) was completely missed by the model",
                "tokens": tokens,
                "pred_span": None,
                "gold_span": (gs, ge, gl),
            })

    return errors


# =============================================================================
# Data loading
# =============================================================================

def load_data(filepath: str) -> List[Dict]:
    """Load eval_input.json or biobert_preds.json."""
    path = Path(filepath)
    if not path.exists():
        print(f"[Error] File not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Load] {filepath}: {len(data)} documents")
    return data


# =============================================================================
# Printing
# =============================================================================

SEP = "=" * 68

def print_metrics(model_name: str, metrics: Dict):
    print(f"\n{SEP}")
    print(f"  {model_name}  --  Evaluation Results")
    print(SEP)
    if "_seqeval_report" in metrics:
        print(metrics["_seqeval_report"])
    else:
        print(f"  {'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}  {'Support':>8}")
        print(f"  {'-'*60}")
        for label, m in metrics.items():
            pfx = "-> " if label == "micro avg" else "   "
            print(f"{pfx}{label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f}"
                  f" {m['f1']:>10.4f}  {m.get('support', 0):>8}")


def print_comparison(base_metrics: Dict, bio_metrics: Dict,
                     base_name: str = "Baseline",
                     bio_name:  str = "BioBERT"):
    print(f"\n{SEP}")
    print(f"  {base_name}  vs  {bio_name}  --  F1 Comparison")
    print(SEP)
    key_labels = ["Disease", "Chemical", "micro avg"]
    print(f"  {'Label':<16} {base_name + ' F1':>14} {bio_name + ' F1':>14} {'F1 gain':>10}")
    print(f"  {'-'*56}")
    for label in key_labels:
        b  = base_metrics.get(label, {}).get("f1")
        bb = bio_metrics.get(label, {}).get("f1")
        if b is None or bb is None:
            continue
        delta = bb - b
        pfx   = "-> " if label == "micro avg" else "   "
        sign  = "+" if delta >= 0 else ""
        print(f"{pfx}{label:<16} {b:>14.4f} {bb:>14.4f} {sign}{delta:>9.4f}")


def highlight_sentence(tokens: List[str],
                       pred_span: tuple,
                       gold_span: tuple) -> str:
    """
    Reconstruct the full sentence from tokens and annotate the relevant spans.

    Markers used:
      >>token(s)<<[PRED]  -- what the model predicted
      >>token(s)<<[GOLD]  -- what the gold standard says

    When pred and gold spans are different, both are marked simultaneously.
    When they are the same position (label confusion), a single marker is used.
    Tokens outside any marked span are joined with spaces as-is.
    """
    n = len(tokens)
    # Build a per-token annotation: None | "PRED" | "GOLD" | "BOTH"
    anno = [None] * n

    if pred_span is not None:
        ps, pe, _ = pred_span
        for i in range(ps, pe):
            anno[i] = "PRED"

    if gold_span is not None:
        gs, ge, _ = gold_span
        for i in range(gs, ge):
            anno[i] = "BOTH" if anno[i] == "PRED" else "GOLD"

    # Build the annotated sentence token by token
    parts = []
    i = 0
    while i < n:
        tag = anno[i]
        if tag is None:
            parts.append(tokens[i])
            i += 1
        else:
            # Collect the run of tokens sharing the same tag
            j = i + 1
            while j < n and anno[j] == tag:
                j += 1
            span_str = " ".join(tokens[i:j])
            if tag == "PRED":
                parts.append(f">>{span_str}<<[PRED]")
            elif tag == "GOLD":
                parts.append(f">>{span_str}<<[GOLD]")
            else:  # BOTH (label confusion: same position, different label)
                parts.append(f">>{span_str}<<[PRED/GOLD]")
            i = j

    return " ".join(parts)


def print_error_analysis(errors:     List[Dict],
                          model_name: str = "model",
                          n_examples: int = 25):
    """
    Print error type distribution and up to n_examples representative cases.

    Cases are allocated to each error type proportionally to its frequency,
    so every error type is represented. Random seed 42 ensures reproducibility.
    """
    total = len(errors)
    if total == 0:
        print(f"\n  {model_name}: no prediction errors.")
        return

    print(f"\n{SEP}")
    print(f"  {model_name}  --  Error Analysis Report ({total} errors total)")
    print(SEP)

    # -- Error type distribution ----------------------------------------------
    type_counts = Counter(e["type"] for e in errors)
    print(f"\n  Error type distribution:\n")
    print(f"  {'Type':<20} {'Count':>6}  {'%':>7}  Bar chart")
    print(f"  {'-'*55}")
    for etype, count in type_counts.most_common():
        pct = count / total * 100
        bar = "█" * max(1, round(pct / 3))
        print(f"  {etype:<20} {count:>6}  {pct:>6.1f}%  {bar}")

    # -- Select representative examples proportionally ------------------------
    print(f"\n  Representative error examples"
          f" ({n_examples} shown, grouped by type):")

    errors_by_type = defaultdict(list)
    for e in errors:
        errors_by_type[e["type"]].append(e)

    sorted_types = [t for t, _ in type_counts.most_common()]

    # Allocate quota per type
    quota: Dict[str, int] = {}
    allocated = 0
    for i, etype in enumerate(sorted_types):
        if i == len(sorted_types) - 1:
            quota[etype] = max(1, n_examples - allocated)
        else:
            q = max(1, round(type_counts[etype] / total * n_examples))
            quota[etype] = q
            allocated += q

    random.seed(42)
    selected: Dict[str, List[Dict]] = {}
    for etype in sorted_types:
        pool = errors_by_type[etype]
        k    = min(quota.get(etype, 1), len(pool))
        selected[etype] = random.sample(pool, k)

    # Print examples
    for etype in sorted_types:
        cases = selected[etype]
        desc  = ERROR_TYPE_DESC.get(etype, "")
        print(f"\n  +-- {etype}")
        print(f"  |   {desc}")
        print(f"  |   Total: {type_counts[etype]}  |  Shown: {len(cases)}")
        for case in cases:
            print(f"  |   {'-'*88}")
            print(f"  |   PMID      : {case['pmid']}")
            # Highlighted sentence
            if case.get("tokens"):
                sent = highlight_sentence(
                    case["tokens"], case.get("pred_span"), case.get("gold_span")
                )
                # Wrap long sentences to fit within 88 chars
                prefix = "  |   Sentence  : "
                wrap_width = 88 - len(prefix)
                words = sent.split(" ")
                line, lines = "", []
                for w in words:
                    if line and len(line) + 1 + len(w) > wrap_width:
                        lines.append(line)
                        line = w
                    else:
                        line = (line + " " + w).strip()
                if line:
                    lines.append(line)
                cont_prefix = "  |   " + " " * (len(prefix) - len("  |   "))
                print(prefix + lines[0])
                for l in lines[1:]:
                    print(cont_prefix + l)
            print(f"  |   Predicted  : {case['pred']}")
            print(f"  |   Gold       : {case['gold']}")
            print(f"  |   Note       : {case['note']}")
        print(f"  |")


# =============================================================================
# Core evaluation pipeline
# =============================================================================

def evaluate(filepath:   str,
             model_name: str = "model",
             n_examples: int = 25) -> Tuple[Dict, List[Dict]]:
    """Load file, compute metrics, collect all error cases, return results."""
    data = load_data(filepath)

    all_gold, all_pred = [], []
    all_errors = []

    for item in data:
        tokens      = item["tokens"]
        gold_labels = item["gold_labels"]
        pred_labels = item["pred_labels"]
        pmid        = item.get("pmid", "?")

        all_gold.append(gold_labels)
        all_pred.append(pred_labels)
        all_errors.extend(categorize_errors(tokens, gold_labels, pred_labels, pmid))

    metrics = compute_metrics(all_gold, all_pred)
    print_metrics(model_name, metrics)
    print_error_analysis(all_errors, model_name, n_examples)
    return metrics, all_errors


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NER evaluation: seqeval P/R/F1 + 20-30 categorised error examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_ner.py --file data/eval_input.json --model "Lexicon Baseline"
  python evaluate_ner.py --baseline data/eval_input.json \\
                         --biobert  data/biobert_preds.json \\
                         --report   data/error_analysis_report.txt
        """
    )
    parser.add_argument("--file",       metavar="PATH",
                        help="JSON file with gold_labels and pred_labels (single-model mode)")
    parser.add_argument("--model",      default="model",
                        help="Model name shown in output (default: 'model')")
    parser.add_argument("--baseline",   metavar="PATH",
                        help="Baseline JSON file (comparison mode)")
    parser.add_argument("--biobert",    metavar="PATH",
                        help="BioBERT JSON file (comparison mode)")
    parser.add_argument("--report",     metavar="PATH",
                        default="error_analysis_report.txt",
                        help="Path to save the full report (default: error_analysis_report.txt)")
    parser.add_argument("--n-examples", type=int, default=25,
                        help="Number of error examples to display, recommended 20-30 (default: 25)")
    args = parser.parse_args()

    def capture_and_save(fn, *a, **kw):
        """Run fn(*a, **kw), tee output to terminal and report file."""
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        fn(*a, **kw)
        sys.stdout = old
        text = buf.getvalue()
        print(text)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Report] Saved to: {args.report}")

    if args.file:
        metrics, errors = evaluate(args.file, args.model, args.n_examples)
        def write_report():
            print_metrics(args.model, metrics)
            print_error_analysis(errors, args.model, args.n_examples)
        capture_and_save(write_report)

    elif args.baseline and args.biobert:
        base_data = load_data(args.baseline)
        bio_data  = load_data(args.biobert)

        base_gold, base_pred, all_base_errors = [], [], []
        bio_gold,  bio_pred,  all_bio_errors  = [], [], []

        for item in base_data:
            base_gold.append(item["gold_labels"])
            base_pred.append(item["pred_labels"])
            all_base_errors.extend(
                categorize_errors(item["tokens"], item["gold_labels"],
                                  item["pred_labels"], item.get("pmid", "?")))

        for item in bio_data:
            bio_gold.append(item["gold_labels"])
            bio_pred.append(item["pred_labels"])
            all_bio_errors.extend(
                categorize_errors(item["tokens"], item["gold_labels"],
                                  item["pred_labels"], item.get("pmid", "?")))

        base_metrics = compute_metrics(base_gold, base_pred)
        bio_metrics  = compute_metrics(bio_gold,  bio_pred)

        def write_report():
            print_metrics("Lexicon Baseline", base_metrics)
            print_metrics("BioBERT (fine-tuned)", bio_metrics)
            print_comparison(base_metrics, bio_metrics)
            print_error_analysis(all_base_errors, "Lexicon Baseline", args.n_examples)
            if all_bio_errors:
                print_error_analysis(all_bio_errors, "BioBERT (fine-tuned)", args.n_examples)

        capture_and_save(write_report)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
