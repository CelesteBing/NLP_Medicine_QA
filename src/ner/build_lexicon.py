"""
Step 1: Build a Medical Lexicon from Real Official Data Sources

"""

import re
import json
import gzip
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════
# Source 1 — BC5CDR (PubTator format)
# ══════════════════════════════════════════════════════════════════

def parse_bc5cdr_pubtator(filepath: str) -> dict[str, set[str]]:
    """
    Parse a BC5CDR PubTator-format file and extract all gold entity strings.

    PubTator format:
        Each article consists of multiple lines and articles are separated by blank lines.
        Title line:      PMID|t|title text
        Abstract line:   PMID|a|abstract text
        Annotation line: PMID\tstart\tend\tentity text\ttype\tCUI
        Type values:     Disease or Chemical

    """
    lexicon = defaultdict(set)
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}\nPlease download the BC5CDR dataset first as described above.")

    opener = gzip.open if str(filepath).endswith('.gz') else open
    annotation_pattern = re.compile(
        r'^(\d+)\t(\d+)\t(\d+)\t(.+?)\t(Disease|Chemical)\t(.*)$'
    )

    with opener(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            m = annotation_pattern.match(line)
            if m:
                entity_text = m.group(4).strip()
                entity_type = m.group(5)      # "Disease" or "Chemical"
                cui         = m.group(6)

                # Skip composite annotations (CUI is "-1" or empty; usually ambiguous entities)
                if not entity_text or entity_text == '-1':
                    continue
                # Filter out very short terms (single character) or very long terms
                # (more than 10 words, usually sentence fragments)
                if len(entity_text) < 2:
                    continue
                word_count = len(entity_text.split())
                if word_count > 10:
                    continue

                lexicon[entity_type].add(entity_text.lower())

    print(f"[BC5CDR] Parsing completed: Disease={len(lexicon['Disease'])}, Chemical={len(lexicon['Chemical'])}")
    return dict(lexicon)


# ══════════════════════════════════════════════════════════════════
# Source 2 — MeSH XML (desc2026.xml)
# ══════════════════════════════════════════════════════════════════

# Mapping from MeSH tree number prefix to entity type
# C.*  = Diseases               → Disease
# D.*  = Chemicals and Drugs    → Chemical
# D01–D10 are chemical categories, while some D20+ branches are not chemicals
MESH_TREE_TO_TYPE = {
    'C': 'Disease',
    'D': 'Chemical',
}

def parse_mesh_xml(filepath: str) -> dict[str, set[str]]:
    """
    Parse a MeSH descriptor XML file (desc2024.xml) and extract disease and chemical terms.

    Simplified MeSH XML structure:
        <DescriptorRecord>
          <DescriptorName>
            <String>Diabetes Mellitus</String>
          </DescriptorName>
          <TreeNumberList>
            <TreeNumber>C18.452.394</TreeNumber>   ← starts with C → Disease
          </TreeNumberList>
          <ConceptList>
            <Concept>
              <TermList>
                <Term>
                  <String>Diabetes</String>        ← synonym
                </Term>
              </TermList>
            </Concept>
          </ConceptList>
        </DescriptorRecord>

    """
    lexicon = defaultdict(set)
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            "Please download it."
        )

    print(f"[MeSH] Parsing {filepath} now (this file is large, please be patient) ...")

    # Iterative parsing to save memory
    context = ET.iterparse(filepath, events=('start', 'end'))
    context = iter(context)
    _, root = next(context)

    record_count = 0
    for event, elem in context:
        if event == 'end' and elem.tag == 'DescriptorRecord':
            record_count += 1
            if record_count % 5000 == 0:
                print(f"  Processed {record_count} records ...")

            # Collect all TreeNumbers
            tree_numbers = [tn.text for tn in elem.findall('.//TreeNumber') if tn.text]

            # Determine the type
            is_disease  = any(tn.startswith('C') for tn in tree_numbers)
            is_chemical = any(tn.startswith('D') for tn in tree_numbers)

            if not (is_disease or is_chemical):
                root.clear()
                continue

            entity_type = 'Disease' if is_disease else 'Chemical'

            # Collect all term strings (main term + synonyms)
            terms = set()
            for string_el in elem.findall('.//String'):
                if string_el.text:
                    term = string_el.text.strip().lower()
                    # Exclude terms that are too short, too long, or contain problematic forms
                    if 2 <= len(term) <= 100 and len(term.split()) <= 8:
                        # Exclude purely numeric terms or terms starting with digits
                        if not re.match(r'^\d', term):
                            terms.add(term)

            lexicon[entity_type].update(terms)
            root.clear()

    print(f"[MeSH] Parsing completed: Disease={len(lexicon['Disease'])}, Chemical={len(lexicon['Chemical'])}, Total records={record_count}")
    return dict(lexicon)


# ══════════════════════════════════════════════════════════════════
# Lexicon merge, save, load
# ══════════════════════════════════════════════════════════════════

def merge_lexicons(*lexicons: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    Merge multiple lexicons by taking the union.
    The same term may belong to different categories in different sources (for example, glucose).
    Strategy: Disease has higher priority than Chemical to reduce the risk of misclassifying disease terms as chemicals.
    """
    merged: dict[str, set[str]] = defaultdict(set)
    for lex in lexicons:
        for entity_type, terms in lex.items():
            merged[entity_type].update(terms)

    # If a term appears in both Disease and Chemical, remove it from Chemical
    overlap = merged.get('Disease', set()) & merged.get('Chemical', set())
    if overlap:
        merged['Chemical'] -= overlap
        print(f"[Merge] Removed {len(overlap)} overlapping terms from Chemical (kept them in Disease)")

    return dict(merged)


def save_lexicon(lexicon: dict[str, set[str]], output_path: str):
    """Save the lexicon as JSON; convert sets to sorted lists for readability and version control."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {k: sorted(v) for k, v in lexicon.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in data.values())
    print(f"\nLexicon saved to {output_path}")
    print(f"  Disease:  {len(data.get('Disease', []))} entries")
    print(f"  Chemical: {len(data.get('Chemical', []))} entries")
    print(f"  Total:    {total} entries")


def load_lexicon(lexicon_path: str) -> dict[str, list[str]]:
    """Load a previously built lexicon from JSON."""
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total = sum(len(v) for v in data.values())
    print(f"[Lexicon] Loaded: Disease={len(data.get('Disease',[]))}, Chemical={len(data.get('Chemical',[]))}, Total={total}")
    return data


def print_stats(lexicon_path: str):
    """Print lexicon statistics, including length distribution and sample terms."""
    data = load_lexicon(lexicon_path)
    for entity_type, terms in data.items():
        lengths = [len(t.split()) for t in terms]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        print(f"\n── {entity_type} ({len(terms)} entries) ──")
        print(f"  Average term length: {avg_len:.1f} words, Max: {max_len} words")
        # Group by word count
        from collections import Counter
        dist = Counter(lengths)
        for wc in sorted(dist.keys()):
            print(f"  {wc} word(s): {dist[wc]} entries")
        # Show 10 random examples
        import random
        samples = random.sample(list(terms), min(10, len(terms)))
        print(f"  Examples: {samples}")


# ══════════════════════════════════════════════════════════════════
# Main program
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build a medical NER lexicon from real official data sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    parser.add_argument('--bc5cdr',     metavar='PATH', help='Path to the BC5CDR PubTator-format file')
    parser.add_argument('--mesh',       metavar='PATH', help='Path to the MeSH desc2026.xml file')
    parser.add_argument('--output',     metavar='PATH', default='data/lexicon.json', help='Output lexicon JSON path')
    parser.add_argument('--stats',      metavar='PATH', help='View statistics of an already built lexicon')
    args = parser.parse_args()

    if args.stats:
        print_stats(args.stats)
        return

    if not any([args.bc5cdr, args.mesh]):
        parser.print_help()
        print("\nError: please specify at least one data source.")
        return

    lexicons = []

    if args.bc5cdr:
        lexicons.append(parse_bc5cdr_pubtator(args.bc5cdr))
    if args.mesh:
        lexicons.append(parse_mesh_xml(args.mesh))

    merged = merge_lexicons(*lexicons)
    save_lexicon(merged, args.output)


if __name__ == '__main__':
    main()
