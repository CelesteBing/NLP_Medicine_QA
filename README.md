# Information Extraction and Intelligent Question Answering in Medicine

**Team Name:** Bugs help bugs  
**Course:** COMP6713 – Natural Language Processing, 2026 T1, UNSW Sydney

---

## Team Members

| Name | zID |
|------|-----|
| Hanjin Hong | z5653224 |
| Yangxincan Li | z5614324 |
| Haoyue Bing| z5655080 |
| Ke Yang | z5562468 |
| Shuochen Zhu | z5507779 |

---

## Project Overview

This project builds a Biomedical Named Entity Recognition (NER) and Retrieval-Augmented Question Answering (RAG) system for the medical domain. It combines a lexicon-based regex baseline and a fine-tuned BioBERT model for entity extraction, and a FAISS-based dense retrieval pipeline paired with GPT-3.5-turbo for evidence-grounded question answering over PubMedQA.

**Key capabilities:**
- Biomedical NER: lexicon-based regex baseline (BC5CDR + MeSH) and fine-tuned BioBERT
- Retrieval-Augmented QA: `all-MiniLM-L6-v2` embeddings + FAISS cosine retrieval + GPT-3.5-turbo generation
- Automated evaluation: Exact Match and ROUGE-L on a 100-sample PubMedQA test set
- Interactive Gradio demo for NER visualisation and QA interaction

---

## Project Structure

```
NLP_Medicine_QA/
├── data/
│   ├── raw/                         # Raw datasets
│   │   └── bc5cdr/                  # BC5CDR dataset (NER)
│   │       ├── CDR_TrainingSet.PubTator.txt
│   │       └── CDR_TestSet.PubTator.txt
│   │
│   ├── processed/                  # Preprocessed data
│   │   ├── eval_input.json         # BIO-formatted NER inputs
│   │   ├── lexicon.json            # Lexicon for baseline NER
│   │   └── rag_test_results.json   # QA predictions + evaluation results
│   │
│   ├── predictions/                # Model predictions
│   │   └── biobert_preds.json      # BioBERT NER predictions
│   │
│   └── test/                       # Test folder (placeholder)
│       └── .gitkeep
│
├── models/
│   ├── .gitkeep
│   └── README.md
│
├── notebooks/
│   ├── .gitkeep
│   ├── BioBert.ipynb               # BioBERT fine-tuning notebook
│   └── commands_outputs.ipynb      # CLI / experiment logs
│
├── reports/
│   ├── .gitkeep
│   └── error_analysis_report.txt   # NER error analysis results
│
├── src/
│   ├── demo/
│   │   └── gradio_demo.py          # Gradio UI for NER + QA
│   │
│   ├── ner/
│   │   ├── baseline_ner.py         # Rule-based NER
│   │   ├── biobert_infer.py        # BioBERT inference
│   │   ├── build_lexicon.py        # Lexicon construction
│   │   ├── evaluate_ner.py         # NER evaluation + error analysis
│   │   └── run_ner.py              # NER entry point
│   │
│   ├── qa/
│   │   └── run_qa.py               # QA interface (rag_answer)
│   │
│   ├── retrieval/
│   │   ├── prepare_pubmedqa.py     # Prepare QA dataset
│   │   ├── retriever.py            # FAISS-based retrieval
│   │   └── rag_pipeline.py         # Full RAG pipeline (retrieval + LLM + evaluation)
│   │
│   └── utils/
│       └── io.py                   # File I/O utilities
│
├── evaluate_rag.py                 # Standalone QA evaluation script
├── run.py                          # CLI entry point (NER + QA)
├── requirements.txt                # Dependencies
├── README.md                       # Project overview
├── structure.md                    # Project structure description
└── .gitignore
```

---

## Environment Setup

### Prerequisites

- Python 3.9 or above
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include: `torch`, `transformers==4.41.2`, `sentence-transformers==2.7.0`, `faiss-cpu`, `langchain`, `openai==1.30.1`, `gradio>=4.0.0`, `datasets`, `rouge-score`, `seqeval>=1.2.2`.

### Set OpenAI API Key

The RAG pipeline uses GPT-3.5-turbo via the OpenAI API. Set your key as an environment variable before running any QA task:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

---

## Data Setup

### BC5CDR Dataset (NER)

The BC5CDR corpus is included in `data/raw/bc5cdr/`:
```
data/raw/bc5cdr/CDR_TrainingSet.PubTator.txt
data/raw/bc5cdr/CDR_TestSet.PubTator.txt
```

### MeSH Descriptor File

The MeSH XML file (`desc2026.xml`) is **not included** in the submission due to its size (~300MB). Download it from the [NLM MeSH FTP site](https://www.nlm.nih.gov/databases/download/mesh.html) and place it at:

```
data/raw/mesh/desc2026.xml
```


### PubMedQA Dataset

Preprocessed PubMedQA chunks are included in `data/processed/`. To regenerate from scratch (loads 500 samples from Hugging Face):

```bash
python src/retrieval/prepare_pubmedqa.py
```

---

## Model Setup

### Fine-tuned BioBERT

The fine-tuned BioBERT model is not included in the submission due to file size. 

**Fine-tune from scratch:**

Open and run `notebooks/BioBert.ipynb`. The notebook covers data preparation, training, and saving the model to `models/biobert/`.

---

## Running the Project

### Step 1 — Build the NER Lexicon (first-time setup only)

```bash
python src/ner/build_lexicon.py
```

This generates `data/processed/lexicon.json` from BC5CDR and MeSH.

### Step 2 — Prepare PubMedQA Data (first-time setup only)

```bash
python src/retrieval/prepare_pubmedqa.py
```

This generates `data/processed/pubmedqa_docs.pkl` and `pubmedqa_chunks.pkl`.

---

### Option A: CLI (`run.py`)

Run NER on a text file:

```bash
python run.py --task ner --input path/to/input.txt --output output.json
```

Run QA on a text file with a question:

```bash
python run.py --task qa --input path/to/input.txt --question "What are the side effects of aspirin?" --output output.json
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--task` | Yes | `ner` or `qa` |
| `--input` | Yes | Path to input text file |
| `--question` | QA only | Question string for QA task |
| `--output` | No | Output JSON path (default: `output.json`) |

---

### Option B: Module Scripts

**Run baseline NER:**
```bash
python src/ner/baseline_ner.py
```

**Run BioBERT NER inference:**
```bash
python src/ner/biobert_infer.py
```

**Evaluate NER (precision, recall, F1, error analysis):**
```bash
python src/ner/evaluate_ner.py
```

**Run RAG pipeline batch test:**
```bash
python src/retrieval/rag_pipeline.py
```

---

### Option C: Gradio Web Demo

```bash
python src/demo/gradio_demo.py
```

Open the URL shown in the terminal (typically `http://127.0.0.1:7860`). The demo has two tabs:
- **NER Demo** – Enter medical text and visualise extracted Chemical/Disease entities using BioBERT
- **QA Demo** – Enter a biomedical question and view the generated answer with retrieved evidence

---

## Evaluation

### NER Evaluation

```bash
python src/ner/evaluate_ner.py
```

Outputs: precision, recall, F1 per entity type; error distribution saved to `reports/error_analysis_report.txt`.

### RAG QA Evaluation (Exact Match + ROUGE-L)

```bash
python evaluate_rag.py
```

Evaluates on `data/test/pubmedqa_testset_100.csv` (100 samples). Prints Exact Match and ROUGE-L scores, and extracts up to 6 bad cases for qualitative analysis.

> Requires `OPENAI_API_KEY` to be set in the environment before running.

---

## Notes

- All file paths in the code use `pathlib.Path` with paths relative to the project root — no absolute paths are used.
- Notebook outputs in `notebooks/` have been retained as required.
