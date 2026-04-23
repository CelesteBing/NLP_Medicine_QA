
---

# **Project Structure**

This document describes the directory layout and responsibilities of each component in the **Medical NLP QA System**.

---

## **Overview**

The project is organized into modular components covering:

* Data processing
* Named Entity Recognition (NER)
* Retrieval-Augmented Question Answering (RAG)
* Evaluation and error analysis
* User interfaces (CLI and demo)

---

## **Directory Structure**

```text
project/
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

## **Module Responsibilities**

### **Data Layer (`data/`)**

* Stores both raw datasets and processed files
* Raw data is not committed if too large
* Processed data is used for model input and evaluation

---

### **NER Module (`src/ner/`)**

Responsible for biomedical entity extraction:

* `build_lexicon.py`
  → Constructs lexicon from BC5CDR and MeSH

* `baseline_ner.py`
  → Rule-based entity extraction

* `biobert_infer.py`
  → Neural model inference

* `evaluate_ner.py`
  → Computes metrics and error distribution

---

### **Retrieval Module (`src/retrieval/`)**

Implements the retrieval component of RAG:

* `prepare_pubmedqa.py`  
  → Preprocess QA dataset and build document chunks

* `retriever.py`  
  → Generates embeddings and performs FAISS-based similarity search

* `rag_pipeline.py`  
  → Integrates retrieval with LLM generation and evaluation

---

### **QA Module (`src/qa/`)**

* Provides a unified interface (`rag_answer`)
* Connects retrieval and generation

---

### **RAG Pipeline (`src/retrieval/rag_pipeline.py`)**

Core of QA system:

* Runs retrieval + generation
* Performs batch evaluation
* Computes metrics
* Extracts bad cases automatically

---

### **Demo Module (`src/demo/`)**

* Interactive web interface using Gradio
* Includes:

  * NER visualization
  * QA interaction

---

### **Reports (`reports/`)**

* Stores evaluation outputs
* Includes:

  * NER error analysis
  * QA evaluation summaries

---

### **QA Evaluation (`evaluate_rag.py`)**

* Computes QA metrics (Exact Match, ROUGE-L)
* Evaluates end-to-end RAG performance
* Outputs structured evaluation results

---

### **CLI (`run.py`)**

* Unified entry point for:

  * NER task
  * QA task
* Allows quick testing without UI

---

## **Data Flow**

```text
Raw Text / Question
   ↓
NER Module (Baseline / BioBERT)
   ↓
Optional Structured Information
   ↓
Retriever (FAISS + Embeddings)
   ↓
Relevant Context (Top-k Chunks)
   ↓
LLM Generation (RAG)
   ↓
Answer + Supporting Evidence
```

---

## **Notes**

* Large files (e.g., MeSH XML) should not be pushed to GitHub
* Model files may be stored locally or downloaded separately
* QA module requires API access for full functionality
* Evaluation outputs are reproducible using provided scripts

---


