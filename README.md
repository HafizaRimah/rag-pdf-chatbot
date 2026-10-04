# RAG PDF Chatbot

A learning-focused Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents.

Built completely from scratch — **no LangChain or high-level orchestration frameworks** — to understand and demonstrate how every layer of a modern RAG system operates under the hood.

---

## 🏛️ Architecture & Data Flow

```text
       [ PDF Document(s) ]
               │
               ▼
   1. PDF Text Extraction (PyMuPDF)
               │
               ▼
   2. Sentence-Aware Chunking (chunker)
               │
               ▼
   3. Dense Vector Embeddings (Sentence Transformers: all-MiniLM-L6-v2)
               │
               ▼
   4. Vector Store & Indexing (FAISS: IndexFlatIP)
               │
   ┌───────────┴───────────────────────────────┐
   │                                           │
[ User Query ]                                 │
   │                                           │
   ▼                                           ▼
Embed Query ────► 5. Semantic Search (Retriever) ────► Top-k Relevant Chunks
                                                              │
                                                              ▼
                                                   6. Local LLM (Ollama / llama3.2)
                                                              │
                                                              ▼
                                              7. Grounded Answer with Sources
                                                              │
                                                              ▼
                                                   Streamlit Web UI / CLI
```

---

## 🧩 Components Overview

| Module | File | Purpose |
|---|---|---|
| **PDF Loader** | [`src/pdf_loader.py`](src/pdf_loader.py) | Extracts raw text from single PDFs or directories using PyMuPDF; tracks page numbers and source filenames; handles validation and empty page skipping. |
| **Chunker** | [`src/chunker.py`](src/chunker.py) | Splits text at natural sentence boundaries into overlapping chunks to preserve contextual continuity without splitting sentences. |
| **Embedder** | [`src/embedder.py`](src/embedder.py) | Generates 384-dimensional dense vectors using `all-MiniLM-L6-v2`; loads the model once and reuses it for both batch chunks and single queries. |
| **Vector Store** | [`src/vector_store.py`](src/vector_store.py) | Manages FAISS vector indexing (`IndexFlatIP`) with normalized vectors for cosine similarity; manages metadata and index persistence to disk. |
| **Retriever** | [`src/retriever.py`](src/retriever.py) | End-to-end semantic search engine; builds FAISS indices from extracted pages and matches incoming queries against stored chunks. |
| **LLM Generator** | [`src/llm.py`](src/llm.py) | Connects to a local Ollama server (`llama3.2`) using Python standard library HTTP requests; enforces strict prompt grounding and disclaimers against hallucination. |
| **RAG Pipeline** | [`src/rag_pipeline.py`](src/rag_pipeline.py) | Master orchestrator coordinating extraction, indexing, retrieval, and LLM generation. Includes an interactive CLI loop. |
| **Web UI** | [`app.py`](app.py) | Streamlit interface supporting multi-PDF file upload, customizable retrieval parameters, interactive chat history, and collapsible source citations. |

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.11+**
- **Ollama** installed on your system ([ollama.com](https://ollama.com))

Make sure Ollama is running and has the `llama3.2` model downloaded:

```bash
# Pull the model
ollama pull llama3.2

# Start the Ollama server (if not already running as a background service)
ollama serve
```

### 2. Environment Setup

Clone the repository and set up your Python virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Streamlit Web Application

Launch the web app directly in your browser:

```bash
streamlit run app.py
```

1. Use the **sidebar** to upload one or more PDF files.
2. Adjust **chunk size** and **top-k** if desired.
3. Click **"Process Documents"** to extract text, generate embeddings, and build the FAISS index.
4. Ask questions in the chat prompt and inspect the expandable **"View sources"** dropdown for page numbers, similarity scores, and excerpt text.

### Running the Interactive CLI

You can also run the pipeline from your terminal:

```bash
# Process PDFs in the default data/documents folder:
python -m src.rag_pipeline

# Or pass a specific PDF file or folder:
python -m src.rag_pipeline path/to/document.pdf
```

---

## 🧪 Testing

The project includes unit and integration tests for every component.

To run the complete test suite:

```bash
python tests/run_all_tests.py
```

Or run individual tests:

```bash
python tests/test_pdf_loader.py
python tests/test_chunker.py
python tests/test_embedder.py
python tests/test_vector_store.py
python tests/test_retriever.py
python tests/test_llm.py
python tests/test_rag_pipeline.py
```

---

## 📂 Project Structure

```text
rag-pdf-chatbot/
├── .venv/                      # Virtual environment
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Pinned dependencies
├── README.md                   # Project documentation
├── app.py                      # Streamlit web application
│
├── data/
│   └── documents/              # Directory for input PDF documents
│
├── examples/
│   └── embedding_demo.py       # Standalone educational embedding demo
│
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py           # PyMuPDF text extractor
│   ├── chunker.py              # Sentence-aware text chunker
│   ├── embedder.py             # Sentence Transformers wrapper
│   ├── vector_store.py         # FAISS vector store
│   ├── retriever.py            # Semantic search engine
│   ├── llm.py                  # Local Ollama client
│   └── rag_pipeline.py         # Master end-to-end pipeline
│
└── tests/
    ├── run_all_tests.py        # Master test suite runner
    ├── test_pdf_loader.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_vector_store.py
    ├── test_retriever.py
    ├── test_llm.py
    └── test_rag_pipeline.py
```

---

## ✅ Completed Phases

- [x] **Phase 1** — Project inspection, clean structure, `.gitignore`, and `requirements.txt`
- [x] **Phase 2** — PDF text extraction with PyMuPDF
- [x] **Phase 3** — Sentence-aware text chunking with overlap
- [x] **Phase 4** — Sentence Transformers embeddings (`all-MiniLM-L6-v2`)
- [x] **Phase 5** — FAISS vector storage (`IndexFlatIP` with metadata persistence)
- [x] **Phase 6** — Semantic retrieval pipeline
- [x] **Phase 7** — Local LLM generation via Ollama (`llama3.2`)
- [x] **Phase 8** — Complete end-to-end RAG pipeline
- [x] **Phase 9** — Streamlit web user interface
- [x] **Phase 10** — Comprehensive test suite, error handling, and complete documentation
