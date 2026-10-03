# RAG PDF Chatbot

A learning-focused Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents.

Built from scratch — no LangChain — to understand how RAG works under the hood.

## Architecture

```
PDF → Text Extraction → Chunking → Embeddings → FAISS Index
                                                      ↓
                              User Query → Embedding → Semantic Search
                                                      ↓
                                               Relevant Chunks → LLM → Answer with Sources
```

## Progress

- [x] Phase 1 — Project setup and structure
- [ ] Phase 2 — PDF text extraction
- [ ] Phase 3 — Text chunking
- [ ] Phase 4 — Embeddings
- [ ] Phase 5 — FAISS vector storage
- [ ] Phase 6 — Retrieval
- [ ] Phase 7 — LLM generation
- [ ] Phase 8 — Complete RAG pipeline
- [ ] Phase 9 — Streamlit UI
- [ ] Phase 10 — Testing, docs, final cleanup

## Tech Stack

- **Python 3.11**
- **Sentence Transformers** — `all-MiniLM-L6-v2` for embeddings
- **FAISS** — vector similarity search
- **PyMuPDF** — PDF text extraction
- **Streamlit** — web interface
- **NumPy** — vector operations

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
rag-pdf-chatbot/
├── data/documents/    ← Place your PDFs here
├── examples/          ← Learning scripts and demos
├── src/               ← Core RAG modules
├── tests/             ← Test files
├── app.py             ← Streamlit UI (Phase 9)
├── requirements.txt
└── README.md
```
