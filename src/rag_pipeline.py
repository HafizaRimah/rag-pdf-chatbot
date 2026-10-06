"""
rag_pipeline.py — Complete RAG pipeline: PDF to answer.

This is the top-level module that connects every piece:

    PDF files
        -> pdf_loader (extract text)
        -> chunker (split into chunks)
        -> embedder (create vectors)
        -> vector_store (FAISS index)
        -> retriever (semantic search)
        -> llm (generate answer)
        -> Answer with sources

Usage:
    from src.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.load_documents("data/documents")
    answer = rag.ask("What is machine learning?")
"""

from src.pdf_loader import load_pdf, load_pdfs_from_directory
from src.retriever import Retriever
from src.llm import LLM


class RAGPipeline:
    """
    Complete RAG pipeline: load PDFs, build index, answer questions.

    This class is the single entry point for the entire system.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            model_name:    Sentence Transformer model for embeddings.
            llm_model:     Ollama model for text generation.
            chunk_size:    Target chunk size in characters.
            chunk_overlap: Overlap between chunks.
            top_k:         Number of chunks to retrieve per query.
        """
        self.retriever = Retriever(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.llm = LLM(model=llm_model)
        self.top_k = top_k
        self.is_indexed = False
        self.document_count = 0
        self.chunk_count = 0

    def load_documents(self, path: str):
        """
        Load PDFs and build the search index.

        Accepts either a single PDF file or a directory of PDFs.

        Args:
            path: Path to a PDF file or a directory containing PDFs.
        """
        import os

        print("=" * 60)
        print("Loading documents...")
        print("=" * 60)

        # Load pages from PDF(s)
        if os.path.isdir(path):
            pages = load_pdfs_from_directory(path)
        else:
            pages = load_pdf(path)

        if not pages:
            print("[!] No pages extracted. Check your PDF files.")
            return

        self.document_count = len(set(p["source"] for p in pages))

        # Build the FAISS index (chunk -> embed -> store)
        self.retriever.build_index(pages)
        self.chunk_count = self.retriever.vector_store.size
        self.is_indexed = True

        print(f"\n{'='*60}")
        print(f"[OK] Ready! {self.document_count} document(s), {self.chunk_count} chunks indexed")
        print(f"{'='*60}")

    def ask(self, question: str, top_k: int = None) -> dict:
        """
        Ask a question and get an answer based on the loaded documents.

        This is the main method — the whole RAG pipeline in one call:
          1. Retrieve relevant chunks
          2. Format them as context
          3. Send to LLM with the question
          4. Return the answer with sources

        Args:
            question: The user's question.
            top_k:    Override the default number of chunks to retrieve.

        Returns:
            A dict with:
            {
                "answer": "The LLM's answer...",
                "sources": [
                    {"text": "chunk...", "score": 0.85,
                     "source": "file.pdf", "page_number": 2},
                    ...
                ],
                "question": "The original question"
            }
        """
        if not self.is_indexed:
            return {
                "answer": "No documents loaded. Use load_documents() first.",
                "sources": [],
                "question": question,
            }

        k = top_k or self.top_k

        # Step 1: Retrieve relevant chunks
        results = self.retriever.search(question, top_k=k)

        if not results:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "question": question,
            }

        # Step 2: Format context for the LLM
        context = self.retriever.format_results(results)

        # Step 3: Generate answer
        answer = self.llm.generate(question=question, context=context)

        # Retain sources that meet a baseline relevance score threshold
        relevant_sources = [r for r in results if r.get("score", 0) >= 0.18]

        return {
            "answer": answer,
            "sources": relevant_sources,
            "question": question,
        }

    def save_index(self, path: str):
        """Save the current index to disk for later reuse."""
        self.retriever.save(path)

    def load_index(self, path: str):
        """Load a previously saved index (skips PDF processing)."""
        self.retriever = Retriever.load(path)
        self.chunk_count = self.retriever.vector_store.size
        self.is_indexed = True
        print(f"[OK] Index loaded: {self.chunk_count} chunks")

    def status(self) -> dict:
        """Return current pipeline status."""
        return {
            "indexed": self.is_indexed,
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "embedding_model": self.retriever.embedder.model_name,
            "llm_model": self.llm.model,
            "llm_available": self.llm.is_available(),
        }

    def format_answer(self, result: dict) -> str:
        """
        Format an ask() result into a readable string.

        Args:
            result: The dict returned by ask().

        Returns:
            A formatted string with the answer and sources.
        """
        lines = []
        lines.append(f"Question: {result['question']}")
        lines.append("")
        lines.append(f"Answer: {result['answer']}")
        lines.append("")

        if result["sources"]:
            lines.append("Sources:")
            seen = set()
            for s in result["sources"]:
                key = f"{s['source']} p.{s['page_number']}"
                if key not in seen:
                    lines.append(f"  - {key} (score: {s['score']:.4f})")
                    seen.add(key)

        return "\n".join(lines)


# --- Interactive demo when run directly ---
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("RAG Pipeline -- Interactive Demo")
    print("=" * 60)

    # Determine PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "data/documents"

    # Initialize pipeline
    rag = RAGPipeline()

    # Check LLM availability
    if not rag.llm.is_available():
        print("\n[!] Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    # Load documents
    rag.load_documents(pdf_path)

    if not rag.is_indexed:
        print("\n[!] No documents were indexed. Add PDFs to data/documents/")
        sys.exit(1)

    # Interactive Q&A loop
    print("\nAsk questions about your documents (type 'quit' to exit):\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        print("\nThinking...\n")
        result = rag.ask(question)
        print(rag.format_answer(result))
        print()
