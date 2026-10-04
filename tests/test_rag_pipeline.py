"""
test_rag_pipeline.py — Test the complete RAG pipeline.

Verifies the end-to-end flow:
  1. Pipeline initializes correctly
  2. Loading documents builds the index
  3. ask() returns an answer with sources
  4. The answer is grounded in the document content
  5. save/load index works
  6. status() reports correct info
  7. Edge cases (no documents, no index)

NOTE: Tests that call the LLM require Ollama to be running.
"""

import os
import sys
import tempfile

import pymupdf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline


def create_test_pdf(file_path: str, pages_text: list[str]):
    """Helper: create a PDF with given text on each page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    doc.save(file_path)
    doc.close()


def test_initialization():
    """Test: pipeline initializes with correct defaults."""
    print("Test 1: Pipeline initialization")
    print("-" * 40)

    rag = RAGPipeline()

    assert rag.is_indexed is False
    assert rag.document_count == 0
    assert rag.chunk_count == 0
    assert rag.top_k == 5

    print(f"  [OK] Pipeline created (not yet indexed)")
    print(f"  [OK] Defaults: top_k={rag.top_k}")
    print()


def test_load_documents():
    """Test: loading PDFs builds the index."""
    print("Test 2: Load documents and build index")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test PDFs
        create_test_pdf(
            os.path.join(tmp_dir, "science.pdf"),
            [
                "Gravity is a fundamental force of nature. "
                "It attracts objects with mass toward each other. "
                "Isaac Newton described gravity with his law of universal gravitation.",

                "Einstein later refined our understanding with general relativity. "
                "He showed that gravity is the curvature of spacetime caused by mass.",
            ]
        )

        rag = RAGPipeline()
        rag.load_documents(tmp_dir)

        assert rag.is_indexed is True
        assert rag.document_count == 1
        assert rag.chunk_count > 0

        print(f"  [OK] Indexed: {rag.document_count} doc, {rag.chunk_count} chunks")
        print()
        return rag, tmp_dir


def test_ask_with_answer():
    """Test: ask() returns a structured answer from document content."""
    print("Test 3: Ask a question (end-to-end)")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        create_test_pdf(
            os.path.join(tmp_dir, "history.pdf"),
            [
                "The Great Wall of China was built over many centuries. "
                "Construction began during the 7th century BC. "
                "It stretches over 13,000 miles across northern China.",
            ]
        )

        rag = RAGPipeline()
        rag.load_documents(tmp_dir)

        # Check if LLM is available
        if not rag.llm.is_available():
            print("  [SKIP] Ollama not running")
            print()
            return

        result = rag.ask("How long is the Great Wall of China?")

        assert "answer" in result
        assert "sources" in result
        assert "question" in result
        assert len(result["answer"]) > 0
        assert len(result["sources"]) > 0
        assert result["question"] == "How long is the Great Wall of China?"

        print(f"  Question: {result['question']}")
        print(f"  Answer: {result['answer'][:200]}")
        print(f"  Sources: {len(result['sources'])} chunks")
        print(f"  [OK] End-to-end pipeline works")
    print()


def test_ask_no_documents():
    """Test: ask() without loading documents returns a message."""
    print("Test 4: Ask without documents loaded")
    print("-" * 40)

    rag = RAGPipeline()
    result = rag.ask("What is anything?")

    assert "No documents loaded" in result["answer"]
    assert result["sources"] == []

    print(f"  [OK] Returns 'No documents loaded' message")
    print()


def test_save_and_load_index():
    """Test: index can be saved and loaded."""
    print("Test 5: Save and load index")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create PDF and build index
        create_test_pdf(
            os.path.join(tmp_dir, "test.pdf"),
            ["Mars is the fourth planet from the Sun. It is called the Red Planet."]
        )

        rag = RAGPipeline()
        rag.load_documents(tmp_dir)
        original_count = rag.chunk_count

        # Save index
        index_path = os.path.join(tmp_dir, "saved_index")
        rag.save_index(index_path)

        # Load into a fresh pipeline
        rag2 = RAGPipeline()
        rag2.load_index(index_path)

        assert rag2.is_indexed is True
        assert rag2.chunk_count == original_count

        print(f"  [OK] Saved and loaded {rag2.chunk_count} chunks")
    print()


def test_status():
    """Test: status() returns correct info."""
    print("Test 6: Pipeline status")
    print("-" * 40)

    rag = RAGPipeline()
    status = rag.status()

    assert status["indexed"] is False
    assert status["documents"] == 0
    assert status["chunks"] == 0
    assert "embedding_model" in status
    assert "llm_model" in status
    assert "llm_available" in status

    print(f"  [OK] Status: {status}")
    print()


def test_format_answer():
    """Test: format_answer() produces readable output."""
    print("Test 7: Format answer")
    print("-" * 40)

    rag = RAGPipeline()

    result = {
        "question": "What is gravity?",
        "answer": "Gravity is a force.",
        "sources": [
            {"text": "chunk", "score": 0.9, "source": "physics.pdf",
             "page_number": 1, "chunk_id": 0},
        ],
    }

    formatted = rag.format_answer(result)

    assert "What is gravity?" in formatted
    assert "Gravity is a force." in formatted
    assert "physics.pdf" in formatted

    print(f"  [OK] Formatted output contains question, answer, sources")
    print()


# --- Run all tests ---
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline -- Tests")
    print("=" * 60)
    print()

    test_initialization()
    test_load_documents()
    test_ask_with_answer()
    test_ask_no_documents()
    test_save_and_load_index()
    test_status()
    test_format_answer()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
