"""
test_retriever.py — Test the retriever module.

Verifies that:
  1. build_index() creates a searchable index from pages
  2. search() returns semantically relevant results
  3. Results are ranked correctly (most relevant first)
  4. save/load round-trip works
  5. format_results() produces readable output
  6. Edge cases are handled (empty pages, empty query results)
"""

import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import Retriever


# Sample pages simulating pdf_loader output
SAMPLE_PAGES = [
    {
        "text": (
            "Python is a high-level programming language created by Guido van Rossum. "
            "It is widely used in web development, data science, and automation. "
            "Python has a simple syntax that is easy to learn."
        ),
        "page_number": 1,
        "source": "programming.pdf",
    },
    {
        "text": (
            "Machine learning is a subset of artificial intelligence. "
            "It enables computers to learn patterns from data. "
            "Common algorithms include decision trees, neural networks, and SVMs."
        ),
        "page_number": 2,
        "source": "ai_book.pdf",
    },
    {
        "text": (
            "The Eiffel Tower is located in Paris, France. "
            "It was built in 1889 and stands 330 meters tall. "
            "Millions of tourists visit it every year."
        ),
        "page_number": 3,
        "source": "travel.pdf",
    },
]


# Load the model once for all tests
print("Setting up: loading model (this may take a moment)...\n")
retriever = Retriever()


def test_build_index():
    """Test: build_index creates a populated vector store."""
    print("Test 1: Build index from pages")
    print("-" * 40)

    retriever.build_index(SAMPLE_PAGES)

    assert retriever.vector_store.size > 0, "Index should not be empty"
    assert len(retriever.chunks) > 0, "Chunks list should not be empty"

    print(f"  [OK] Index built: {retriever.vector_store.size} vectors")
    print(f"  [OK] Chunks stored: {len(retriever.chunks)}")
    print()


def test_search_relevance():
    """Test: search returns the most relevant chunk for each query."""
    print("Test 2: Search returns relevant results")
    print("-" * 40)

    # Query about Python should match the programming page
    results = retriever.search("What programming language did Guido create?", top_k=3)

    assert len(results) > 0, "Should return results"
    assert "programming.pdf" in results[0]["source"], (
        f"Expected programming.pdf first, got {results[0]['source']}"
    )

    print(f"  Query: 'What programming language did Guido create?'")
    print(f"  Top result: {results[0]['source']} (score: {results[0]['score']:.4f})")
    print(f"  [OK] Correct source ranked first")
    print()


def test_search_ranking():
    """Test: results are ordered by descending score."""
    print("Test 3: Results are ranked by score (descending)")
    print("-" * 40)

    results = retriever.search("artificial intelligence and learning", top_k=3)

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"Scores not in descending order: {scores}"
    )

    for r in results:
        print(f"  {r['score']:.4f}  ->  {r['source']}, p.{r['page_number']}")

    print(f"  [OK] Scores are in descending order")
    print()


def test_save_and_load():
    """Test: saved retriever can be loaded and searched."""
    print("Test 4: Save and load round-trip")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test_retriever_index")

        # Save
        retriever.save(path)

        # Load into a new retriever
        loaded = Retriever.load(path)

        assert loaded.vector_store.size == retriever.vector_store.size

        # Search should work on the loaded retriever
        results = loaded.search("Tell me about Paris", top_k=2)
        assert len(results) > 0
        assert results[0]["source"] == "travel.pdf"

        print(f"  [OK] Loaded retriever: {loaded.vector_store.size} vectors")
        print(f"  [OK] Search works after loading")
        print(f"  [OK] Correct result: {results[0]['source']}")
    print()


def test_format_results():
    """Test: format_results produces a readable string."""
    print("Test 5: Format results")
    print("-" * 40)

    results = retriever.search("Eiffel Tower height", top_k=2)
    formatted = retriever.format_results(results)

    assert isinstance(formatted, str)
    assert len(formatted) > 0
    assert "Result 1" in formatted
    assert "score:" in formatted

    # Empty results
    empty_formatted = retriever.format_results([])
    assert "No relevant results" in empty_formatted

    print(f"  [OK] Formatted output ({len(formatted)} chars)")
    print(f"  [OK] Empty results handled")
    print()


def test_empty_pages():
    """Test: building index from empty pages doesn't crash."""
    print("Test 6: Empty pages")
    print("-" * 40)

    empty_retriever = Retriever()
    empty_retriever.build_index([])

    assert empty_retriever.vector_store.size == 0
    results = empty_retriever.search("anything", top_k=3)
    assert results == []

    print("  [OK] Empty pages produce empty index")
    print("  [OK] Search on empty index returns []")
    print()


# --- Run all tests ---
if __name__ == "__main__":
    print("=" * 60)
    print("Retriever -- Tests")
    print("=" * 60)
    print()

    test_build_index()
    test_search_relevance()
    test_search_ranking()
    test_save_and_load()
    test_format_results()
    test_empty_pages()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
