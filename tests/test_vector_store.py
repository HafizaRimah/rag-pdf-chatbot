"""
test_vector_store.py — Test the FAISS vector store module.

Verifies that:
  1. Vectors can be added and the store reports correct size
  2. Search returns results ranked by similarity
  3. Save/load round-trips correctly
  4. Metadata is preserved through add/search
  5. Edge cases are handled (empty store, mismatched lengths)
"""

import os
import sys
import tempfile

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStore


# Dimension matching our embedding model
DIM = 384


def make_normalized_vector(seed: int) -> np.ndarray:
    """Helper: create a deterministic normalized vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v


def test_add_and_size():
    """Test: adding vectors increases the store size."""
    print("Test 1: Add vectors and check size")
    print("-" * 40)

    store = VectorStore(dimension=DIM)
    assert store.size == 0, "New store should be empty"

    # Add 3 vectors
    embeddings = np.random.randn(3, DIM).astype(np.float32)
    chunks = [
        {"text": f"Chunk {i}", "chunk_id": i, "page_number": 1, "source": "test.pdf"}
        for i in range(3)
    ]

    store.add(embeddings, chunks)
    assert store.size == 3, f"Expected 3 vectors, got {store.size}"

    # Add 2 more
    more_embeddings = np.random.randn(2, DIM).astype(np.float32)
    more_chunks = [
        {"text": f"Chunk {i}", "chunk_id": i, "page_number": 2, "source": "test.pdf"}
        for i in range(3, 5)
    ]

    store.add(more_embeddings, more_chunks)
    assert store.size == 5, f"Expected 5 vectors, got {store.size}"

    print(f"  [OK] Store size after adds: {store.size}")
    print()


def test_search_ranking():
    """Test: search returns the most similar vector first."""
    print("Test 2: Search returns correct ranking")
    print("-" * 40)

    store = VectorStore(dimension=DIM)

    # Create a known query vector
    query = make_normalized_vector(seed=0)

    # Create vectors: one identical to query, one opposite, one random
    identical = query.copy()
    opposite = -query
    random_vec = make_normalized_vector(seed=99)

    embeddings = np.array([opposite, random_vec, identical], dtype=np.float32)
    chunks = [
        {"text": "Opposite", "chunk_id": 0, "page_number": 1, "source": "t.pdf"},
        {"text": "Random", "chunk_id": 1, "page_number": 1, "source": "t.pdf"},
        {"text": "Identical", "chunk_id": 2, "page_number": 1, "source": "t.pdf"},
    ]

    store.add(embeddings, chunks)

    results = store.search(query, top_k=3)

    # The identical vector should be first (highest score)
    assert results[0]["text"] == "Identical", f"Expected 'Identical' first, got '{results[0]['text']}'"
    # The opposite vector should be last (lowest score)
    assert results[-1]["text"] == "Opposite", f"Expected 'Opposite' last, got '{results[-1]['text']}'"

    print(f"  Results:")
    for r in results:
        print(f"    {r['score']:.4f}  ->  {r['text']}")
    print(f"  [OK] Correct ranking: Identical > Random > Opposite")
    print()


def test_search_metadata():
    """Test: search results contain all metadata fields."""
    print("Test 3: Metadata preserved in search results")
    print("-" * 40)

    store = VectorStore(dimension=DIM)

    embeddings = np.random.randn(2, DIM).astype(np.float32)
    chunks = [
        {"text": "About AI", "chunk_id": 0, "page_number": 3, "source": "ai_book.pdf"},
        {"text": "About ML", "chunk_id": 1, "page_number": 7, "source": "ml_guide.pdf"},
    ]

    store.add(embeddings, chunks)

    query = np.random.randn(DIM).astype(np.float32)
    results = store.search(query, top_k=2)

    for r in results:
        assert "text" in r, "Missing 'text'"
        assert "score" in r, "Missing 'score'"
        assert "chunk_id" in r, "Missing 'chunk_id'"
        assert "page_number" in r, "Missing 'page_number'"
        assert "source" in r, "Missing 'source'"

    print(f"  [OK] All metadata fields present in results")
    print()


def test_save_and_load():
    """Test: save/load preserves vectors and metadata."""
    print("Test 4: Save and load round-trip")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test_index")

        # Create and populate a store
        store = VectorStore(dimension=DIM)
        np.random.seed(42)
        embeddings = np.random.randn(4, DIM).astype(np.float32)
        chunks = [
            {"text": f"Chunk {i}", "chunk_id": i, "page_number": i + 1, "source": "test.pdf"}
            for i in range(4)
        ]
        store.add(embeddings, chunks)

        # Save
        store.save(path)

        # Verify files exist
        assert os.path.exists(f"{path}.faiss"), "FAISS file not created"
        assert os.path.exists(f"{path}.json"), "JSON file not created"

        # Load into a new store
        loaded = VectorStore.load(path)

        assert loaded.size == store.size, f"Size mismatch: {loaded.size} vs {store.size}"
        assert loaded.dimension == store.dimension
        assert len(loaded.metadata) == len(store.metadata)
        assert loaded.metadata[0]["text"] == "Chunk 0"
        assert loaded.metadata[3]["source"] == "test.pdf"

        # Search should still work on loaded store
        query = np.random.randn(DIM).astype(np.float32)
        results = loaded.search(query, top_k=2)
        assert len(results) == 2

        print(f"  [OK] Saved and loaded {loaded.size} vectors")
        print(f"  [OK] Metadata preserved after reload")
        print(f"  [OK] Search works on loaded store")
    print()


def test_empty_store():
    """Test: empty store returns empty results."""
    print("Test 5: Empty store")
    print("-" * 40)

    store = VectorStore(dimension=DIM)
    query = np.random.randn(DIM).astype(np.float32)
    results = store.search(query, top_k=5)

    assert results == [], f"Expected empty list, got {results}"
    print("  [OK] Empty store returns empty results")
    print()


def test_mismatched_lengths():
    """Test: error when embeddings and chunks have different lengths."""
    print("Test 6: Mismatched lengths")
    print("-" * 40)

    store = VectorStore(dimension=DIM)
    embeddings = np.random.randn(3, DIM).astype(np.float32)
    chunks = [{"text": "Only one", "chunk_id": 0, "page_number": 1, "source": "t.pdf"}]

    try:
        store.add(embeddings, chunks)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [OK] ValueError raised for mismatched lengths")
    print()


# --- Run all tests ---
if __name__ == "__main__":
    print("=" * 60)
    print("Vector Store -- Tests")
    print("=" * 60)
    print()

    test_add_and_size()
    test_search_ranking()
    test_search_metadata()
    test_save_and_load()
    test_empty_store()
    test_mismatched_lengths()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
