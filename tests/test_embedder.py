"""
test_embedder.py — Test the embedding module.

Verifies that:
  1. The model loads and produces correct-shaped vectors
  2. Batch embedding works for multiple texts
  3. Single query embedding works
  4. Similar texts produce similar vectors (semantic check)
  5. embed_chunks() works with chunk dicts
  6. Edge cases are handled
"""

import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import Embedder


# Load the model once for all tests
print("Setting up: loading model (this may take a moment)...\n")
embedder = Embedder()


def cosine_similarity(a, b):
    """Helper: compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_model_loads():
    """Test: model loads and reports correct dimension."""
    print("Test 1: Model loaded correctly")
    print("-" * 40)

    assert embedder.model is not None, "Model should not be None"
    assert embedder.dimension == 384, f"Expected 384 dims, got {embedder.dimension}"
    assert embedder.model_name == "all-MiniLM-L6-v2"

    print(f"  [OK] Model: {embedder.model_name}")
    print(f"  [OK] Dimension: {embedder.dimension}")
    print()


def test_embed_texts():
    """Test: batch embedding returns correct shape."""
    print("Test 2: Batch embedding (embed_texts)")
    print("-" * 40)

    texts = [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]

    embeddings = embedder.embed_texts(texts)

    assert isinstance(embeddings, np.ndarray), "Should return NumPy array"
    assert embeddings.shape == (3, 384), f"Expected (3, 384), got {embeddings.shape}"

    # Vectors should not be all zeros
    for i in range(len(texts)):
        assert np.linalg.norm(embeddings[i]) > 0, f"Vector {i} is all zeros"

    print(f"  [OK] Shape: {embeddings.shape}")
    print(f"  [OK] All vectors non-zero")
    print()


def test_embed_query():
    """Test: single query embedding returns correct shape."""
    print("Test 3: Query embedding (embed_query)")
    print("-" * 40)

    query = "What is machine learning?"
    embedding = embedder.embed_query(query)

    assert isinstance(embedding, np.ndarray), "Should return NumPy array"
    assert embedding.shape == (384,), f"Expected (384,), got {embedding.shape}"
    assert np.linalg.norm(embedding) > 0, "Vector should not be all zeros"

    print(f"  [OK] Shape: {embedding.shape}")
    print(f"  [OK] Vector non-zero (norm: {np.linalg.norm(embedding):.4f})")
    print()


def test_semantic_similarity():
    """Test: similar texts should have higher cosine similarity."""
    print("Test 4: Semantic similarity check")
    print("-" * 40)

    texts = [
        "Dogs are loyal pets.",           # About dogs
        "Cats are independent animals.",   # About cats (related)
        "Python is a programming language.", # Unrelated
    ]

    embeddings = embedder.embed_texts(texts)
    query = embedder.embed_query("Tell me about pet animals")

    # Compute similarities
    scores = []
    for i, text in enumerate(texts):
        score = cosine_similarity(query, embeddings[i])
        scores.append(score)
        print(f"  {score:.4f}  ->  {text}")

    # "Dogs" and "Cats" should score higher than "Python programming"
    assert scores[0] > scores[2], "Dogs should be more similar to pets than Python"
    assert scores[1] > scores[2], "Cats should be more similar to pets than Python"

    print(f"\n  [OK] Pet-related texts score higher than unrelated text")
    print()


def test_embed_chunks():
    """Test: embed_chunks works with chunk dicts from chunker."""
    print("Test 5: Embedding chunk dicts (embed_chunks)")
    print("-" * 40)

    chunks = [
        {"text": "First chunk about AI.", "chunk_id": 0, "page_number": 1, "source": "test.pdf"},
        {"text": "Second chunk about ML.", "chunk_id": 1, "page_number": 1, "source": "test.pdf"},
    ]

    embeddings = embedder.embed_chunks(chunks)

    assert embeddings.shape == (2, 384), f"Expected (2, 384), got {embeddings.shape}"

    print(f"  [OK] Shape: {embeddings.shape}")
    print()


def test_empty_input():
    """Test: empty input returns empty array."""
    print("Test 6: Empty input")
    print("-" * 40)

    result = embedder.embed_texts([])
    assert isinstance(result, np.ndarray), "Should return NumPy array"
    assert result.size == 0, "Should be empty"

    print("  [OK] Empty list returns empty array")
    print()


# --- Run all tests ---
if __name__ == "__main__":
    print("=" * 60)
    print("Embedder -- Tests")
    print("=" * 60)
    print()

    test_model_loads()
    test_embed_texts()
    test_embed_query()
    test_semantic_similarity()
    test_embed_chunks()
    test_empty_input()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
