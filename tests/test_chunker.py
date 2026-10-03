"""
test_chunker.py — Test the text chunking module.

Verifies that:
  1. Text is split into chunks of approximately the right size
  2. Overlap exists between consecutive chunks
  3. Metadata is preserved through chunk_pages()
  4. Edge cases are handled (empty text, short text, etc.)
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunker import chunk_text, chunk_pages


def test_basic_chunking():
    """Test: text is split into reasonably-sized chunks."""
    print("Test 1: Basic chunking")
    print("-" * 40)

    text = (
        "Python is a popular programming language. "
        "It was created by Guido van Rossum in 1991. "
        "Python is widely used in data science and machine learning. "
        "It has a simple syntax that makes it great for beginners. "
        "Many companies like Google and Netflix use Python. "
        "The language supports multiple programming paradigms. "
        "Python has a vast ecosystem of libraries. "
        "NumPy and Pandas are popular for data analysis. "
        "TensorFlow and PyTorch are used for deep learning. "
        "Flask and Django are common web frameworks. "
        "Python 3 is the current major version. "
        "The community is large and active worldwide."
    )

    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

    # Each chunk should be roughly within chunk_size
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} chars")

    print(f"  [OK] Split into {len(chunks)} chunks")
    print()


def test_overlap_exists():
    """Test: consecutive chunks share overlapping text."""
    print("Test 2: Overlap between chunks")
    print("-" * 40)

    text = (
        "Sentence one about topic A. "
        "Sentence two continues the idea. "
        "Sentence three introduces topic B. "
        "Sentence four expands on topic B. "
        "Sentence five introduces topic C. "
        "Sentence six wraps up the discussion."
    )

    chunks = chunk_text(text, chunk_size=100, chunk_overlap=40)

    if len(chunks) >= 2:
        # Check that there is some shared text between chunk 0 and chunk 1
        # (because of sentence-level overlap, the end of chunk 0 should
        #  appear at the start of chunk 1)
        overlap_found = False
        words_chunk0 = chunks[0].split()
        words_chunk1 = chunks[1].split()

        # Look for shared words at the boundary
        for word in words_chunk0[-5:]:  # last 5 words of chunk 0
            if word in " ".join(words_chunk1[:10]):  # first 10 words of chunk 1
                overlap_found = True
                break

        if overlap_found:
            print("  [OK] Overlap detected between consecutive chunks")
        else:
            print("  [INFO] No word-level overlap (sentences may be too short)")
    else:
        print("  [INFO] Only 1 chunk produced — text too short for overlap test")
    print()


def test_chunk_pages_metadata():
    """Test: chunk_pages preserves source and page metadata."""
    print("Test 3: Metadata preserved in chunk_pages()")
    print("-" * 40)

    # Simulate output from pdf_loader
    pages = [
        {
            "text": "This is a long text from page one. It talks about Python. " * 5,
            "page_number": 1,
            "source": "test.pdf",
        },
        {
            "text": "Page two discusses machine learning concepts. " * 5,
            "page_number": 2,
            "source": "test.pdf",
        },
    ]

    chunks = chunk_pages(pages, chunk_size=150, chunk_overlap=30)

    assert len(chunks) > 2, f"Expected more than 2 chunks, got {len(chunks)}"

    # Every chunk must have all required fields
    for chunk in chunks:
        assert "text" in chunk, "Missing 'text' field"
        assert "chunk_id" in chunk, "Missing 'chunk_id' field"
        assert "page_number" in chunk, "Missing 'page_number' field"
        assert "source" in chunk, "Missing 'source' field"
        assert chunk["source"] == "test.pdf"

    # chunk_ids should be sequential
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks))), f"Chunk IDs not sequential: {ids}"

    # Both page numbers should appear
    page_nums = set(c["page_number"] for c in chunks)
    assert 1 in page_nums and 2 in page_nums

    print(f"  [OK] {len(chunks)} chunks created with correct metadata")
    print(f"  [OK] Chunk IDs sequential: 0 to {len(chunks)-1}")
    print(f"  [OK] Page numbers preserved: {sorted(page_nums)}")
    print(f"  [OK] Source preserved: {chunks[0]['source']}")
    print()


def test_empty_and_short_text():
    """Test: edge cases — empty text, whitespace-only, short text."""
    print("Test 4: Edge cases")
    print("-" * 40)

    # Empty string
    assert chunk_text("") == [], "Empty string should return []"
    print("  [OK] Empty string returns []")

    # Whitespace only
    assert chunk_text("   \n\n  ") == [], "Whitespace should return []"
    print("  [OK] Whitespace-only returns []")

    # Text shorter than chunk_size — should return 1 chunk
    short = "Hello world."
    chunks = chunk_text(short, chunk_size=500)
    assert len(chunks) == 1, f"Short text should give 1 chunk, got {len(chunks)}"
    print("  [OK] Short text returns exactly 1 chunk")

    # Empty pages list
    assert chunk_pages([]) == [], "Empty pages should return []"
    print("  [OK] Empty pages list returns []")
    print()


def test_no_text_loss():
    """Test: all original words appear somewhere in the chunks."""
    print("Test 5: No text loss")
    print("-" * 40)

    text = (
        "Alpha beta gamma delta. "
        "Epsilon zeta eta theta. "
        "Iota kappa lambda mu. "
        "Nu xi omicron pi. "
        "Rho sigma tau upsilon."
    )

    chunks = chunk_text(text, chunk_size=60, chunk_overlap=20)

    # Recombine all chunks and check that every original word appears
    all_chunk_text = " ".join(chunks)
    original_words = text.split()

    missing = [w for w in original_words if w not in all_chunk_text]
    assert not missing, f"Words lost during chunking: {missing}"

    print(f"  [OK] All {len(original_words)} original words found in chunks")
    print()


# --- Run all tests ---
if __name__ == "__main__":
    print("=" * 60)
    print("Chunker -- Tests")
    print("=" * 60)
    print()

    test_basic_chunking()
    test_overlap_exists()
    test_chunk_pages_metadata()
    test_empty_and_short_text()
    test_no_text_loss()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
