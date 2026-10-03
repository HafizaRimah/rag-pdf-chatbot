"""
test_pdf_loader.py — Test the PDF loader module.

Creates a temporary test PDF, then verifies that:
  1. load_pdf() extracts text correctly
  2. Page numbers and source metadata are correct
  3. load_pdfs_from_directory() finds and loads PDFs
  4. Error handling works for missing files and non-PDFs
"""

import os
import sys
import tempfile
import pymupdf  # PyMuPDF — used to CREATE the test PDF

# Add project root to path so we can import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_loader import load_pdf, load_pdfs_from_directory


def create_test_pdf(file_path: str, pages_text: list[str]):
    """Helper: create a simple PDF with the given text on each page."""
    doc = pymupdf.open()  # New empty PDF

    for text in pages_text:
        page = doc.new_page()  # A4 size by default
        # Insert text at position (72, 72) — 1 inch from top-left
        page.insert_text((72, 72), text, fontsize=12)

    doc.save(file_path)
    doc.close()


def test_load_single_pdf():
    """Test: load a single PDF and check extracted content."""
    print("Test 1: Load a single PDF")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, "test_doc.pdf")

        # Create a 3-page PDF
        create_test_pdf(pdf_path, [
            "This is page one about Python programming.",
            "This is page two about machine learning.",
            "This is page three about neural networks.",
        ])

        # Load it
        pages = load_pdf(pdf_path)

        # Check results
        assert len(pages) == 3, f"Expected 3 pages, got {len(pages)}"
        assert pages[0]["page_number"] == 1
        assert pages[1]["page_number"] == 2
        assert pages[2]["page_number"] == 3
        assert pages[0]["source"] == "test_doc.pdf"
        assert "Python" in pages[0]["text"]
        assert "machine learning" in pages[1]["text"]
        assert "neural networks" in pages[2]["text"]

        print(f"  [OK] Loaded {len(pages)} pages")
        print(f"  [OK] Page numbers correct: {[p['page_number'] for p in pages]}")
        print(f"  [OK] Source metadata correct: {pages[0]['source']}")
        print(f"  [OK] Text content extracted correctly")
    print()


def test_load_directory():
    """Test: load all PDFs from a directory."""
    print("Test 2: Load PDFs from a directory")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create two PDFs
        create_test_pdf(
            os.path.join(tmp_dir, "doc_a.pdf"),
            ["Document A, page 1.", "Document A, page 2."]
        )
        create_test_pdf(
            os.path.join(tmp_dir, "doc_b.pdf"),
            ["Document B, page 1."]
        )

        # Also create a non-PDF file (should be ignored)
        with open(os.path.join(tmp_dir, "notes.txt"), "w") as f:
            f.write("This is not a PDF")

        # Load all
        pages = load_pdfs_from_directory(tmp_dir)

        assert len(pages) == 3, f"Expected 3 total pages, got {len(pages)}"
        sources = set(p["source"] for p in pages)
        assert sources == {"doc_a.pdf", "doc_b.pdf"}, f"Wrong sources: {sources}"

        print(f"  [OK] Loaded {len(pages)} pages from {len(sources)} PDFs")
        print(f"  [OK] Non-PDF files were ignored")
    print()


def test_error_handling():
    """Test: proper errors for bad input."""
    print("Test 3: Error handling")
    print("-" * 40)

    # Missing file
    try:
        load_pdf("nonexistent.pdf")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  [OK] FileNotFoundError raised for missing file")

    # Non-PDF file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not a pdf")
        txt_path = f.name

    try:
        load_pdf(txt_path)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [OK] ValueError raised for non-PDF file")
    finally:
        os.unlink(txt_path)

    # Missing directory
    try:
        load_pdfs_from_directory("nonexistent_dir")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  [OK] FileNotFoundError raised for missing directory")

    print()


def test_empty_directory():
    """Test: empty directory returns empty list."""
    print("Test 4: Empty directory")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pages = load_pdfs_from_directory(tmp_dir)
        assert pages == [], f"Expected empty list, got {pages}"
        print("  [OK] Empty directory returns empty list")
    print()


# --- Run all tests ---
if __name__ == "__main__":
    print("=" * 60)
    print("PDF Loader -- Tests")
    print("=" * 60)
    print()

    test_load_single_pdf()
    test_load_directory()
    test_error_handling()
    test_empty_directory()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
