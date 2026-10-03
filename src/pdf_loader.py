"""
pdf_loader.py — Extract text from PDF files using PyMuPDF.

This module handles:
  1. Loading a single PDF file
  2. Extracting text from each page
  3. Returning structured data (text + metadata per page)
  4. Loading all PDFs from a directory

Usage:
    from src.pdf_loader import load_pdf, load_pdfs_from_directory

    pages = load_pdf("data/documents/myfile.pdf")
    all_pages = load_pdfs_from_directory("data/documents")
"""

import os
import pymupdf  # PyMuPDF (previously imported as 'fitz')


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a single PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dicts, one per page:
        [
            {
                "text": "page content...",
                "page_number": 1,
                "source": "myfile.pdf"
            },
            ...
        ]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """

    # --- Validate input ---
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF file: {file_path}")

    # --- Open and extract ---
    doc = pymupdf.open(file_path)
    file_name = os.path.basename(file_path)

    pages = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text()  # Extract plain text from this page

        # Skip pages that have no meaningful text
        if text.strip():
            pages.append({
                "text": text,
                "page_number": page_number + 1,  # 1-indexed for humans
                "source": file_name,
            })

    doc.close()

    return pages


def load_pdfs_from_directory(directory: str) -> list[dict]:
    """
    Load and extract text from all PDF files in a directory.

    Args:
        directory: Path to the directory containing PDF files.

    Returns:
        A combined list of page dicts from all PDFs found.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all PDF files (non-recursive — only top-level)
    pdf_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"[!] No PDF files found in: {directory}")
        return []

    # Load each PDF
    all_pages = []

    for pdf_path in pdf_files:
        print(f"  Loading: {os.path.basename(pdf_path)}")
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    print(f"\n[OK] Loaded {len(all_pages)} pages from {len(pdf_files)} PDF(s)")

    return all_pages


# --- Quick test when run directly ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.pdf_loader <path_to_pdf>")
        print("   or: python -m src.pdf_loader <directory>")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        pages = load_pdfs_from_directory(target)
    else:
        pages = load_pdf(target)

    # Show results
    for page in pages:
        print(f"\n{'='*60}")
        print(f"📄 {page['source']} — Page {page['page_number']}")
        print(f"{'='*60}")
        # Show first 500 chars as preview
        preview = page["text"][:500]
        print(preview)
        if len(page["text"]) > 500:
            print(f"\n... ({len(page['text'])} total characters)")
