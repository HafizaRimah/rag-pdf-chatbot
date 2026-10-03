"""
chunker.py — Split extracted PDF text into overlapping chunks.

Why chunking?
  Embedding models have a limited input size (e.g. 256 tokens for
  all-MiniLM-L6-v2). Whole pages are often too long, so we split
  them into smaller, overlapping pieces.

  Overlap ensures that ideas spanning a chunk boundary are not lost.

Usage:
    from src.chunker import chunk_pages

    pages = load_pdf("myfile.pdf")       # from pdf_loader
    chunks = chunk_pages(pages)          # list of chunk dicts
"""


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split a text string into overlapping chunks.

    Strategy:
      1. Split the text into sentences (on '. ', '? ', '! ', newlines).
      2. Accumulate sentences into a chunk until chunk_size is reached.
      3. When starting a new chunk, carry over the last few sentences
         from the previous chunk to create overlap.

    This is better than splitting at a fixed character count because
    it avoids cutting sentences in half.

    Args:
        text:          The text to split.
        chunk_size:    Target size for each chunk (in characters).
        chunk_overlap: How many characters of overlap between chunks.

    Returns:
        A list of text chunks (strings).
    """

    if not text or not text.strip():
        return []

    # --- Step 1: Split into sentences ---
    # We split on common sentence endings followed by a space or newline.
    # This is simple but effective for most PDF text.
    sentences = _split_into_sentences(text)

    if not sentences:
        return []

    # --- Step 2: Build chunks from sentences ---
    chunks = []
    current_chunk_sentences = []
    current_chunk_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence would exceed chunk_size,
        # save the current chunk and start a new one.
        if current_chunk_length + sentence_length > chunk_size and current_chunk_sentences:
            # Save the current chunk
            chunk_text_str = " ".join(current_chunk_sentences)
            chunks.append(chunk_text_str)

            # Start new chunk with overlap:
            # Keep sentences from the end of the previous chunk
            # until we have roughly chunk_overlap characters.
            overlap_sentences = []
            overlap_length = 0

            for s in reversed(current_chunk_sentences):
                if overlap_length + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s)

            current_chunk_sentences = overlap_sentences
            current_chunk_length = overlap_length

        current_chunk_sentences.append(sentence)
        current_chunk_length += sentence_length

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunk_text_str = " ".join(current_chunk_sentences)
        chunks.append(chunk_text_str)

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentence-like segments.

    Splits on:
      - Period + space  (". ")
      - Question mark + space  ("? ")
      - Exclamation + space  ("! ")
      - Newlines  ("\\n")

    Each segment is stripped of extra whitespace.
    Empty segments are dropped.
    """

    # First, normalize newlines into a sentence separator
    text = text.replace("\n", " . ")

    # Split on sentence-ending punctuation followed by space
    segments = []
    current = []

    for char in text:
        current.append(char)

        # Check if we just completed a sentence-ending pattern
        if len(current) >= 2:
            last_two = "".join(current[-2:])
            if last_two in (". ", "? ", "! "):
                # End of a sentence — save it
                sentence = "".join(current).strip()
                if sentence:
                    segments.append(sentence)
                current = []

    # Don't forget remaining text
    remaining = "".join(current).strip()
    if remaining:
        segments.append(remaining)

    return segments


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Chunk all pages from the PDF loader into smaller pieces.

    Takes the output of load_pdf() or load_pdfs_from_directory()
    and splits each page's text into overlapping chunks, while
    preserving metadata.

    Args:
        pages:         List of page dicts from pdf_loader.
        chunk_size:    Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A list of chunk dicts:
        [
            {
                "text": "chunk content...",
                "chunk_id": 0,
                "page_number": 1,
                "source": "myfile.pdf"
            },
            ...
        ]
    """

    all_chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        source = page["source"]

        # Split this page's text into chunks
        text_chunks = chunk_text(text, chunk_size, chunk_overlap)

        for chunk in text_chunks:
            all_chunks.append({
                "text": chunk,
                "chunk_id": chunk_id,
                "page_number": page_number,
                "source": source,
            })
            chunk_id += 1

    return all_chunks


# --- Quick test when run directly ---
if __name__ == "__main__":
    # Demo with sample text
    sample_text = (
        "Python is a popular programming language. It was created by Guido van Rossum. "
        "Python is widely used in web development, data science, and machine learning. "
        "It has a simple and readable syntax that makes it great for beginners. "
        "Many companies like Google, Netflix, and Instagram use Python. "
        "The language supports multiple programming paradigms including procedural, "
        "object-oriented, and functional programming. "
        "Python has a vast ecosystem of libraries and frameworks. "
        "NumPy and Pandas are popular for data analysis. "
        "TensorFlow and PyTorch are used for deep learning. "
        "Flask and Django are common web frameworks."
    )

    print("Original text length:", len(sample_text))
    print()

    chunks = chunk_text(sample_text, chunk_size=200, chunk_overlap=50)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")
        print()

    print(f"Total: {len(chunks)} chunks")
