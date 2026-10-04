"""
retriever.py — Semantic retrieval: from a question to relevant chunks.

This module ties together the embedder and vector store into a
single, easy-to-use retrieval pipeline:

    1. Take a user's question (plain text)
    2. Embed it into a vector using the Embedder
    3. Search the VectorStore for the most similar chunks
    4. Return the top-k results with scores and metadata

It also provides a convenience function to build the full index
from raw PDF pages (chunk -> embed -> store).

Usage:
    from src.retriever import Retriever

    # Build the index from PDF pages
    retriever = Retriever()
    retriever.build_index(pages)

    # Search
    results = retriever.search("What is machine learning?", top_k=5)

    # Save / load for reuse
    retriever.save("my_index")
    retriever = Retriever.load("my_index")
"""

from src.chunker import chunk_pages
from src.embedder import Embedder
from src.vector_store import VectorStore


class Retriever:
    """
    Semantic retriever: question in, relevant chunks out.

    Combines chunking, embedding, and vector search into one pipeline.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the retriever.

        Args:
            model_name:    Sentence Transformer model to use.
            chunk_size:    Target chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.embedder = Embedder(model_name=model_name)
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []  # Keep chunks for reference

    def build_index(self, pages: list[dict]):
        """
        Build the search index from PDF pages.

        Pipeline: pages -> chunk -> embed -> store in FAISS

        Args:
            pages: List of page dicts from pdf_loader
                   (each with "text", "page_number", "source").
        """
        # Step 1: Chunk the pages
        print(f"\n--- Chunking {len(pages)} pages ---")
        self.chunks = chunk_pages(
            pages,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        print(f"Created {len(self.chunks)} chunks")

        if not self.chunks:
            print("[!] No chunks created. Is the PDF empty?")
            return

        # Step 2: Embed all chunks
        print(f"\n--- Embedding {len(self.chunks)} chunks ---")
        embeddings = self.embedder.embed_chunks(self.chunks)
        print(f"Embedding shape: {embeddings.shape}")

        # Step 3: Store in FAISS
        print(f"\n--- Storing in vector index ---")
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.vector_store.add(embeddings, self.chunks)
        print(f"Index size: {self.vector_store.size} vectors")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for chunks most relevant to a query.

        Args:
            query: The user's question (plain text string).
            top_k: Number of results to return.

        Returns:
            List of result dicts, ranked by relevance:
            [
                {
                    "text": "chunk content...",
                    "score": 0.85,
                    "chunk_id": 3,
                    "page_number": 2,
                    "source": "myfile.pdf"
                },
                ...
            ]
        """
        if self.vector_store.size == 0:
            print("[!] Index is empty. Call build_index() first.")
            return []

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        # Search the vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results

    def save(self, path: str):
        """
        Save the retriever's index to disk for later reuse.

        Args:
            path: Base path (without extension).
        """
        self.vector_store.save(path)

    @classmethod
    def load(cls, path: str, model_name: str = "all-MiniLM-L6-v2") -> "Retriever":
        """
        Load a previously saved index.

        Args:
            path:       Base path (without extension).
            model_name: Must match the model used when building.

        Returns:
            A Retriever instance ready to search.
        """
        retriever = cls(model_name=model_name)
        retriever.vector_store = VectorStore.load(path)
        retriever.chunks = retriever.vector_store.metadata
        return retriever

    def format_results(self, results: list[dict]) -> str:
        """
        Format search results into a readable string.

        Useful for printing or for passing context to an LLM later.

        Args:
            results: List of result dicts from search().

        Returns:
            A formatted string with all results.
        """
        if not results:
            return "No relevant results found."

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"--- Result {i} (score: {r['score']:.4f}) ---")
            lines.append(f"Source: {r['source']}, Page: {r['page_number']}")
            lines.append(r["text"])
            lines.append("")

        return "\n".join(lines)


# --- Quick demo when run directly ---
if __name__ == "__main__":
    # Simulate the full pipeline with fake pages
    print("=" * 60)
    print("Retriever Demo")
    print("=" * 60)

    # Fake pages (as if from pdf_loader)
    pages = [
        {
            "text": (
                "Python is a high-level programming language. "
                "It was created by Guido van Rossum and released in 1991. "
                "Python emphasizes code readability and simplicity. "
                "It supports multiple programming paradigms."
            ),
            "page_number": 1,
            "source": "demo.pdf",
        },
        {
            "text": (
                "Machine learning is a branch of artificial intelligence. "
                "It allows computers to learn from data without being "
                "explicitly programmed. Supervised learning uses labeled data. "
                "Unsupervised learning finds patterns in unlabeled data."
            ),
            "page_number": 2,
            "source": "demo.pdf",
        },
        {
            "text": (
                "The Eiffel Tower is a wrought-iron lattice tower in Paris. "
                "It was designed by Gustave Eiffel and built between 1887 "
                "and 1889. The tower is 330 meters tall and is the most "
                "visited paid monument in the world."
            ),
            "page_number": 3,
            "source": "demo.pdf",
        },
    ]

    # Build index
    retriever = Retriever()
    retriever.build_index(pages)

    # Search
    queries = [
        "What is Python?",
        "How does machine learning work?",
        "Tell me about the Eiffel Tower",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        results = retriever.search(query, top_k=2)
        print(retriever.format_results(results))
