"""
vector_store.py — Store and search embeddings using FAISS.

FAISS (Facebook AI Similarity Search) is a library for efficient
similarity search of dense vectors. It lets us:
  1. Store thousands of embedding vectors in an index
  2. Quickly find the most similar vectors to a query
  3. Save/load the index to/from disk

How it works:
  - We use IndexFlatIP (Inner Product) which computes dot-product
    similarity. Since our Sentence Transformer vectors are already
    normalized (unit length), dot product = cosine similarity.
  - For a query vector, FAISS scans all stored vectors and returns
    the top-k most similar ones — much faster than a Python loop.

Usage:
    from src.vector_store import VectorStore

    store = VectorStore(dimension=384)
    store.add(embeddings, chunks)
    results = store.search(query_vector, top_k=5)
    store.save("my_index")
    store = VectorStore.load("my_index")
"""

import json
import os

import faiss
import numpy as np


class VectorStore:
    """
    A FAISS-based vector store for semantic search.

    Stores embedding vectors alongside their chunk metadata,
    and provides fast similarity search.
    """

    def __init__(self, dimension: int = 384):
        """
        Create a new empty vector store.

        Args:
            dimension: Size of embedding vectors (384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension

        # Create a FAISS index using Inner Product (dot product) similarity.
        # Since our embeddings are L2-normalized by Sentence Transformers,
        # inner product equals cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        # Store chunk metadata alongside the vectors.
        # metadata[i] corresponds to the i-th vector in the FAISS index.
        self.metadata = []

    def add(self, embeddings: np.ndarray, chunks: list[dict]):
        """
        Add embedding vectors and their corresponding chunk metadata.

        Args:
            embeddings: NumPy array of shape (N, dimension).
            chunks:     List of chunk dicts (from chunker.py).
                        Must have same length as embeddings.

        Raises:
            ValueError: If embeddings and chunks have mismatched lengths.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
            )

        if len(embeddings) == 0:
            return

        # Ensure embeddings are float32 (FAISS requires this)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize vectors so dot product = cosine similarity
        faiss.normalize_L2(embeddings)

        # Add vectors to the FAISS index
        self.index.add(embeddings)

        # Store the corresponding metadata
        self.metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search for the most similar chunks to a query embedding.

        Args:
            query_embedding: A single embedding vector of shape (dimension,).
            top_k:           Number of top results to return.

        Returns:
            A list of result dicts, ranked by similarity:
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
        if self.index.ntotal == 0:
            return []

        # Reshape to (1, dimension) for FAISS and ensure float32
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # Don't request more results than we have vectors
        k = min(top_k, self.index.ntotal)

        # Search — returns distances (similarity scores) and indices
        scores, indices = self.index.search(query, k)

        # Build result list
        results = []
        for i in range(k):
            idx = indices[0][i]
            score = float(scores[0][i])

            if idx == -1:  # FAISS returns -1 for missing results
                continue

            result = {
                **self.metadata[idx],  # Copy all chunk metadata
                "score": score,
            }
            results.append(result)

        return results

    def save(self, path: str):
        """
        Save the vector store (index + metadata) to disk.

        Creates two files:
          - {path}.faiss  — the FAISS index
          - {path}.json   — the chunk metadata

        Args:
            path: Base path (without extension).
        """
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")

        # Save metadata as JSON
        with open(f"{path}.json", "w", encoding="utf-8") as f:
            json.dump({
                "dimension": self.dimension,
                "metadata": self.metadata,
            }, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved vector store: {self.index.ntotal} vectors")
        print(f"     Index: {path}.faiss")
        print(f"     Metadata: {path}.json")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """
        Load a vector store from disk.

        Args:
            path: Base path (without extension).
                  Expects {path}.faiss and {path}.json files.

        Returns:
            A VectorStore instance with the loaded data.

        Raises:
            FileNotFoundError: If the files don't exist.
        """
        faiss_path = f"{path}.faiss"
        json_path = f"{path}.json"

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Metadata not found: {json_path}")

        # Load metadata
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create a new store and load into it
        store = cls(dimension=data["dimension"])
        store.index = faiss.read_index(faiss_path)
        store.metadata = data["metadata"]

        print(f"[OK] Loaded vector store: {store.index.ntotal} vectors")

        return store

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self.index.ntotal


# --- Quick demo when run directly ---
if __name__ == "__main__":
    # Create some fake embeddings for demonstration
    print("Vector Store Demo")
    print("=" * 50)

    dimension = 384
    store = VectorStore(dimension=dimension)

    # Simulate 5 chunk embeddings (random, for demo only)
    np.random.seed(42)
    fake_embeddings = np.random.randn(5, dimension).astype(np.float32)

    fake_chunks = [
        {"text": "Python is great.", "chunk_id": 0, "page_number": 1, "source": "demo.pdf"},
        {"text": "ML uses data.", "chunk_id": 1, "page_number": 1, "source": "demo.pdf"},
        {"text": "Paris is in France.", "chunk_id": 2, "page_number": 2, "source": "demo.pdf"},
        {"text": "Neural nets learn.", "chunk_id": 3, "page_number": 2, "source": "demo.pdf"},
        {"text": "FAISS is fast.", "chunk_id": 4, "page_number": 3, "source": "demo.pdf"},
    ]

    # Add to store
    store.add(fake_embeddings, fake_chunks)
    print(f"\nStored {store.size} vectors")

    # Search with a random query
    query = np.random.randn(dimension).astype(np.float32)
    results = store.search(query, top_k=3)

    print(f"\nTop 3 results:")
    for r in results:
        print(f"  {r['score']:.4f}  ->  {r['text']} (page {r['page_number']})")
