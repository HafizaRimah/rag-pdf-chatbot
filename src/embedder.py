"""
embedder.py — Generate embeddings using Sentence Transformers.

This module wraps the Sentence Transformer model to convert text
chunks into dense vector embeddings suitable for semantic search.

How it works:
  1. Load a pre-trained model (default: all-MiniLM-L6-v2)
  2. Pass text strings to the model
  3. Get back fixed-size vectors (384 dimensions for MiniLM)

These vectors capture the "meaning" of each text chunk.
Chunks with similar meaning will have vectors that are close
together in the 384-dimensional space.

Usage:
    from src.embedder import Embedder

    embedder = Embedder()
    vectors = embedder.embed_texts(["Hello world", "Hi there"])
    query_vector = embedder.embed_query("greeting")
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper around a Sentence Transformer model for generating embeddings.

    Why a class instead of bare functions?
      The model takes a few seconds to load. By putting it in a class,
      we load it once and reuse it for many calls — much faster than
      reloading the model every time we embed something.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Load the embedding model.

        Args:
            model_name: Name of the Sentence Transformer model.
                        Default "all-MiniLM-L6-v2" is a good balance
                        of speed and quality (384-dim vectors).
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text strings.

        Used for embedding document chunks (the "database" side).

        Args:
            texts: List of text strings to embed.

        Returns:
            NumPy array of shape (len(texts), dimension).
            Each row is the embedding vector for one text.
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 10,  # Show progress for large batches
            convert_to_numpy=True,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a single query string.

        Used for embedding the user's question (the "search" side).

        Args:
            query: The query string to embed.

        Returns:
            NumPy array of shape (dimension,) — a single vector.
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding

    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        """
        Embed the text from a list of chunk dicts (from chunker.py).

        Convenience method that extracts the "text" field from each
        chunk dict and embeds them all at once.

        Args:
            chunks: List of chunk dicts (must have a "text" key).

        Returns:
            NumPy array of shape (len(chunks), dimension).
        """
        texts = [chunk["text"] for chunk in chunks]
        return self.embed_texts(texts)


# --- Quick demo when run directly ---
if __name__ == "__main__":
    # Create the embedder
    embedder = Embedder()

    # Some sample texts
    texts = [
        "Python is a programming language.",
        "Machine learning uses algorithms to learn from data.",
        "The Eiffel Tower is located in Paris, France.",
        "Deep learning is a subset of machine learning.",
    ]

    # Embed the texts
    print(f"\nEmbedding {len(texts)} texts...")
    embeddings = embedder.embed_texts(texts)
    print(f"Result shape: {embeddings.shape}")
    print(f"  -> {len(texts)} texts, each as a {embedder.dimension}-dim vector")

    # Embed a query
    query = "What is deep learning?"
    print(f"\nQuery: '{query}'")
    query_embedding = embedder.embed_query(query)
    print(f"Query vector shape: {query_embedding.shape}")

    # Quick similarity check (cosine similarity)
    print("\nSimilarity scores:")
    for i, text in enumerate(texts):
        score = np.dot(query_embedding, embeddings[i]) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embeddings[i])
        )
        print(f"  {score:.4f}  ->  {text}")
