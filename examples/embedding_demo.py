from sentence_transformers import SentenceTransformer
import numpy as np


# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Some example document sentences
documents = [
    "Python is a popular programming language.",
    "Machine learning allows computers to learn from data.",
    "Neural networks are inspired by the human brain.",
    "The capital of France is Paris.",
    "Deep learning uses neural networks with multiple layers."
]


# 3. Create embeddings for our documents
document_embeddings = model.encode(documents)


# 4. Our user's question
query = "What is Paris?"


# 5. Create an embedding for the question
query_embedding = model.encode(query)


# 6. Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# 7. Compare the question with every document
scores = []

for document, embedding in zip(documents, document_embeddings):
    score = cosine_similarity(query_embedding, embedding)
    scores.append((document, score))


# 8. Sort from most similar to least similar
scores.sort(key=lambda x: x[1], reverse=True)


# 9. Display results
print("\nQuery:")
print(query)

print("\nMost relevant documents:")

for document, score in scores:
    print(f"{score:.4f}  ->  {document}")