from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.array([])
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True)


def semantic_similarity_matrix(list_a: list[str], list_b: list[str]) -> np.ndarray:
    """Returns an (len(list_a) x len(list_b)) cosine similarity matrix."""
    if not list_a or not list_b:
        return np.zeros((len(list_a), len(list_b)))
    emb_a = embed_texts(list_a)
    emb_b = embed_texts(list_b)
    return cosine_similarity(emb_a, emb_b)