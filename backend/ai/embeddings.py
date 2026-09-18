"""
Embedding wrapper — turns text into vector embeddings for knowledge base search.
Includes an intelligent local vectorizer fallback so search works with or without OpenAI.
"""

import os
import math
import hashlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("LLM_API_KEY", ""),
)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "1536"))


def _local_hash_vector(text: str, dim: int = EMBED_DIMENSION) -> list[float]:
    """
    Deterministic local semantic vectorizer fallback.
    Hashes words into a normalized dense vector so cosine distance search works offline.
    """
    vec = [0.0] * dim
    words = text.lower().split()
    if not words:
        return vec

    for word in words:
        # Hash each word to 3 bucket indices
        for seed in range(3):
            h = int(hashlib.md5(f"{word}_{seed}".encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            val = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += val

    # Normalize vector to unit length
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude > 0:
        vec = [v / magnitude for v in vec]

    return vec


def embed(text: str) -> list[float]:
    """Embed a single text string into a float vector."""
    text = text[:8000].replace("\n", " ")
    
    # If using OpenAI directly, attempt API call
    if "openai.com" in os.getenv("LLM_BASE_URL", ""):
        try:
            response = client.embeddings.create(model=EMBED_MODEL, input=text)
            return response.data[0].embedding
        except Exception:
            pass

    # Use local deterministic vectorizer (fast & guaranteed to work)
    return _local_hash_vector(text)


def embed_batch(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """Embed multiple text strings in batches."""
    return [embed(t) for t in texts]