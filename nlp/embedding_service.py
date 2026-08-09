"""
Embedding generation and FAISS semantic search service.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config import settings


# Load resources once
@lru_cache(maxsize=1)
def _load_resources():
    """
    Load SentenceTransformer, FAISS index and review dataframe once.
    Cached using lru_cache so they are not reloaded on every request.
    """

    import faiss
    from sentence_transformers import SentenceTransformer

    print("=" * 80)
    print("Loading resources...")
    print("=" * 80)

    # Load embedding model
    print("Loading SentenceTransformer...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    print("✓ Model loaded")

    # Load FAISS index
    print("\nLoading FAISS index...")

    if not os.path.exists(settings.INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {settings.INDEX_PATH}\n"
            "Run: python scripts/build_index.py"
        )

    index = faiss.read_index(settings.INDEX_PATH)

    print(" FAISS loaded")
    print(f"Total vectors: {index.ntotal:,}")

    # Load dataframe
    print("\nLoading review dataframe...")

    if not os.path.exists(settings.DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {settings.DATA_PATH}"
        )

    df = pd.read_parquet(settings.DATA_PATH)

    print("DataFrame loaded")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {list(df.columns)}")

    print("=" * 80)

    return model, index, df


# Semantic Search
def semantic_search(
    query: str,
    top_k: int = 10,
    platform_filter: Optional[str] = None,
    sentiment_filter: Optional[str] = None,
    min_rating: float = 1.0,
) -> list[dict]:
    """
    Perform semantic search over customer reviews.

    Args
    ----
    query:
        Natural language search query.

    top_k:
        Number of reviews to return.

    platform_filter:
        amazon | yelp | twitter_airline

    sentiment_filter:
        positive | neutral | negative

    min_rating:
        Minimum rating.

    Returns
    -------
    List[dict]
    """

    if not query.strip():
        return []

    model, index, df = _load_resources()

    print("\nEncoding query...")
    print("Query:", query)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    print("Embedding shape:", query_embedding.shape)

    # Search FAISS
    fetch_k = min(top_k * 5, index.ntotal)

    print("\nSearching FAISS...")
    print(f"Fetching {fetch_k} candidates")

    scores, indices = index.search(
        query_embedding,
        fetch_k
    )

    print("\nTop Similarity Scores")
    print(scores[0][:10])

    print("\nTop Indices")
    print(indices[0][:10])

    results = []

    print("\nChecking retrieved reviews...\n")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        print(f"Candidate {rank}")
        print("Index      :", idx)
        print("Similarity :", round(float(score), 4))

        if idx < 0 or idx >= len(df):
            print(" Invalid dataframe index")
            continue

        row = df.iloc[idx]

        print("Platform   :", row["platform"])
        print("Sentiment  :", row["sentiment_label"])
        print("Rating     :", row["rating"])

        # Platform filter
        if platform_filter:

            if row["platform"] != platform_filter:
                print(" Platform filter failed")
                continue

        # Sentiment filter
        if sentiment_filter:

            if row["sentiment_label"] != sentiment_filter:
                print(" Sentiment filter failed")
                continue

        # Rating filter
        if row["rating"] < min_rating:
            print(" Rating filter failed")
            continue

        print(" Accepted")

        results.append(
            {
                "review_id": row["review_id"],
                "text": row["text"],
                "platform": row["platform"],
                "rating": row["rating"],
                "sentiment_label": row["sentiment_label"],
                "similarity_score": round(float(score), 4),
            }
        )

        print("-" * 80)

        if len(results) >= top_k:
            break

    print("\nFinished")
    print(f"Returned {len(results)} reviews")

    return results


# Embed single text
def embed_text(text: str) -> np.ndarray:
    """
    Convert a single text into a normalized embedding.
    Used by FastAPI analyse endpoint.
    """

    model, _, _ = _load_resources()

    return model.encode([text],normalize_embeddings=True)[0]