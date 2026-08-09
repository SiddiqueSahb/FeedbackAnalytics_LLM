"""
Generates sentence embeddings and builds FAISS + LangChain indexes.
Run once after preprocessing. Re-run only when dataset changes.

Usage:
    python scripts/build_index.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json
import numpy as np
import pandas as pd
import faiss

from config import settings

"""
def main():
    #Load data
    if not os.path.exists(settings.DATA_PATH):
        print(f"Data not found at {settings.DATA_PATH}")
        print("    Run: python scripts/preprocess.py first")
        sys.exit(1)

    print(f" Loading data from {settings.DATA_PATH}...")
    df = pd.read_parquet(settings.DATA_PATH)
    texts    = df["cleaned_text"].tolist()
    meta_cols = ["platform", "rating", "sentiment_label", "review_id"]
    if "product_category" in df.columns:
        meta_cols.append("product_category")
    metadata = df[meta_cols].to_dict("records")
    print(f"    Loaded {len(texts):,} reviews")

    # Generate embeddings
    print(f"\n Generating embeddings with {settings.EMBEDDING_MODEL}...")
    print("    This may take a few minutes...")
    from sentence_transformers import SentenceTransformer
    model      = SentenceTransformer(settings.EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalise for cosine similarity
    )
    print(f"    Embedding shape: {embeddings.shape}")

    os.makedirs("data/embeddings", exist_ok=True)

    #Save raw embeddings
    np.save(settings.EMBED_PATH, embeddings)
    print(f"\n Embeddings saved → {settings.EMBED_PATH}")
"""
def main():

    # Path to the already-generated embeddings.
    embedding_path = settings.EMBED_PATH

    # Check that the embeddings exist.
    if not os.path.exists(embedding_path):
        print(f" Embeddings not found at {embedding_path}")
        print("Run build_index.py first to generate embeddings.")
        return

    # Load the existing embeddings.
    print(f" Loading embeddings from {embedding_path}...")

    embeddings = np.load(
        embedding_path
    ).astype(np.float32)

    print(f" Embedding shape: {embeddings.shape}")

    # Create FAISS index using Inner Product.
    # Since embeddings were L2-normalised, Inner Product = cosine similarity.
    print("\n Building FAISS index...")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    # Add all existing embeddings to FAISS.
    index.add(embeddings)

    # Create the output directory.
    os.makedirs(
        os.path.dirname(settings.INDEX_PATH),
        exist_ok=True
    )

    # Save the FAISS index.
    faiss.write_index(
        index,
        settings.INDEX_PATH
    )

    print(f" FAISS index saved → {settings.INDEX_PATH}")
    print(f" Number of vectors → {index.ntotal}")

    print("\n FAISS index build complete!")


if __name__ == "__main__":
    main()
    