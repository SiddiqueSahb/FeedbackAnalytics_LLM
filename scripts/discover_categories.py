"""
Dynamic Complaint Category Discovery using BERTopic.
=====================================================
Reads your real review data, groups similar complaints together
automatically, and saves the discovered categories to a JSON file.

This replaces the hardcoded 7 static categories with categories that
come directly from YOUR data — making the dissertation methodology
data-driven rather than researcher-assumed.

How it works (simple version):
  1. Load reviews from reviews_unified.parquet
  2. Sample up to --sample reviews (default 50,000) for speed
  3. BERTopic reads all text and groups similar complaints together
  4. Top keywords per group become the category name
  5. Save discovered categories to data/processed/complaint_categories.json
  6. categoriser.py loads from that file automatically from now on

Usage:
    python scripts/discover_categories.py
    python scripts/discover_categories.py --sample 30000
    python scripts/discover_categories.py --topics 15
    python scripts/discover_categories.py --platform amazon

Outputs:
    data/processed/complaint_categories.json   ← loaded by categoriser.py
    data/processed/bertopic_info.json          ← topic details for dissertation

"""

from __future__  import annotations
import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import pandas as pd

from config import settings
from logger import get_logger

OUTPUT_CATEGORIES = "data/processed/complaint_categories.json"
OUTPUT_TOPIC_INFO = "data/processed/bertopic_info.json"

#Fallback static categories (used if BERTopic fails)
STATIC_FALLBACK: list[str] = [
    "product quality",
    "delivery and shipping",
    "customer service",
    "pricing and value",
    "user experience",
    "technical issues",
    "refund and returns",
]

def clean_topic_label(keywords: list[tuple[str,float]]) -> str:
    # Remove very generic words that add no meaning
    noise = {
        "good", "bad", "great", "nice", "poor", "just", "really",
        "very", "much", "like", "love", "hate", "also", "even",
        "got", "get", "one", "use", "used", "item", "thing",
        "product", "review", "star", "stars", "would", "could",
    }
    clean = [word.replace("_", " ").strip() for word, score in keywords if word.lower() not in noise and len(word) > 2]
    if not clean:
        clean = [word for word, _ in keywords[:4]]
    return ",".join(clean[:4]).title()


def run_bertopic(texts:list[str],n_topics:int|str = "auto",min_topic_size:int = 30):
    """
    Run BERTopic on the provided texts.
    Returns:
        categories  — list of discovered category label strings
        topic_info  — list of dicts with topic details 
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    print(f"  Loading embedding model: {settings.EMBEDDING_MODEL}")
    #converting text to numbers
    embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    #UMAP: reduce 384-dim embeddings to 5-dim for clustering 
    umap_model = UMAP(n_neighbors=15,n_components=5,min_dist=0.0,metric="cosine",random_state=42,low_memory=True,)

    #HDBSCAN: cluster the reduced embeddings 
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size,min_samples=10,metric="euclidean",cluster_selection_method="eom",prediction_data=True,)

    # Create and configure the BERTopic model.
    topic_model = BERTopic(
    embedding_model=embed_model, # SentenceTransformer used to generate semantic embeddings.
    umap_model=umap_model,  # UMAP model used for dimensionality reduction.
    hdbscan_model=hdbscan_model, # HDBSCAN model used for document clustering
    nr_topics=n_topics, # Target number of topics after automatic topic reduction.
    top_n_words=10, # Number of representative words displayed for each topic.
    verbose=True, # Display training progress in the console.
    calculate_probabilities=False,# Disable probability computation to improve processing speed    
)

    # Fit the BERTopic model, store the assigned topics, and ignore the returned probabilities.
    topics, _ = topic_model.fit_transform.fit_transform(texts)

    #extract topic info -  Retrieve summary information for all discovered topics.
    topic_info_df = topic_model.get_topic_info()

    #to store categoriesand topic details
    categories:list[str] = []
    topic_details:list[dict] = []

    # Iterate through each topic in the topic information DataFrame. (index,row in topic_info_df)
    for _,row in topic_info_df.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        
        top_words_raw = topic_model.get_topic(topic_id) #output - (keyword - topic, importance_score)
        if not top_words_raw:
            continue

        # top_words_raw = [('battery', 0.09), ('charge', 0.08), ...]
        label = clean_topic_label(top_words_raw) #sending keywords to clean_topic_label function
        keywords = [word for word, _ in top_words_raw[:10]]
        count    = int(row.get("Count", 0))

        categories.append(label)
        topic_details.append({
            "topic_id":  topic_id,
            "label":     label,
            "keywords":  keywords,
            "count":     count,
        })
    return categories, topic_details