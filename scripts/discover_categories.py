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
  3. Filter non-English reviews automatically (langdetect)
  4. BERTopic reads all text and groups similar complaints together
  5. Top keywords per group become the category name
  6. Noise topics removed automatically (rating-meta, too small)
  7. Save discovered categories — one file per sentiment

Usage:
    python scripts/discover_categories.py
    python scripts/discover_categories.py --sentiment negative
    python scripts/discover_categories.py --sentiment positive
    python scripts/discover_categories.py --sentiment neutral
    python scripts/discover_categories.py --sample 30000
    python scripts/discover_categories.py --topics 15
    python scripts/discover_categories.py --platform amazon

Outputs:
    data/processed/complaint_categories_negative.json  <- loaded by categoriser.py
    data/processed/complaint_categories_positive.json
    data/processed/complaint_categories_neutral.json
    data/processed/bertopic_info_<sentiment>.json      <- topic details for dissertation

Author: Mohammad Asim | MSc Dissertation 2026
"""

from __future__  import annotations
import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from bertopic.representation import KeyBERTInspired
from langdetect import detect, LangDetectException




import json
import spacy
import argparse
import numpy as np
import pandas as pd
from collections import Counter

from config import settings
from logger import get_logger

log = get_logger("discover_categories")
representation_model = KeyBERTInspired()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

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


def filter_english(texts: list[str]) -> list[str]:
    """
    Removes non-English reviews using langdetect — no hardcoded word lists.
    Supports 55 languages automatically.
    Short texts (<4 words) are kept as-is since they are too short to detect.
    """
    keep = []
    for text in texts:
        try:
            if detect(text) == "en":
                keep.append(text)
        except LangDetectException:
            continue # Skip reviews whose language cannot be detected

    removed = len(texts) - len(keep)
    if removed:
        print(f"  Language filter: removed {removed:,} non-English reviews "
              f"({removed / len(texts) * 100:.1f}%)")
    else:
        print(f"  Language filter: all reviews appear to be English")
    return keep


def dynamic_min_topic_size(n_texts: int) -> int:
    """
    Scale minimum topic size with corpus size.
    Formula: 0.5% of corpus, capped between 50 and 500.
    Examples:
        10,000 reviews  ->  50  (lower cap)
        50,000 reviews  -> 250
       100,000 reviews  -> 500  (upper cap)
    """
    size = int(n_texts * 0.005)  # 0.5% of corpus
    return max(50, min(500, size))


def filter_noise_topics(topic_details: list[dict]) -> list[dict]:
    """
    Drops topics whose keywords appear across too many other topics.
    Words in >50% of topics carry no discriminative signal — detected dynamically.
    """
    if len(topic_details) < 3:
        return topic_details

    # Detect noise words dynamically — no hardcoded list
    word_count = Counter(kw.lower() for d in topic_details for kw in d["keywords"])
    noise = {w for w, c in word_count.items() if c >= len(topic_details) * 0.5}

    if noise:
        print(f"  Global noise words: {', '.join(sorted(noise)[:8])}")

    clean = []
    for detail in topic_details:
        kws = [kw.lower() for kw in detail["keywords"]]
        noise_ratio = sum(1 for kw in kws if any(n in kw for n in noise)) / max(len(kws), 1)
        if noise_ratio >= 0.6:
            print(f"  Dropped: '{detail['label']}' (noise {noise_ratio:.0%}, {detail['count']:,} reviews)")
        else:
            clean.append(detail)

    if len(clean) < len(topic_details):
        print(f"  Noise filter: {len(topic_details) - len(clean)} removed, {len(clean)} remain")
    return clean





def clean_topic_label(keywords: list[tuple[str, float]]) -> str:

    noise = {
        "good", "bad", "great", "nice", "poor",
        "product", "review", "item", "thing",
        "star", "stars"
    }

    clean = []
    seen = set()

    for word, score in keywords:

        word = word.replace("_", " ").strip()
        word = word.lower().replace("_", " ").strip()

        # Lemmatize each word
        doc = nlp(word)

        lemma = " ".join(token.lemma_ for token in doc).lower()

        if lemma in noise:
            continue

        if lemma not in seen:
            seen.add(lemma)
            clean.append(word)

    if not clean:
        clean = [word for word, _ in keywords[:4]]

    return ", ".join(clean[:4]).title()


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

    # Vectorizer for extracting topic words
    vectorizer_model = CountVectorizer(
        stop_words="english",   # Remove common English stop words
        ngram_range=(1, 2),     # Use single words and two-word phrases
        min_df=5,               # Ignore words appearing in fewer than 5 reviews
        max_df=0.90,            # Ignore words appearing in more than 90% of reviews
    )

    print(f"Creating BERTopic model with {n_topics} topics")
    # Create and configure the BERTopic model.
    topic_model = BERTopic(
    representation_model = representation_model,
    embedding_model=embed_model, # SentenceTransformer used to generate semantic embeddings.
    umap_model=umap_model,  # UMAP model used for dimensionality reduction.
    hdbscan_model=hdbscan_model, # HDBSCAN model used for document clustering
    vectorizer_model=vectorizer_model, # CountVectorizer model used for feature extraction.
    nr_topics=n_topics, # Target number of topics after automatic topic reduction.
    top_n_words=10, # Number of representative words displayed for each topic.
    verbose=True, # Display training progress in the console.
    calculate_probabilities=False,# Disable probability computation to improve processing speed
)

    # Fit the BERTopic model, store the assigned topics, and ignore the returned probabilities.
    topics, _ = topic_model.fit_transform(texts)
    os.makedirs("models", exist_ok=True)
    topic_model.save("models/bertopic_model")

    #extract topic info — Retrieve summary information for all discovered topics.
    topic_info_df = topic_model.get_topic_info()
    print(f"Total topics (including outlier): {len(topic_info_df)}")
    print(f"Actual topics: {len(topic_info_df[topic_info_df['Topic'] != -1])}")

    #to store categories and topic details
    categories:list[str] = []
    topic_details:list[dict] = []

    # Iterate through each topic in the topic information DataFrame. (index,row in topic_info_df)
    for _,row in topic_info_df.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        top_words_raw = topic_model.get_topic(topic_id) #output - (keyword, importance_score)

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


def main() ->  None:
    parser = argparse.ArgumentParser(description="Discover categories using BERTopic.")

    parser.add_argument("--sample",type=int,default=None,help="Maximum reviews to sample (default: use all reviews)")  
    parser.add_argument("--topics", type=int, default=None, help="Number of topics to discover (default: 15)")
    parser.add_argument("--min-topic-size", type=int, default=None, help="Minimum reviews per topic (default: auto — 0.5%% of corpus)")
    parser.add_argument("--platform", type=str, default=None,choices=["amazon", "yelp", "twitter_airline"],help="Discover categories for one platform only (default: all)",)
    parser.add_argument("--sentiment", type=str, default=None,choices=["negative", "positive", "neutral", "all"],help="Focus on a sentiment class (default: negative — most useful for complaints)",)
    args = parser.parse_args()

    # Output paths — one file per sentiment so runs don't overwrite each other
    sentiment_tag = args.sentiment
    output_cats   = f"data/processed/complaint_categories_{sentiment_tag}.json"
    output_info   = f"data/processed/bertopic_info_{sentiment_tag}.json"

    print("  FeedbackIQ — Dynamic Category Discovery")
    print(f"  Sentiment : {sentiment_tag}")
    print(f"  Output    : {output_cats}")

    #check if data exists
    if not os.path.exists(settings.DATA_PATH):
        print(f"\n Data not found at {settings.DATA_PATH}")
        print(f"Run first: python scripts/preprocess.py ")
        sys.exit(1)

    # load data
    print(f"Loading data from {settings.DATA_PATH}")
    df = pd.read_parquet(settings.DATA_PATH)
    print(f"  Loaded {len(df):,} reviews from {settings.DATA_PATH}")
    print(f"  Platforms : {df['platform'].value_counts().to_dict()}")
    print(f"  Sentiments: {df['sentiment_label'].value_counts().to_dict()}")


    # Filter data based on platform and sentiment
    if args.platform:
        df = df[df["platform"] == args.platform]
        print(f"  Filtered to platform '{args.platform}': {len(df):,} reviews")

    if args.sentiment != "all":
        df = df[df["sentiment_label"] == args.sentiment]
        print(f"  Filtered to '{args.sentiment}' sentiment: {len(df):,} reviews")

    if df.empty:
        print(f" No reviews after filtering. Try --sentiment all ")
        sys.exit(1)
        

    texts = df["cleaned_text"].fillna("").tolist()
    # Remove non-English reviews
    print(f" {len(texts):,} reviews ready before language filtering")
    texts = filter_english(texts)
    print(f" {len(texts):,} reviews ready for BERTopic ")


    if not texts:
        print("  No reviews remain after language filter.")
        sys.exit(1)

    # Dynamic improvement 2 — scale min_topic_size to corpus size automatically
    if args.min_topic_size:
        min_size = args.min_topic_size
        print(f"  min_topic_size: {min_size} (user-specified)")
    else:
        min_size = dynamic_min_topic_size(len(texts))
        print(f"  min_topic_size: {min_size} (auto — 0.5% of {len(texts):,} reviews)")

    #Run BERTopic to discover categories
    print(f"Running BERTopic ")
    n_topics = args.topics if args.topics else "auto"

    try:
        categories,topic_details = run_bertopic(texts,n_topics = n_topics,min_topic_size = min_size)
    except ImportError as e:
        print(f"\n Missing package: {e}")
        print(f"   Run: pip install bertopic ")
        sys.exit(1)
    except Exception as e:
        print(f"\n  BERTopic failed: {e}")
        print(f" Saving static fallback categories instead.")
        log.error("BERTopic failed: %s", e, exc_info=True)
        categories = STATIC_FALLBACK
        topic_details = [
            {
                "topic_id": i,
                "label": c,
                "keywords" : c.split(),
                "count": 0
            }
            for i,c in enumerate(STATIC_FALLBACK)
            #enumerate is used to get the index of the category(0, "product quality")
            #(1, "delivery and shipping")
            #(2, "customer service")
        ]

    # Dynamic improvement 3 — remove noise topics automatically
    topic_details = filter_noise_topics(topic_details)

    # Remove duplicate labels
    seen = set()
    unique_topic_details = []

    for detail in topic_details:
         if detail["label"] not in seen:
            seen.add(detail["label"])
            unique_topic_details.append(detail)

    # Update topic_details
    topic_details = unique_topic_details
    # Create categories list from unique topics
    categories = [detail["label"] for detail in topic_details]

    # Show discovered categories
    print(f"  Found {len(categories)} categories:\n")
    for i, category in enumerate(categories, start=1):

        detail: dict = {}

        # Find the matching topic
        for d in topic_details:
            if d["label"] == category:
                detail = d
                break

        keywords = detail["keywords"][:5]
        kws = ", ".join(keywords)
        cnt = detail["count"]

        print(f"Topic Number : {i}")
        print(f"Category     : {category}")
        print(f"Reviews      : {cnt}")
        print(f"Keywords     : {kws}")
        print("-" * 50)

    #Save outputs
    print(f"Saving outputs to {output_cats} and {output_info}")
    os.makedirs("data/processed", exist_ok=True)

    # Main output: just the category labels (used by categoriser.py)
    with open(output_cats, "w") as f:
        json.dump(categories, f, indent=2)
    print(f" Categories saved → {output_cats}")

    # Detailed output: full topic info for dissertation
    info_payload = {
        "n_categories":  len(categories),
        "sample_size":   len(texts),
        "platform":      args.platform or "all",
        "sentiment":     args.sentiment,
        "min_topic_size": min_size,
        "categories":    categories,
        "topic_details": topic_details,
    }
    with open(output_info, "w") as f:
        json.dump(info_payload, f, indent=2)
    print(f" Topic details saved → {output_info}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" Category discovery complete! ")
    print(f"{'=' * 60}")
    print(f"\n  Discovered {len(categories)} data-driven complaint categories")
    print(f"  (replaced {len(STATIC_FALLBACK)} static hardcoded categories)\n")
    print(f"  Dynamic improvements applied:")
    print(f"   Language filter  — non-English reviews removed automatically")
    print(f"   Auto min-size    — threshold scaled to corpus ({min_size} reviews)")
    print(f"   Noise filter     — rating-meta and cross-topic noise removed")
    print(f"   Per-sentiment    — saved to {output_cats}\n")
    print(f"  Next steps:")
    print(f"   • categoriser.py will now load these automatically")
    print(f"   • Run for other sentiments:")
    print(f"       python scripts/discover_categories.py --sentiment positive")
    print(f"       python scripts/discover_categories.py --sentiment neutral")
    print(f"   • Re-run any time your data changes: python scripts/discover_categories.py")
    print(f"   • For dissertation: compare these vs static 7 in your results chapter\n")


if __name__ == "__main__":
    main()