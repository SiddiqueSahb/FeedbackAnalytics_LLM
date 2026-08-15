"""
Analytics service for the FeedbackIQ dashboard.
"""

from __future__ import annotations

import os
import sys
import threading
from collections import Counter
from functools import lru_cache

import pandas as pd
import spacy

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

from config import settings
from logger import get_logger

log = get_logger("service.analytics")

_nlp_lock = threading.Lock()
_df_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_nlp_uncached():
    """
    Loaded on first use, not at import time. Loading it at module level used
    to run the moment this file was imported — i.e. during app startup,
    before uvicorn had even started accepting connections, delaying
    readiness for every endpoint, not just the ones that need spaCy.

    ner and lemmatizer are disabled: get_top_keywords() only reads
    doc.noun_chunks, which needs the tagger + parser, not those two.
    """
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def _get_nlp():
    with _nlp_lock:
        return _get_nlp_uncached()


@lru_cache(maxsize=1)
def _load_df_uncached() -> pd.DataFrame:
    """Load the processed dataset once."""

    if not os.path.exists(settings.DATA_PATH):
        log.warning("Dataset not found: %s", settings.DATA_PATH)
        return pd.DataFrame()

    log.info("Loading dataset...")
    return pd.read_parquet(settings.DATA_PATH)


def _load_df() -> pd.DataFrame:
    # Lock closes the gap lru_cache leaves open on its own: without it, two
    # concurrent first requests could each start reading the 407MB parquet
    # file at once instead of the second one waiting for the first's result.
    with _df_lock:
        return _load_df_uncached()


def get_summary_stats() -> dict:
    """Return dashboard summary statistics."""

    df = _load_df()

    if df.empty:
        return {}

    return {
        "total_reviews": len(df),
        "platforms": df["platform"].nunique(),
        "avg_rating": round(df["rating"].mean(), 2),
        "positive_pct": round(
            (df["sentiment_label"] == "positive").mean() * 100,
            1,
        ),
        "negative_pct": round(
            (df["sentiment_label"] == "negative").mean() * 100,
            1,
        ),
        "neutral_pct": round(
            (df["sentiment_label"] == "neutral").mean() * 100,
            1,
        ),
    }

def get_sentiment_by_platform() -> list[dict]:
    """Return sentiment counts grouped by platform."""

    df = _load_df()

    if df.empty:
        return []

    result = (
        df.groupby(["platform", "sentiment_label"])
        .size()
        .reset_index(name="count")
        .sort_values("platform")
    )

    return result.to_dict("records")


def get_rating_distribution() -> list[dict]:
    """Return the number of reviews for each rating."""

    df = _load_df()

    if df.empty:
        return []

    ratings = (
        df["rating"]
        .value_counts()
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="count")
    )

    return ratings.to_dict("records")

@lru_cache(maxsize=8)
def get_top_keywords(
    sentiment: str | None = None,
    n: int = 30,
) -> list[dict]:
    """Return the most common business-related phrases."""

    df = _load_df()

    if df.empty:
        return []

    if sentiment:
        df = df[df["sentiment_label"] == sentiment]

    counter = Counter()

    # nlp.pipe() batches internally instead of paying full per-call
    # pipeline overhead for each of up to ~640K rows one at a time —
    # same output, meaningfully faster.
    texts = df["cleaned_text"].dropna().tolist()

    for doc in _get_nlp().pipe(texts, batch_size=200):

        phrases = [
            chunk.text.lower().strip()
            for chunk in doc.noun_chunks
            if len(chunk.text) > 2
        ]

        counter.update(phrases)

    return [
        {
            "word": word,
            "count": count,
        }
        for word, count in counter.most_common(n)
    ]

def get_platform_list() -> list[str]:
    """Return all available review platforms."""

    df = _load_df()

    if df.empty:
        return []

    return sorted(df["platform"].dropna().unique().tolist())

def get_trend_data() -> list[dict]:
    """Return monthly sentiment trends."""

    df = _load_df()

    if df.empty or "date" not in df.columns:
        return []

    try:

        trend_df = df.copy()

        trend_df["date"] = pd.to_datetime(
            trend_df["date"],
            errors="coerce",
        )

        trend_df = trend_df.dropna(subset=["date"])

        trend_df["month"] = (
            trend_df["date"]
            .dt.to_period("M")
            .astype(str)
        )

        trends = (
            trend_df.groupby(
                ["month", "sentiment_label"]
            )
            .size()
            .reset_index(name="count")
            .sort_values("month")
        )

        return trends.to_dict("records")

    except Exception:
        log.exception("Unable to generate trend data.")
        return []