"""
Analytics service for the FeedbackIQ dashboard.
"""

from __future__ import annotations

import json
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

# Precomputed keyword counts written by scripts/precompute_keywords.py.
# See get_top_keywords() below for why they're precomputed rather than
# calculated on demand.
KEYWORDS_FILE = os.path.join("data", "results", "keywords", "top_keywords.json")

# Fallback sample size used only when that file is absent. 25,000 reviews
# takes a minute or two rather than the better part of an hour, and
# noun-phrase *rankings* are stable well below this size — but the counts
# are sample counts, so anything returned this way is labelled as such and
# the dashboard says so on the chart.
KEYWORD_SAMPLE_SIZE = 25_000
KEYWORD_SAMPLE_SEED = 42


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

@lru_cache(maxsize=1)
def _load_precomputed_keywords() -> dict | None:
    """
    Read the precomputed keyword counts, or None if they haven't been
    generated yet. Cached because the file doesn't change while the process
    is running; restart the backend after regenerating it.
    """

    if not os.path.exists(KEYWORDS_FILE):
        log.warning(
            "No precomputed keywords at %s — falling back to a %s-review "
            "sample. Run scripts/precompute_keywords.py for exact counts.",
            KEYWORDS_FILE,
            f"{KEYWORD_SAMPLE_SIZE:,}",
        )
        return None

    try:
        with open(KEYWORDS_FILE, encoding="utf-8") as handle:
            payload = json.load(handle)

    except (OSError, json.JSONDecodeError):
        log.exception("Precomputed keyword file is unreadable: %s", KEYWORDS_FILE)
        return None

    if not isinstance(payload.get("keywords"), dict):
        log.error("Precomputed keyword file has no 'keywords' object: %s", KEYWORDS_FILE)
        return None

    log.info(
        "Loaded precomputed keywords generated %s",
        payload.get("generated_at", "at an unknown time"),
    )

    return payload


@lru_cache(maxsize=8)
def get_top_keywords(
    sentiment: str | None = None,
    n: int = 30,
) -> list[dict]:
    """
    Return the most common business-related phrases.

    Reads precomputed counts where they exist, and only falls back to
    computing them live — from a sample — where they don't.

    The reason for that split: doing this properly means running spaCy's
    tagger and dependency parser over every review, because `.noun_chunks`
    needs the parse. At 642,692 reviews averaging ~41 words that's roughly
    26 million words, which takes tens of minutes on CPU. It used to happen
    inside the request, so /api/analytics/keywords could not return within
    any reasonable HTTP timeout and the dashboard hung waiting for it. The
    corpus is static, so there was never a reason to pay that cost per
    request — scripts/precompute_keywords.py pays it once, offline.

    Every returned record carries `basis` ("corpus" or "sample") and
    `n_reviews`, so a caller can tell exact counts from sampled ones instead
    of having to assume. The dashboard surfaces this as a caption.
    """

    precomputed = _load_precomputed_keywords()

    if precomputed is not None:

        key = sentiment or "all"
        entries = precomputed["keywords"].get(key)

        if entries is None:
            log.warning(
                "Precomputed keywords have no '%s' bucket; sampling instead.", key
            )

        else:
            reviewed = precomputed.get("review_counts", {}).get(key, 0)

            return [
                {
                    "word": entry["word"],
                    "count": entry["count"],
                    "basis": "corpus",
                    "n_reviews": reviewed,
                }
                for entry in entries[:n]
            ]

    # ---- fallback: sample, and say so -------------------------------------

    df = _load_df()

    if df.empty:
        return []

    if sentiment:
        df = df[df["sentiment_label"] == sentiment]

    texts = df["cleaned_text"].dropna()

    if texts.empty:
        return []

    sampled = len(texts) > KEYWORD_SAMPLE_SIZE

    if sampled:
        # Fixed seed so repeated calls and restarts agree with each other —
        # a chart whose numbers move on refresh is worse than one that's
        # approximate for a stated reason.
        texts = texts.sample(KEYWORD_SAMPLE_SIZE, random_state=KEYWORD_SAMPLE_SEED)

    log.info(
        "Computing keywords live over %d reviews (sampled=%s).", len(texts), sampled
    )

    counter = Counter()

    # nlp.pipe() batches internally instead of paying full per-call
    # pipeline overhead for each row one at a time — same output,
    # meaningfully faster.
    for doc in _get_nlp().pipe(texts.tolist(), batch_size=200):

        counter.update(
            chunk.text.lower().strip()
            for chunk in doc.noun_chunks
            if len(chunk.text) > 2
        )

    return [
        {
            "word": word,
            "count": count,
            "basis": "sample" if sampled else "corpus",
            "n_reviews": len(texts),
        }
        for word, count in counter.most_common(n)
    ]


def warm_cache() -> None:
    """
    Load the parquet and the precomputed keywords ahead of the first request.

    Called from the app's startup hook in backend/main.py. Without it the
    first dashboard visit pays a 407MB read before anything renders; with it
    that read overlaps with the server coming up. Deliberately swallows
    errors — a failed warm-up should make the first request slow, not stop
    the service from starting.
    """

    try:
        rows = len(_load_df())
        _load_precomputed_keywords()
        log.info("Warm-up complete: %d rows resident.", rows)

    except Exception:
        log.exception("Warm-up failed; data will be loaded on first request.")

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