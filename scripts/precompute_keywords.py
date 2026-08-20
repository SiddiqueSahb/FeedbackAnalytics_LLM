"""
Precompute the dashboard's keyword frequencies.

Why this exists
---------------
`get_top_keywords()` in backend/services/analytics_service.py ran spaCy over
every review in the corpus on the first request: 642,692 rows averaging ~41
words is roughly 26 million words through the tagger and dependency parser
(the parser is what `.noun_chunks` needs, and it's the expensive half). On a
single CPU core that's tens of minutes, so the dashboard's `/api/analytics/
keywords` call never returned inside any sane HTTP timeout and the page
appeared to hang.

Nothing about that work needs to happen per request — the corpus is static.
This script does it once, offline, over the *full* corpus, and writes the
counts to JSON. The endpoint then reads the file and answers instantly, with
exact corpus counts rather than an approximation.

Run it after any change to data/processed/reviews_unified.parquet:

    python scripts/precompute_keywords.py

Expect it to take a while — it is doing the full 26M words. Progress is
printed per bucket. If the output file is absent the endpoint falls back to
a clearly-labelled sample, so the dashboard still works before you run this.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import spacy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.analytics_service import (  # noqa: E402
    KEYWORD_FILTER_VERSION,
    normalise_phrase,
)
from config import settings  # noqa: E402

# Keep more than the dashboard asks for (it requests 30, the API allows up to
# 100) so the same file serves every reasonable `n` without recomputing.
TOP_N = 200

# One bucket per value the dashboard's sentiment filter can take, plus the
# unfiltered case.
BUCKETS = [None, "positive", "negative", "neutral"]

OUTPUT_DIR = os.path.join("data", "results", "keywords")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "top_keywords.json")

BATCH_SIZE = 200


def count_phrases(nlp, texts: list[str]) -> Counter:
    """
    Noun-chunk frequencies over a list of texts.

    Phrase filtering comes from normalise_phrase() in analytics_service, not
    from a copy here. That is deliberate: the API falls back to computing
    these live from a sample when this file is missing, and if the two used
    different rules the dashboard would report different numbers for the same
    corpus depending on which path served the request.
    """

    counter: Counter = Counter()
    started = time.time()

    for i, doc in enumerate(nlp.pipe(texts, batch_size=BATCH_SIZE), start=1):

        counter.update(
            phrase
            for phrase in (normalise_phrase(chunk) for chunk in doc.noun_chunks)
            if phrase is not None
        )

        if i % 25_000 == 0:
            rate = i / max(time.time() - started, 1e-9)
            remaining = (len(texts) - i) / max(rate, 1e-9)
            print(
                f"    {i:,}/{len(texts):,} reviews "
                f"({rate:,.0f}/s, ~{remaining / 60:.1f} min left)",
                flush=True,
            )

    return counter


def main() -> int:

    if not os.path.exists(settings.DATA_PATH):
        print(f"Dataset not found: {settings.DATA_PATH}", file=sys.stderr)
        return 1

    print(f"Reading {settings.DATA_PATH} ...", flush=True)
    df = pd.read_parquet(
        settings.DATA_PATH,
        columns=["cleaned_text", "sentiment_label"],
    )
    print(f"  {len(df):,} rows", flush=True)

    # Same pipeline configuration the service uses, so the counts this script
    # produces are the ones the service would have produced itself.
    print("Loading spaCy (en_core_web_sm, ner + lemmatizer disabled) ...", flush=True)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    results: dict[str, list[dict]] = {}
    counts: dict[str, int] = {}
    overall_start = time.time()

    for bucket in BUCKETS:

        key = bucket or "all"
        subset = df if bucket is None else df[df["sentiment_label"] == bucket]
        texts = subset["cleaned_text"].dropna().tolist()
        counts[key] = len(texts)

        print(f"\n[{key}] {len(texts):,} reviews", flush=True)

        if not texts:
            results[key] = []
            continue

        counter = count_phrases(nlp, texts)

        results[key] = [
            {"word": word, "count": count}
            for word, count in counter.most_common(TOP_N)
        ]

        print(f"  {len(counter):,} distinct phrases, keeping top {TOP_N}", flush=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": settings.DATA_PATH,
        "basis": "corpus",
        "spacy_model": "en_core_web_sm",
        # Recorded so the API can refuse a file built by an older filter
        # rather than silently serving counts the current rules wouldn't
        # produce. See KEYWORD_FILTER_VERSION in analytics_service.py.
        "filter_version": KEYWORD_FILTER_VERSION,
        "top_n": TOP_N,
        "review_counts": counts,
        "keywords": results,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    elapsed = (time.time() - overall_start) / 60
    print(f"\nWrote {OUTPUT_FILE} in {elapsed:.1f} min", flush=True)
    print("The dashboard will now serve exact corpus counts instantly.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
