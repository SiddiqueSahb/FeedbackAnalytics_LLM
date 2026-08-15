"""
Sentiment diagnostics — why the off-the-shelf models lose on this corpus.
=========================================================================
The headline comparison in scripts/evaluate_models.py shows fine-tuned
DistilBERT (0.824 accuracy) and logistic regression (0.803) beating
pre-trained RoBERTa (0.633) and VADER (0.514). Aggregate accuracy hides
the reason: both off-the-shelf models almost never predict the neutral
class (per-class F1 of 0.225 and 0.086 respectively), so they behave like
binary classifiers on a three-class, rating-derived label scheme.

This script tests that explanation with three experiments:

  A. Per-platform breakdown
     RoBERTa here is cardiffnlp/twitter-roberta-base-sentiment-latest,
     trained on tweets. If the problem is domain mismatch, it should do
     noticeably better on the Twitter Airline subset than on Amazon or
     Yelp. This turns a plausible argument into measured evidence.

  B. Binary re-evaluation (neutral removed)
     If the three-class label scheme is what penalises the off-the-shelf
     models, removing neutral should close most of the gap.

  C. VADER threshold sensitivity
     VADER's default neutral band is compound in [-0.05, 0.05], which is
     very narrow. Widening it shows how much of VADER's failure is a
     threshold-mapping artefact rather than the lexicon itself.

All three reuse ONE set of cached predictions, so the models are run over
the sample only once. Re-running the script after that is fast.

Outputs (data/results/sentiment_diagnostics/):
    predictions.csv                 cached per-review predictions (all models)
    per_platform_metrics.csv/.png   experiment A
    binary_comparison.csv/.png      experiment B
    vader_threshold_sweep.csv/.png  experiment C
    diagnostics_summary.csv         one-row roll-up

Usage:
    python evaluate/evaluate_sentiment_diagnostics.py
    python evaluate/evaluate_sentiment_diagnostics.py --refresh   # re-run inference
"""

from __future__ import annotations

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path

# Set before torch is imported (transformers pulls it in).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from config import settings

# ── Settings ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
# Reviews sampled per (platform, sentiment) cell — 3 platforms x 3 classes.
# 200 gives 1,800 reviews total, enough for stable per-platform F1 while
# keeping two transformer models on CPU to a sensible runtime.
SAMPLE_PER_CELL = 200
MIN_TEXT_LENGTH = 6

PLATFORMS = ["amazon", "yelp", "twitter_airline"]
CLASSES = ["negative", "neutral", "positive"]

OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "sentiment_diagnostics"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

# Colour-blind-friendly palette (Okabe & Ito, 2008)
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentiment_diagnostics")


def pretty(label: str) -> str:
    """'twitter_airline' -> 'Twitter Airline'."""
    return str(label).replace("_", " ").title()


def load_sample() -> pd.DataFrame:
    """
    Rebuild the same held-out test split used by scripts/evaluate_models.py
    (20%, stratified, random_state=42), then take a balanced subsample with
    an equal number of reviews per platform-and-class cell.

    Balancing per cell matters for experiment A: Twitter Airline is only
    about 2% of the corpus, so a plain random sample would leave too few
    tweets to say anything reliable about per-platform performance.
    """
    data_path = PROJECT_ROOT / settings.DATA_PATH
    logger.info("Loading %s", data_path)
    df = pd.read_parquet(data_path)

    # Same filtering and split as the main evaluation script, so this
    # sample really is drawn from the same held-out data.
    df = df.dropna(subset=["cleaned_text", "sentiment_label"])
    df = df[df["cleaned_text"].str.len() > MIN_TEXT_LENGTH]
    df = df[df["sentiment_label"].isin(CLASSES)]
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["sentiment_label"]
    )
    logger.info("Held-out test set: %d reviews", len(test_df))

    cells = []
    for platform in PLATFORMS:
        for cls in CLASSES:
            cell = test_df[
                (test_df["platform"] == platform) & (test_df["sentiment_label"] == cls)
            ]
            if cell.empty:
                logger.warning("No %s / %s reviews in the test split", platform, cls)
                continue
            take = min(len(cell), SAMPLE_PER_CELL)
            cells.append(cell.sample(n=take, random_state=RANDOM_SEED))

    sample = pd.concat(cells).reset_index(drop=True)
    logger.info("Diagnostic sample: %d reviews", len(sample))
    logger.info("Per platform: %s", sample["platform"].value_counts().to_dict())
    return sample[["review_id", "platform", "sentiment_label", "cleaned_text"]]


def load_models() -> dict[str, object]:
    """
    Instantiate the five classifiers. Any model that cannot be loaded is
    skipped with a warning instead of stopping the run.

    FineTunedSentiment silently falls back to RoBERTa when the fine-tuned
    weights are missing, which would quietly duplicate the RoBERTa column
    and invalidate the comparison — so its _loaded flag is checked here.
    """
    from nlp.sentiment import VaderSentiment, RobertaSentiment, FineTunedSentiment
    from nlp.classical_models import NaiveBayesSentiment, LogisticRegressionSentiment

    models: dict[str, object] = {}

    for name, factory in [
        ("VADER", VaderSentiment),
        ("Naive Bayes", NaiveBayesSentiment),
        ("Logistic Regression", LogisticRegressionSentiment),
        ("RoBERTa (pre-trained)", RobertaSentiment),
        ("DistilBERT (fine-tuned)", FineTunedSentiment),
    ]:
        try:
            model = factory()
        except Exception as exc:
            logger.warning("Could not load %s: %s", name, exc)
            continue

        # Classical models expose _loaded; so does FineTunedSentiment.
        if getattr(model, "_loaded", True) is False:
            logger.warning("%s is not available (untrained or weights missing) — skipped", name)
            continue

        models[name] = model
        logger.info("Loaded %s", name)

    return models


def run_predictions(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Predict every sampled review with every available model, once.

    Alongside the predicted label this stores each model's per-class scores
    (needed for the forced-binary comparison in experiment B) and VADER's
    signed compound score (needed for the threshold sweep in experiment C).
    VaderSentiment.predict() reports abs(compound), which loses the sign,
    so the compound value is taken from the analyser directly.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    models = load_models()
    if not models:
        raise RuntimeError("No sentiment models could be loaded — nothing to evaluate.")

    vader_analyzer = SentimentIntensityAnalyzer()

    rows = []
    total = len(sample)
    for i, row in enumerate(sample.itertuples(index=False), start=1):
        text = str(row.cleaned_text)
        record = {
            "review_id": row.review_id,
            "platform": row.platform,
            "true_label": row.sentiment_label,
            "vader_compound": vader_analyzer.polarity_scores(text)["compound"],
        }

        for name, model in models.items():
            try:
                result = model.predict(text)
                record[f"pred::{name}"] = result.get("label", "error")
                # Keep the raw class scores so experiment B can force a
                # binary decision without re-running the models.
                record[f"scores::{name}"] = json.dumps(result.get("scores", {}))
            except Exception as exc:
                logger.debug("%s failed on %s: %s", name, row.review_id, exc)
                record[f"pred::{name}"] = "error"
                record[f"scores::{name}"] = "{}"

        rows.append(record)
        if i % 200 == 0 or i == total:
            logger.info("Predicted %d / %d reviews", i, total)

    predictions = pd.DataFrame(rows)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Cached predictions -> %s", PREDICTIONS_PATH)
    return predictions


def model_names(predictions: pd.DataFrame) -> list[str]:
    """Model names recovered from the 'pred::<name>' columns."""
    return [c.split("::", 1)[1] for c in predictions.columns if c.startswith("pred::")]


# ── Experiment A: per-platform breakdown ────────────────────────────────────
def experiment_a_per_platform(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Macro F1 and accuracy for each model on each platform separately.

    The point of interest is RoBERTa: it was pre-trained on Twitter data,
    so if domain mismatch explains its weak overall score, its Twitter
    Airline figures should be clearly better than its Amazon figures.
    """
    rows = []
    for name in model_names(predictions):
        for platform in PLATFORMS:
            subset = predictions[predictions["platform"] == platform]
            subset = subset[subset[f"pred::{name}"] != "error"]
            if subset.empty:
                continue
            y_true = subset["true_label"]
            y_pred = subset[f"pred::{name}"]
            rows.append({
                "Model": name,
                "Platform": pretty(platform),
                "Reviews": len(subset),
                "Accuracy": round(accuracy_score(y_true, y_pred), 4),
                "Macro F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
                "Neutral F1": round(
                    f1_score(y_true, y_pred, labels=["neutral"], average="macro", zero_division=0), 4
                ),
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "per_platform_metrics.csv", index=False)

    # Grouped bar chart: one group per model, one bar per platform.
    pivot = table.pivot(index="Model", columns="Platform", values="Macro F1")
    ax = pivot.plot(kind="bar", figsize=(11, 6), color=OKABE_ITO[: len(pivot.columns)])
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("")
    ax.set_title("Sentiment Performance by Platform")
    ax.set_ylim(0, 1)
    ax.legend(title="Platform")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_platform_metrics.png", dpi=300)
    plt.close()

    logger.info("Experiment A done")
    return table


# ── Experiment B: binary re-evaluation ──────────────────────────────────────
def _forced_binary(scores_json: str, fallback_label: str) -> str:
    """
    Decide between negative and positive only, using the model's own class
    scores and ignoring its neutral score. Falls back to the original label
    when scores are unavailable (in which case a neutral prediction simply
    counts as wrong, since the true label here is never neutral).
    """
    try:
        scores = json.loads(scores_json) or {}
    except Exception:
        scores = {}

    neg, pos = scores.get("negative"), scores.get("positive")
    if neg is None or pos is None:
        return fallback_label
    return "negative" if neg >= pos else "positive"


def experiment_b_binary(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Re-score every model with the neutral class removed.

    Two variants are reported, because there is no single neutral-free
    reading that is fair to all five models:
      - Strict: true-neutral reviews dropped; a neutral prediction counts
        as an error (it cannot be correct).
      - Forced binary: the same rows, but each model must choose between
        negative and positive using its own class scores.
    The strict figure penalises models that use the neutral class well;
    the forced-binary figure is the one that isolates the effect of the
    label scheme itself.
    """
    non_neutral = predictions[predictions["true_label"] != "neutral"]

    rows = []
    for name in model_names(predictions):
        subset = non_neutral[non_neutral[f"pred::{name}"] != "error"]
        if subset.empty:
            continue
        y_true = subset["true_label"]

        strict_f1 = f1_score(y_true, subset[f"pred::{name}"], average="macro", zero_division=0)

        forced = [
            _forced_binary(s, lbl)
            for s, lbl in zip(subset[f"scores::{name}"], subset[f"pred::{name}"])
        ]
        forced_f1 = f1_score(y_true, forced, average="macro", zero_division=0)

        # Three-class macro F1 on the same sample, for a like-for-like gap.
        full = predictions[predictions[f"pred::{name}"] != "error"]
        three_class_f1 = f1_score(
            full["true_label"], full[f"pred::{name}"], average="macro", zero_division=0
        )

        rows.append({
            "Model": name,
            "Three-class Macro F1": round(three_class_f1, 4),
            "Binary Macro F1 (strict)": round(strict_f1, 4),
            "Binary Macro F1 (forced)": round(forced_f1, 4),
            "Gain from dropping neutral": round(forced_f1 - three_class_f1, 4),
        })

    table = pd.DataFrame(rows).sort_values("Gain from dropping neutral", ascending=False)
    table.to_csv(OUTPUT_DIR / "binary_comparison.csv", index=False)

    plot_data = table.set_index("Model")[["Three-class Macro F1", "Binary Macro F1 (forced)"]]
    ax = plot_data.plot(kind="bar", figsize=(11, 6), color=[OKABE_ITO[0], OKABE_ITO[2]])
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("")
    ax.set_title("Effect of Removing the Neutral Class")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "binary_comparison.png", dpi=300)
    plt.close()

    logger.info("Experiment B done")
    return table


# ── Experiment C: VADER threshold sweep ─────────────────────────────────────
def experiment_c_vader_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Re-label VADER's cached compound scores using progressively wider
    neutral bands, from the default +/-0.05 up to +/-0.60.

    If macro F1 improves substantially as the band widens, VADER's poor
    neutral performance is largely a consequence of an arbitrary decision
    threshold, not of the lexicon failing to represent neutral text.
    """
    bands = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    y_true = predictions["true_label"]

    rows = []
    for band in bands:
        relabelled = np.where(
            predictions["vader_compound"] >= band, "positive",
            np.where(predictions["vader_compound"] <= -band, "negative", "neutral"),
        )
        rows.append({
            "Neutral band": band,
            "Accuracy": round(accuracy_score(y_true, relabelled), 4),
            "Macro F1": round(f1_score(y_true, relabelled, average="macro", zero_division=0), 4),
            "Neutral F1": round(
                f1_score(y_true, relabelled, labels=["neutral"], average="macro", zero_division=0), 4
            ),
            "Predicted neutral (%)": round(float((relabelled == "neutral").mean()) * 100, 1),
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "vader_threshold_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(table["Neutral band"], table["Macro F1"], marker="o", color=OKABE_ITO[0], label="Macro F1")
    ax.plot(table["Neutral band"], table["Neutral F1"], marker="s", color=OKABE_ITO[1], label="Neutral F1")
    ax.plot(table["Neutral band"], table["Accuracy"], marker="^", color=OKABE_ITO[2], label="Accuracy")
    ax.axvline(0.05, linestyle="--", color="grey", linewidth=1)
    ax.annotate("VADER default", xy=(0.05, 0.05), xytext=(0.08, 0.05), fontsize=9, color="grey")
    ax.set_xlabel("Neutral band half-width (|compound| below this = neutral)")
    ax.set_ylabel("Score")
    ax.set_title("VADER Sensitivity to the Neutral Decision Threshold")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vader_threshold_sweep.png", dpi=300)
    plt.close(fig)

    logger.info("Experiment C done")
    return table


def write_summary(
    per_platform: pd.DataFrame, binary: pd.DataFrame, vader: pd.DataFrame
) -> None:
    """Pull the few numbers worth quoting directly in the results chapter."""
    summary: dict[str, object] = {}

    # The domain-mismatch check: RoBERTa on Twitter vs on Amazon.
    rob = per_platform[per_platform["Model"].str.startswith("RoBERTa")]
    for platform in ["Twitter Airline", "Amazon", "Yelp"]:
        cell = rob[rob["Platform"] == platform]
        if not cell.empty:
            summary[f"RoBERTa Macro F1 ({platform})"] = float(cell.iloc[0]["Macro F1"])

    if not binary.empty:
        top = binary.iloc[0]
        summary["Largest gain from dropping neutral (model)"] = top["Model"]
        summary["Largest gain from dropping neutral (value)"] = float(top["Gain from dropping neutral"])

    if not vader.empty:
        best = vader.loc[vader["Macro F1"].idxmax()]
        summary["VADER default Macro F1"] = float(vader.iloc[0]["Macro F1"])
        summary["VADER best Macro F1"] = float(best["Macro F1"])
        summary["VADER best neutral band"] = float(best["Neutral band"])

    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "diagnostics_summary.csv", index=False)
    logger.info("Summary written")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment diagnostics for RQ1.")
    parser.add_argument(
        "--refresh", action="store_true",
        help="re-run model inference even if cached predictions exist",
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if PREDICTIONS_PATH.exists() and not args.refresh:
            logger.info("Reusing cached predictions (%s). Pass --refresh to redo inference.",
                        PREDICTIONS_PATH.name)
            predictions = pd.read_csv(PREDICTIONS_PATH)
        else:
            predictions = run_predictions(load_sample())

        found = model_names(predictions)
        logger.info("Models in predictions: %s", ", ".join(found))

        per_platform = experiment_a_per_platform(predictions)
        binary = experiment_b_binary(predictions)
        vader = experiment_c_vader_thresholds(predictions)
        write_summary(per_platform, binary, vader)

        logger.info("All experiments complete. Results in %s", OUTPUT_DIR)

    except FileNotFoundError as exc:
        logger.error("Required file not found: %s", exc)
    except Exception as exc:
        logger.error("Diagnostics failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
