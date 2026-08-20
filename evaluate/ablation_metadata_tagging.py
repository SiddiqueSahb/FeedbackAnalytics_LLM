"""
Ablation: Metadata Tagging in the RAG Context

Author : Mohammad Asim Siddique

Chapter 3 argues that tagging each retrieved review with its
platform/rating/sentiment metadata before it reaches the generator LLM
(rag/pipeline.py's DOCUMENT_PROMPT) is what makes RAG_PROMPT's grounding
rules ("only cite platforms present in context") enforceable rather than
aspirational. That claim has never been measured -- this script measures it.

Same 5 questions as evaluate_llm_vs_rag.py, run through the RAG pipeline
twice:

  Condition A ("tagged")    -- current default. Each review enters the LLM's
                                context prefixed with
                                [Platform: ... | Rating: ... | Sentiment: ...]
                                exactly as rag/pipeline.py builds it.
  Condition B ("untagged")  -- identical retrieval, but the generator LLM
                                sees only the raw review text, no metadata
                                prefix.

Retrieval is unchanged between conditions (same SIMILARITY_THRESHOLD gate,
same MMR selection) -- only what the generator is shown differs. So any
RAGAS score difference between the two conditions reflects the effect of
tagging on generation, not retrieval quality. Context Precision and Context
Recall are expected to move little or not at all for that reason; the
interesting deltas are Faithfulness and Answer Relevancy.

Toggled by monkeypatching rag.pipeline.DOCUMENT_PROMPT for the duration of
Condition B's generation loop, then restoring the original -- this file
does NOT edit rag/pipeline.py. Python resolves the global name
`DOCUMENT_PROMPT` inside ask() against the module's __dict__ at call time,
so reassigning rag_pipeline_module.DOCUMENT_PROMPT before calling ask()
is sufficient; no import-time binding is broken by doing this.

Same caveats as evaluate_llm_vs_rag.py apply and are not repeated by this
script's design: n=5 questions, and the RAGAS judge is the same underlying
model (Groq llama-3.1-8b-instant) as the generator, object-isolated but not
model-independent.

Usage:  python evaluate/ablation_metadata_tagging.py
"""

from __future__ import annotations

import os
import re
import time
import sys
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from datasets import Dataset

import rag.pipeline as rag_pipeline_module
from rag.pipeline import ask, _get_llm
from config import settings

from langchain_core.prompts import PromptTemplate

# ── ragas / langchain-community compatibility shim ────────────────────────────
# Same fix as evaluate_llm_vs_rag.py -- ragas 0.4.x unconditionally imports
# langchain_community.chat_models.vertexai at module load, which no longer
# exists in this project's installed langchain-community. We never use
# VertexAI, so stub it out before importing ragas.
import types as _types

_vertexai_shim = _types.ModuleType("langchain_community.chat_models.vertexai")


class _StubChatVertexAI:  # pragma: no cover
    pass


_vertexai_shim.ChatVertexAI = _StubChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_shim)

import langchain_community.llms as _langchain_community_llms  # noqa: E402

if not hasattr(_langchain_community_llms, "VertexAI"):
    class _StubVertexAI:  # pragma: no cover
        pass

    _langchain_community_llms.VertexAI = _StubVertexAI

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

from ragas import evaluate  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from ragas.metrics import (  # noqa: E402
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ── Rate-limit retry helper (identical to evaluate_llm_vs_rag.py) ─────────────
# This script makes the same order of magnitude of Groq calls as
# evaluate_llm_vs_rag.py (10 generations + judged metrics), so it needs the
# same backoff behaviour.
MAX_RATE_LIMIT_RETRIES = 6
SLEEP_BETWEEN_QUESTIONS = 8.0


def _extract_retry_after_seconds(exc: Exception, default: float = 8.0) -> float:
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 0.5
        except ValueError:
            pass
    return default


def call_with_retry(fn, *args, max_retries: int = MAX_RATE_LIMIT_RETRIES, **kwargs):
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            is_rate_limit = "rate_limit" in str(exc).lower() or "429" in str(exc)
            if not is_rate_limit or attempt >= max_retries:
                raise
            wait = _extract_retry_after_seconds(exc)
            print(f"    [rate limit] waiting {wait:.1f}s before retry ({attempt + 1}/{max_retries})...")
            time.sleep(wait)


# RAGAS asks the judge to decompose an answer into individual statements and
# return a verdict on each as JSON. That output is far longer than a normal
# chat reply, and if the model stops before closing the JSON, RAGAS raises
# LLMDidNotFinishException and records NaN for that question.
#
# That is not a harmless gap. The means are computed over whatever scored, so
# a failure in one condition and a different failure in the other produces two
# averages taken over different questions — which is not a paired comparison
# and can flip the sign of the reported delta. It did exactly that on the
# 19 August run: faithfulness read +0.058 across the two 4-question means, but
# −0.006 across the 3 questions both conditions actually scored.
#
# 4096 is comfortably above what these prompts need and well inside
# llama-3.1-8b-instant's completion limit. Setting max_tokens costs nothing on
# the free tier — it caps a single response, it does not buy quota. If you hit
# rate limits, lower the questions or add delay; don't lower this.
RAGAS_JUDGE_MAX_TOKENS = 4096


def _get_judge_llm():
    """
    A separate client for the RAGAS judge, not the one the pipeline generates
    with.

    Two reasons for keeping them apart. The judge needs a much larger
    max_tokens than generation does, and raising it on the shared client would
    change how ask() behaves — which would make this ablation's answers
    incomparable with every other result in the study. And a judge should be
    configured for consistency rather than for fluency: temperature 0 here
    against the pipeline's 0.2, so repeated runs vary as little as the API
    allows. It does not make the judge deterministic; it just stops it adding
    variance of its own on top of the generator's.
    """
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL,
        temperature=0,
        max_tokens=RAGAS_JUDGE_MAX_TOKENS,
        # Longer than the pipeline's 30s: these prompts are bigger and there
        # is no user waiting on the other end of an evaluation run.
        timeout=120,
    )


def _get_ragas_llm(llm) -> LangchainLLMWrapper:
    return LangchainLLMWrapper(llm, bypass_n=True)


def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    hf = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf)


OUTPUT_DIR = "data/results/ablation_metadata_tagging"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Identical to evaluate_llm_vs_rag.py's QUESTIONS/GROUND_TRUTHS -- not
# imported from that file, since importing it would re-execute its whole
# top-level evaluation run. Keep these two lists in sync by hand if either
# file's questions ever change.
QUESTIONS = [
    "Why are customers unhappy with battery life?",
    "What delivery problems are customers reporting?",
    "Why are airline passengers dissatisfied?",
    "What customer service issues appear most often?",
    "Summarise the major customer complaints."
]

GROUND_TRUTHS = [
    "Customers mainly complain about battery drain, charging issues and poor battery performance.",
    "Customers frequently complain about delayed deliveries, damaged parcels and missing items.",
    "Passengers commonly complain about delays, cancellations and lost luggage.",
    "Customer service complaints include rude staff, poor communication and unresolved problems.",
    "Across all platforms, the main complaints involve product quality, delivery, refunds and customer service."
]

# The "tagged" condition is rag.pipeline's own DOCUMENT_PROMPT, captured
# before anything is mutated, so it can always be restored exactly.
TAGGED_DOCUMENT_PROMPT = rag_pipeline_module.DOCUMENT_PROMPT

# The "untagged" condition: LangChain's own default document_prompt
# (create_stuff_documents_chain falls back to this when none is supplied).
# Same page_content, no [Platform: ... | Rating: ... | Sentiment: ...] prefix.
UNTAGGED_DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}",
)


def generate_rag_answer(question: str):
    # ask() never raises -- rate-limit errors inside its own try/except get
    # caught there and returned as an "An error occurred: ..." answer string
    # instead of propagating. Detect that here and retry explicitly, since
    # call_with_retry can't see an exception that never left ask(). Same
    # pattern as evaluate_llm_vs_rag.py's generate_rag_answer().
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        result = ask(question)
        answer = result["answer"]
        looks_like_rate_limit = "error occurred" in answer.lower() and (
            "rate_limit" in answer.lower() or "429" in answer
        )
        if not looks_like_rate_limit or attempt >= MAX_RATE_LIMIT_RETRIES:
            break
        wait = _extract_retry_after_seconds(Exception(answer))
        print(f"    [rate limit] ask() hit a rate limit, waiting {wait:.1f}s before retry ({attempt + 1}/{MAX_RATE_LIMIT_RETRIES})...")
        time.sleep(wait)

    return result["answer"], result["sources"]


def run_condition(label: str) -> list[dict]:
    """Run all 5 questions through ask() under whichever DOCUMENT_PROMPT is
    currently set on rag_pipeline_module, and return the raw results."""
    print("\n" + "=" * 80)
    print(f"Generating RAG answers -- condition: {label}")
    print("=" * 80)

    results = []
    for question in QUESTIONS:
        print(f"\nQuestion: {question}")

        start = time.perf_counter()
        answer, sources = generate_rag_answer(question)
        latency = (time.perf_counter() - start) * 1000

        # full_text (untruncated), same reasoning as evaluate_llm_vs_rag.py:
        # RAGAS should judge against exactly what the LLM saw, not a 200-char
        # UI preview. Note this comes from _format_sources() and does NOT
        # depend on DOCUMENT_PROMPT, so contexts are directly comparable
        # between the tagged and untagged conditions -- retrieval itself
        # never changes here.
        contexts = [source["full_text"] for source in sources]

        results.append({
            "Question": question,
            "Answer": answer,
            "Contexts": contexts,
            "Retrieved Reviews": len(contexts),
            "Latency (ms)": round(latency, 2),
        })

        print("✓ Completed")
        time.sleep(SLEEP_BETWEEN_QUESTIONS)

    return results


def build_ragas_dataset(results: list[dict]) -> Dataset:
    data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }
    for i, result in enumerate(results):
        data["user_input"].append(result["Question"])
        data["response"].append(result["Answer"])
        data["retrieved_contexts"].append(result["Contexts"])
        data["reference"].append(GROUND_TRUTHS[i])
    return Dataset.from_dict(data)


# =============================================================================
# Run both conditions
# =============================================================================

print("=" * 80)
print("Ablation: metadata tagging in the RAG context")
print("=" * 80)

# Condition A: tagged (rag.pipeline's real default -- this line is
# technically redundant since nothing has been changed yet, but it makes the
# starting state explicit rather than assumed).
rag_pipeline_module.DOCUMENT_PROMPT = TAGGED_DOCUMENT_PROMPT
tagged_results = run_condition("tagged")

# Condition B: untagged.
rag_pipeline_module.DOCUMENT_PROMPT = UNTAGGED_DOCUMENT_PROMPT
untagged_results = run_condition("untagged")

# Restore the real pipeline's default immediately after generation, before
# any RAGAS judging happens below -- nothing downstream should run against
# the untagged state by accident.
rag_pipeline_module.DOCUMENT_PROMPT = TAGGED_DOCUMENT_PROMPT

print("\nGeneration complete for both conditions.")

# =============================================================================
# Save generated answers
# =============================================================================

tagged_df = pd.DataFrame(tagged_results)
untagged_df = pd.DataFrame(untagged_results)

tagged_df.to_csv(os.path.join(OUTPUT_DIR, "tagged_answers.csv"), index=False)
untagged_df.to_csv(os.path.join(OUTPUT_DIR, "untagged_answers.csv"), index=False)

print("✓ tagged_answers.csv saved")
print("✓ untagged_answers.csv saved")

# =============================================================================
# RAGAS judge (separate Groq instance from the generation llm used inside
# ask(), same object-isolation pattern as evaluate_llm_vs_rag.py)
# =============================================================================

judge_llm = _get_judge_llm()
ragas_llm = _get_ragas_llm(judge_llm)
ragas_embeddings = _get_ragas_embeddings()
eval_run_config = RunConfig(max_workers=1, max_retries=10, max_wait=60)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# =============================================================================
# Evaluate: tagged
# =============================================================================

print("\n" + "=" * 80)
print("Evaluating condition: tagged")
print("=" * 80)

tagged_dataset = build_ragas_dataset(tagged_results)
tagged_scores = evaluate(
    dataset=tagged_dataset,
    metrics=METRICS,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=eval_run_config,
)
tagged_scores_df = tagged_scores.to_pandas()
print(tagged_scores_df)
tagged_scores_df.to_csv(os.path.join(OUTPUT_DIR, "tagged_ragas_scores.csv"), index=False)
print("✓ tagged_ragas_scores.csv saved")

# =============================================================================
# Evaluate: untagged
# =============================================================================

print("\n" + "=" * 80)
print("Evaluating condition: untagged")
print("=" * 80)

untagged_dataset = build_ragas_dataset(untagged_results)
untagged_scores = evaluate(
    dataset=untagged_dataset,
    metrics=METRICS,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=eval_run_config,
)
untagged_scores_df = untagged_scores.to_pandas()
print(untagged_scores_df)
untagged_scores_df.to_csv(os.path.join(OUTPUT_DIR, "untagged_ragas_scores.csv"), index=False)
print("✓ untagged_ragas_scores.csv saved")

# =============================================================================
# Comparison summary
# =============================================================================

print("\n" + "=" * 80)
print("Creating comparison summary")
print("=" * 80)

metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ---------------------------------------------------------------------------
# Pairing check.
#
# pandas .mean() skips NaN silently. If the judge failed on question 4 in the
# tagged condition and question 1 in the untagged one, both columns still
# produce a number — but those two numbers are averages over different
# questions, and subtracting them is not a paired comparison. On the
# 19 August run that turned a real −0.006 into a reported +0.058.
#
# So: fail loudly instead of averaging around the hole. Nothing downstream
# should ever have to guess whether a delta was computed like for like.
# ---------------------------------------------------------------------------
print()
unpaired = []
for col in metric_cols:
    t_missing = tagged_scores_df[col].isna()
    u_missing = untagged_scores_df[col].isna()
    n_missing = int(t_missing.sum() + u_missing.sum())

    if n_missing == 0:
        print(f"  {col:<20} scored in both conditions on all "
              f"{len(tagged_scores_df)} questions")
        continue

    unpaired.append(col)
    print(f"  {col:<20} UNPAIRED — judge failed on "
          f"{int(t_missing.sum())} tagged / {int(u_missing.sum())} untagged")
    for label, mask, df in (("tagged", t_missing, tagged_scores_df),
                            ("untagged", u_missing, untagged_scores_df)):
        for q in df.loc[mask, "user_input"]:
            print(f"      no {col} score ({label}): {q[:60]}")

    both = (~t_missing) & (~u_missing)
    if both.any():
        paired_delta = (tagged_scores_df.loc[both, col].mean()
                        - untagged_scores_df.loc[both, col].mean())
        naive_delta = tagged_scores_df[col].mean() - untagged_scores_df[col].mean()
        print(f"      delta over all scored rows : {naive_delta:+.4f}  <- NOT comparable")
        print(f"      delta over the {int(both.sum())} shared rows: {paired_delta:+.4f}  <- the honest figure")

if unpaired:
    print(
        "\n  WARNING: "
        + ", ".join(unpaired)
        + " cannot be compared between conditions on this run.\n"
        "  The judge did not complete on every question. Raise "
        f"RAGAS_JUDGE_MAX_TOKENS (currently {RAGAS_JUDGE_MAX_TOKENS}) and re-run\n"
        "  before quoting these deltas anywhere."
    )
else:
    print("\n  All metrics paired across both conditions — deltas are comparable.")
print()

comparison = pd.DataFrame({
    "Metric": ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Average Latency (ms)"],
    "Tagged": [
        round(tagged_scores_df["faithfulness"].mean(), 3),
        round(tagged_scores_df["answer_relevancy"].mean(), 3),
        round(tagged_scores_df["context_precision"].mean(), 3),
        round(tagged_scores_df["context_recall"].mean(), 3),
        round(tagged_df["Latency (ms)"].mean(), 2),
    ],
    "Untagged": [
        round(untagged_scores_df["faithfulness"].mean(), 3),
        round(untagged_scores_df["answer_relevancy"].mean(), 3),
        round(untagged_scores_df["context_precision"].mean(), 3),
        round(untagged_scores_df["context_recall"].mean(), 3),
        round(untagged_df["Latency (ms)"].mean(), 2),
    ],
})
comparison["Delta (Tagged - Untagged)"] = comparison.apply(
    lambda row: round(row["Tagged"] - row["Untagged"], 3)
    if isinstance(row["Tagged"], (int, float)) and isinstance(row["Untagged"], (int, float))
    else "-",
    axis=1,
)

print(comparison)
comparison.to_csv(os.path.join(OUTPUT_DIR, "comparison_summary.csv"), index=False)
print("✓ comparison_summary.csv saved")

# =============================================================================
# Graph -- grouped bar chart, all four RAGAS metrics, tagged vs untagged
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

labels = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
tagged_vals = [tagged_scores_df[c].mean() for c in metric_cols]
untagged_vals = [untagged_scores_df[c].mean() for c in metric_cols]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width / 2, tagged_vals, width, label="Tagged (current default)")
bars2 = ax.bar(x + width / 2, untagged_vals, width, label="Untagged")

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Ablation: Metadata Tagging in RAG Context")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bars in (bars1, bars2):
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                 ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metadata_tagging_ablation.png"))
plt.close()

print("\n✓ metadata_tagging_ablation.png saved")
print(f"\nAll outputs written to {OUTPUT_DIR}/")
