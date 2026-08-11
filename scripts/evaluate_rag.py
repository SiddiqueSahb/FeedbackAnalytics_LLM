"""
Evaluation suite for the FeedbackIQ RAG pipeline.
====================================================

Experiments (each writes its own CSV to data/results/evaluation_rag/):

  1. LLM-only vs RAG                                -> 1_llm_only_vs_rag.csv
  2. Top-K retrieval comparison (k = 3, 5, 10, 20)   -> 2_topk_comparison.csv
  3. Prompt engineering comparison (old vs current)  -> 3_prompt_comparison.csv
  4. Hallucination analysis (full question set)      -> 4_hallucination_analysis.csv
                                                         4_hallucination_summary.csv
  5. Retriever comparison (similarity vs MMR)        -> 5_retriever_comparison.csv
  6. Conversation memory evaluation                  -> 6_memory_evaluation.csv
  7a. Metadata-tagging ablation (with/without tags)  -> 7a_metadata_tagging_ablation.csv
  7b. Out-of-scope guard accuracy                     -> 7b_scope_guard_accuracy.csv
                                                         7b_scope_guard_summary.csv
  7c. Latency comparison across configurations        -> 7c_latency_comparison.csv

  A consolidated summary.csv is written at the end with headline numbers
  from every experiment, for quick reference in the dissertation.

RAGAS COMPATIBILITY NOTE
-------------------------
ragas 0.4.x unconditionally imports `langchain_community.chat_models.vertexai`
at module load. That submodule no longer exists in this project's installed
langchain-community (post LangChain 1.0 rebrand — only llms/vertexai.py and
embeddings/vertexai.py remain). We never use VertexAI, so the small shim
below stubs that module out *before* ragas is imported, letting the rest of
ragas load normally. This was verified against the actually-installed
package versions in this project before writing the rest of this file.
Do not remove the shim unless you upgrade ragas past the point it drops the
VertexAI import, or intentionally downgrade langchain-community (not
recommended — langchain_classic / langchain_groq / langchain_huggingface all
depend on the current 1.x API surface your working pipeline already uses).

METRIC SCOPE NOTE
------------------
Only Faithfulness and AnswerRelevancy are used from RAGAS.

ContextPrecision and ContextRecall are intentionally excluded: both require
a hand-labeled "reference" (ground-truth) answer per question, which this
evaluation does not have. Faithfulness (does every claim trace back to
retrieved evidence — the direct hallucination signal) and AnswerRelevancy
(does the answer actually address the question) are the two RAGAS metrics
that work without ground truth, which is why they were chosen here. Adding a
small hand-labeled reference set (10-15 questions with a written "gold"
answer) is a natural next step if you want ContextPrecision/ContextRecall
numbers too — flagged again at the bottom of this file.

In addition to the RAGAS metrics, this script adds two custom, deterministic
checks that don't depend on an LLM judge at all:
  - platform_hallucination: parses which platforms the answer *names* and
    compares that against the platforms actually present in the retrieved
    documents' metadata. This is the concrete, measurable version of "did
    the model cite a platform that wasn't in front of it."
  - fabrication_risk: a small LLM-judge (not RAGAS) asking whether the
    answer states specific facts/stats/quotes that aren't verifiable from
    common knowledge — used mainly for the LLM-only condition in
    experiment 1, where Faithfulness is undefined (no retrieved context to
    check groundedness against).

JUDGE MODEL
-----------
Reuses the same Groq model configured for generation (settings.GROQ_MODEL,
currently llama-3.1-8b-instant) as the RAGAS judge. Using the same model as
both generator and judge is a documented limitation (self-grading bias) —
report it as such if you keep this setup for the dissertation. Swapping in
an independent judge model later only requires changing `_get_ragas_llm`.

USAGE
-----
    python scripts/evaluate_rag.py                # run everything
    python scripts/evaluate_rag.py --fast          # tiny question subset, quick smoke test
    python scripts/evaluate_rag.py --only 1,4,7b   # run specific experiments only

Runtime warning: the full run makes several hundred Groq API calls (every
RAGAS metric call is itself 1-4 LLM calls). On Groq's free tier this can
take a while and may hit rate limits — start with --fast to confirm
everything works end to end before committing to a full run.
"""

from __future__ import annotations

import sys
import os
import re
import time
import argparse
import warnings
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# ── ragas / langchain-community compatibility shim (see module docstring) ────
import types as _types

_vertexai_shim = _types.ModuleType("langchain_community.chat_models.vertexai")


class _StubChatVertexAI:  # pragma: no cover - never actually instantiated
    pass


_vertexai_shim.ChatVertexAI = _StubChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_shim)

import langchain_community.llms as _langchain_community_llms  # noqa: E402

if not hasattr(_langchain_community_llms, "VertexAI"):
    class _StubVertexAI:  # pragma: no cover
        pass

    _langchain_community_llms.VertexAI = _StubVertexAI


# These specific warnings are suppressed deliberately, not blanket-hidden:
#
# 1. ragas.metrics.{Faithfulness,AnswerRelevancy} / LangchainLLMWrapper /
#    LangchainEmbeddingsWrapper are all flagged deprecated in favour of
#    ragas.metrics.collections + llm_factory(provider="groq", ...). That
#    newer path was tried and found broken for Groq in ragas==0.4.3: its
#    generic instructor adapter (ragas/llms/base.py:_patch_client_for_provider)
#    hardcodes `client.messages.create` (Anthropic's method shape) for every
#    non-Gemini provider, including "groq" -- it raises
#    AttributeError: 'Groq' object has no attribute 'messages' before any
#    network call is made. Confirmed by direct construction, not assumed.
#    Until ragas ships a fix, the classic API used below is the only one
#    that actually works with Groq.
# 2. The langchain-community "sunset" warning fires because ragas.llms.base
#    itself unconditionally imports langchain_community.llms.VertexAI at
#    module load (the same import our vertexai shim above has to stub out).
#    That's internal to ragas 0.4.3, not something this project controls.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
warnings.filterwarnings("ignore", message=".*ragas.metrics.collections.*")
warnings.filterwarnings("ignore", message=".*LangchainLLMWrapper is deprecated.*")
warnings.filterwarnings("ignore", message=".*LangchainEmbeddingsWrapper is deprecated.*")

from ragas import evaluate as ragas_evaluate, EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.metrics import Faithfulness, AnswerRelevancy  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain  # noqa: E402
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402

from config import settings  # noqa: E402
from rag.pipeline import (  # noqa: E402
    _get_llm,
    _get_vectorstore,
    _build_history_string,
    _format_chat_history_text,
    DOCUMENT_PROMPT,
)
from rag.prompts import (  # noqa: E402
    RAG_PROMPT,
    CONDENSE_QUESTION_PROMPT,
    is_complaint_question,
    is_complaint_question_llm,
)

# ── Output location ───────────────────────────────────────────────────────────
OUTPUT_DIR = "data/results/evaluation_rag"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Groq's free tier for llama-3.1-8b-instant caps at 6000 tokens/minute (100
# tokens/sec sustained). A single RAG call with tagged review context plus
# its RAGAS scoring pass (Faithfulness + AnswerRelevancy each make their own
# sub-calls reprocessing that same context) can easily run 1500-3000+ tokens
# combined -- comfortably enough to blow the per-minute budget on its own if
# calls are back-to-back. 6s of headroom between calls is a deliberate
# pacing choice, not just politeness; call_with_retry (below) is the
# backstop for whatever this pacing doesn't catch.
SLEEP_BETWEEN_CALLS = 6.0
MAX_RATE_LIMIT_RETRIES = 6


def _extract_retry_after_seconds(exc: Exception, default: float = 8.0) -> float:
    """
    Groq 429 responses include a hint like 'Please try again in 2.82s' in
    the error message; use it when present so we wait exactly as long as
    needed instead of guessing. Falls back to `default` if the message
    doesn't match (format could change, or a different provider's error).
    """
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 0.5  # small safety margin
        except ValueError:
            pass
    return default


def call_with_retry(fn, *args, max_retries: int = MAX_RATE_LIMIT_RETRIES, **kwargs):
    """
    Call fn(*args, **kwargs), retrying with backoff on rate-limit errors
    instead of letting them crash the whole run. Groq's free tier caps at a
    few thousand tokens/minute for llama-3.1-8b-instant -- a full evaluation
    sweep (several hundred calls, many of them multi-call RAGAS metrics)
    hits this routinely. Only rate-limit-shaped errors are retried; anything
    else re-raises immediately.
    """
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


# =============================================================================
# Question banks
# =============================================================================

# Core in-scope questions, reused across every configuration sweep
# (top-k, retriever type, prompt version, metadata-tagging ablation) so those
# experiments stay comparable to each other without exploding the call count.
CORE_QUESTIONS = [
    {"id": "c01", "question": "What are the top customer complaints on Amazon?", "platform_hint": "amazon"},
    {"id": "c02", "question": "Why are customers unhappy with Amazon deliveries?", "platform_hint": "amazon"},
    {"id": "c03", "question": "What do Yelp reviewers complain about most often?", "platform_hint": "yelp"},
    {"id": "c04", "question": "Are customers complaining about staff behavior on Yelp?", "platform_hint": "yelp"},
    {"id": "c05", "question": "What are Twitter users saying about airline delays?", "platform_hint": "twitter_airline"},
]


# Extra questions used only in the broad experiments (1: LLM-only vs RAG,
# 4: hallucination analysis), where breadth across edge cases matters more
# than holding a config fixed.
EDGE_CASE_QUESTIONS = [
    {"id": "e01", "question": "Tell me a joke about airlines.", "category": "out_of_scope"},
    {"id": "e02", "question": "What's the weather like today?", "category": "out_of_scope"},
    {"id": "e03", "question": "What do customers say about the color options of products on TikTok Shop?", "category": "thin_evidence"},
    
]

FULL_QUESTIONS = (
    [{**q, "category": "in_scope"} for q in CORE_QUESTIONS] + EDGE_CASE_QUESTIONS
)

# Multi-turn pairs for the conversation-memory experiment. Turn 2 only makes
# sense if the model can resolve "those" / "that" against turn 1.
MEMORY_PAIRS = [
    {
        "id": "m01",
        "turn1": "What are the top complaints about Amazon?",
        "turn2": "Which of those is most urgent to fix?",
    },
    {
        "id": "m02",
        "turn1": "What are people saying about airline delays on Twitter?",
        "turn2": "Is that the same issue on Yelp?",
    },
]

# Labeled set for the out-of-scope guard accuracy experiment.
SCOPE_GUARD_SET = [
    {"question": "What are the top complaints about Amazon?", "label": "in_scope"},
    {"question": "Why are customers unhappy with Yelp restaurants?", "label": "in_scope"},
    {"question": "What is the history of complaints about Amazon shipping?", "label": "in_scope"},
    {"question": "Which airline complaint shows up most on Twitter?", "label": "in_scope"},
    {"question": "Do customers complain about billing errors?", "label": "in_scope"},
    {"question": "What do people think about delivery speed?", "label": "in_scope"},
    {"question": "Summarize negative sentiment about customer service.", "label": "in_scope"},
    {"question": "What ratings do the worst reviews usually have?", "label": "in_scope"},
    {"question": "Tell me a joke about airlines.", "label": "out_of_scope"},
    {"question": "What's the weather like today?", "label": "out_of_scope"},
    {"question": "Write me a short story about a dragon.", "label": "out_of_scope"},
    {"question": "Who won the last World Cup?", "label": "out_of_scope"},
    {"question": "Can you write some Python code to sort a list?", "label": "out_of_scope"},
    {"question": "Give me a recipe for chocolate cake.", "label": "out_of_scope"},
    {"question": "What's your favorite poem?", "label": "out_of_scope"},
    {"question": "What's the political situation in the news today?", "label": "out_of_scope"},
]

# Verbatim snapshot of the RAG prompt *before* the grounding/anti-hallucination
# rewrite, kept here only so experiment 3 can measure the delta.
OLD_RAG_PROMPT_TEMPLATE = """You are a customer complaints analyst for FeedbackIQ.

YOUR SPECIFIC ROLE:
- Help business analysts explore and understand customer complaint patterns
- Answer questions about what customers are complaining about, why, and how often
- Ground every claim in the retrieved reviews provided below
- Stay focused on complaint analysis — do not answer unrelated questions

STRICT RULES:

1. Base every answer only on the retrieved customer reviews.
2. Do not use outside knowledge or make assumptions.
3. Mention the platform (Amazon, Yelp or Twitter Airline) whenever presenting an insight.
4. If the retrieved reviews do not contain enough evidence, clearly state that.
5. Do not invent statistics, trends or complaint categories.
6. Finish every answer with one clear and actionable business recommendation.
Retrieved Customer Reviews:
{context}

Business Analyst Question: {input}

Structured Answer:
[Platform insights — cite Amazon/Yelp/Twitter Airline]

[Key complaint patterns found in reviews]

[Business recommendation — one specific, measurable action]"""

# input_variables use "input" (not the original "question") purely so this
# historical prompt can be replayed through the current create_stuff_documents_chain
# plumbing, which hardcodes "input" as the query key. The wording/content is
# otherwise the verbatim original.
OLD_RAG_PROMPT = PromptTemplate(
    template=OLD_RAG_PROMPT_TEMPLATE,
    input_variables=["context", "input"],
)

# Plain document formatter (page_content only) — mimics the pipeline's
# behaviour *before* DOCUMENT_PROMPT was added, for experiment 7a.
UNTAGGED_DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}",
)

REFUSAL_STRING = "do not contain enough evidence"

PLATFORM_KEYWORDS = {
    "amazon": ["amazon"],
    "yelp": ["yelp"],
    "twitter_airline": ["twitter airline", "twitter", "airline"],
}


# =============================================================================
# Small helpers shared across experiments
# =============================================================================

def _get_ragas_llm(llm) -> LangchainLLMWrapper:
    # bypass_n=True: ragas otherwise mutates llm.n in place to request
    # multiple completions per call (used internally by AnswerRelevancy).
    # Groq's API rejects n>1 outright ("'n' : number must be at most 1"),
    # and ragas only resets .n on the success path -- if the call errors
    # (as it does against Groq), the mutation is never undone. Combined
    # with using a dedicated judge LLM instance below (never the same
    # object as the generation `llm`), this fully isolates the RAG chain
    # from anything ragas does to its judge.
    return LangchainLLMWrapper(llm, bypass_n=True)


def _get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    hf = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf)


def extract_mentioned_platforms(text: str) -> set[str]:
    """Which platforms does the answer *name*, based on simple keyword matching."""
    text_lower = (text or "").lower()
    found = set()
    for canon, keywords in PLATFORM_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.add(canon)
    return found


def get_retrieved_platforms(source_documents: list) -> set[str]:
    return {
        str(doc.metadata.get("platform", "")).lower()
        for doc in source_documents
        if doc.metadata.get("platform")
    }


def platform_hallucination(answer: str, retrieved_platforms: set[str]) -> bool:
    """True if the answer names a platform that wasn't actually retrieved."""
    mentioned = extract_mentioned_platforms(answer)
    return bool(mentioned - retrieved_platforms)


def is_refusal(answer: str) -> bool:
    return REFUSAL_STRING in (answer or "").lower()


FABRICATION_JUDGE_TEMPLATE = """You are a strict fact-checker.

Question: {question}
Answer: {answer}

Does this answer state specific facts, statistics, percentages, named
sources, or direct quotes that are NOT common knowledge and would need a
verifiable source to confirm? Reply with exactly one word: YES or NO."""


def judge_fabrication_risk(llm, question: str, answer: str) -> Optional[int]:
    """1 = answer makes unverifiable specific claims, 0 = it doesn't, None = judge call failed."""
    try:
        prompt_text = FABRICATION_JUDGE_TEMPLATE.format(question=question, answer=answer)
        response = call_with_retry(llm.invoke, prompt_text)
        text = getattr(response, "content", str(response)).strip().upper()
        return 1 if text.startswith("YES") else 0
    except Exception as exc:
        print(f"    [warn] fabrication judge failed: {exc}")
        return None


def score_with_ragas(
    ragas_llm,
    ragas_embeddings,
    question: str,
    answer: str,
    contexts: list[str],
    metrics=("faithfulness", "answer_relevancy"),
    retries: int = MAX_RATE_LIMIT_RETRIES,
) -> dict:
    """
    Score one Q/A/context triple with RAGAS. Returns a dict with whatever
    metric keys were requested, values are float or None on failure.
    contexts=[] disables faithfulness (it needs evidence to check against) —
    only answer_relevancy is computed in that case.
    """
    out = {m: None for m in metrics}
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts if contexts else [""],
    )
    metric_objs = []
    if "faithfulness" in metrics and contexts:
        metric_objs.append(Faithfulness())
    if "answer_relevancy" in metrics:
        metric_objs.append(AnswerRelevancy())

    if not metric_objs:
        return out

    for attempt in range(retries + 1):
        try:
            ds = EvaluationDataset(samples=[sample])
            result = ragas_evaluate(
                dataset=ds,
                metrics=metric_objs,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=True,
                show_progress=False,
                # max_workers=1: ragas defaults to 16 concurrent workers. With
                # only one sample this mainly matters for Faithfulness and
                # AnswerRelevancy firing their sub-calls at the same time --
                # serializing them spreads token usage over more wall-clock
                # time instead of bursting, which is the whole point of the
                # SLEEP_BETWEEN_CALLS pacing above.
                run_config=RunConfig(max_workers=1, max_retries=10, max_wait=60),
            )
            row = result.to_pandas().iloc[0].to_dict()
            if "faithfulness" in metrics and contexts:
                out["faithfulness"] = row.get("faithfulness")
            if "answer_relevancy" in metrics:
                out["answer_relevancy"] = row.get("answer_relevancy")
            return out
        except Exception as exc:
            is_rate_limit = "rate_limit" in str(exc).lower() or "429" in str(exc)
            if attempt < retries and is_rate_limit:
                wait = _extract_retry_after_seconds(exc)
                print(f"    [rate limit] ragas judge waiting {wait:.1f}s before retry ({attempt + 1}/{retries})...")
                time.sleep(wait)
                continue
            if attempt < retries:
                time.sleep(3)
                continue
            print(f"    [warn] ragas scoring failed: {exc}")
            return out
    return out


def build_chain(
    llm,
    vectorstore,
    k: int = 5,
    search_type: str = "similarity",
    fetch_k: int = 20,
    prompt: PromptTemplate = None,
    document_prompt: PromptTemplate = None,
):
    """
    Parameterized rebuild of the LCEL chain construction in rag/pipeline.py's
    ask() -- create_history_aware_retriever + create_stuff_documents_chain +
    create_retrieval_chain. No memory object: each experiment passes whatever
    chat_history string it wants (usually "") straight into run_rag().
    """
    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    history_aware_retriever = create_history_aware_retriever(llm, retriever, CONDENSE_QUESTION_PROMPT)
    combine_docs_chain = create_stuff_documents_chain(
        llm, prompt or RAG_PROMPT, document_prompt=document_prompt or DOCUMENT_PROMPT
    )
    return create_retrieval_chain(history_aware_retriever, combine_docs_chain)


def run_rag(chain, question: str, chat_history_text: str = "") -> dict:
    """
    chat_history_text: plain-text block from rag.pipeline._format_chat_history_text,
    or "" for a single-turn question. Note the return key is still
    "source_documents" (not "context") purely so every experiment function
    below didn't need touching when the chain moved to LCEL -- the value now
    comes from result["context"], LCEL's key name for retrieved documents.
    """
    t0 = time.perf_counter()
    result = call_with_retry(chain.invoke, {"input": question, "chat_history": chat_history_text})
    elapsed = time.perf_counter() - t0
    return {
        "answer": result["answer"],
        "source_documents": result.get("context", []),
        "latency_sec": elapsed,
    }


def run_llm_only(llm, question: str) -> dict:
    prompt_text = (
        "Answer the following question about customer complaints, using your "
        f"own general knowledge (no retrieved documents are provided):\n\n{question}"
    )
    t0 = time.perf_counter()
    response = call_with_retry(llm.invoke, prompt_text)
    elapsed = time.perf_counter() - t0
    answer = getattr(response, "content", str(response))
    return {"answer": answer, "latency_sec": elapsed}


def save_csv(rows: list[dict], filename: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  saved {path} ({len(df)} rows)")
    return df


# =============================================================================
# Experiment 1 — LLM-only vs RAG
# =============================================================================

def experiment_1_llm_only_vs_rag(llm, vectorstore, ragas_llm, ragas_emb, questions):
    print("\n=== Experiment 1: LLM-only vs RAG ===")
    chain = build_chain(llm, vectorstore, k=5, search_type="similarity", prompt=RAG_PROMPT, document_prompt=DOCUMENT_PROMPT)
    rows = []
    for q in questions:
        print(f"  [{q['id']}] {q['question'][:60]}")

        llm_only = run_llm_only(llm, q["question"])
        llm_only_scores = score_with_ragas(ragas_llm, ragas_emb, q["question"], llm_only["answer"], contexts=[])
        fab_risk = judge_fabrication_risk(llm, q["question"], llm_only["answer"])
        rows.append({
            "question_id": q["id"], "question": q["question"], "condition": "llm_only",
            "answer": llm_only["answer"],
            "retrieved_platforms": "", "mentioned_platforms": ",".join(sorted(extract_mentioned_platforms(llm_only["answer"]))),
            "platform_hallucination": platform_hallucination(llm_only["answer"], set()),
            "faithfulness": None, "answer_relevancy": llm_only_scores["answer_relevancy"],
            "fabrication_risk": fab_risk, "latency_sec": llm_only["latency_sec"],
        })
        time.sleep(SLEEP_BETWEEN_CALLS)

        rag = run_rag(chain, q["question"])
        contexts = [d.page_content for d in rag["source_documents"]]
        retrieved_platforms = get_retrieved_platforms(rag["source_documents"])
        rag_scores = score_with_ragas(ragas_llm, ragas_emb, q["question"], rag["answer"], contexts=contexts)
        rows.append({
            "question_id": q["id"], "question": q["question"], "condition": "rag",
            "answer": rag["answer"],
            "retrieved_platforms": ",".join(sorted(retrieved_platforms)),
            "mentioned_platforms": ",".join(sorted(extract_mentioned_platforms(rag["answer"]))),
            "platform_hallucination": platform_hallucination(rag["answer"], retrieved_platforms),
            "faithfulness": rag_scores["faithfulness"], "answer_relevancy": rag_scores["answer_relevancy"],
            "fabrication_risk": None, "latency_sec": rag["latency_sec"],
        })
        time.sleep(SLEEP_BETWEEN_CALLS)

    return save_csv(rows, "1_llm_only_vs_rag.csv")


# =============================================================================
# Experiment 2 — Top-K comparison
# =============================================================================

def experiment_2_topk_comparison(llm, vectorstore, ragas_llm, ragas_emb, questions, k_values=(3, 5, 10, 20)):
    print("\n=== Experiment 2: Top-K retrieval comparison ===")
    rows = []
    for k in k_values:
        chain = build_chain(llm, vectorstore, k=k, search_type="similarity", prompt=RAG_PROMPT, document_prompt=DOCUMENT_PROMPT)
        for q in questions:
            print(f"  [k={k}] [{q['id']}] {q['question'][:50]}")
            rag = run_rag(chain, q["question"])
            contexts = [d.page_content for d in rag["source_documents"]]
            retrieved_platforms = get_retrieved_platforms(rag["source_documents"])
            scores = score_with_ragas(ragas_llm, ragas_emb, q["question"], rag["answer"], contexts=contexts)
            rows.append({
                "question_id": q["id"], "question": q["question"], "k": k,
                "answer": rag["answer"], "num_retrieved": len(rag["source_documents"]),
                "unique_platforms_retrieved": len(retrieved_platforms),
                "faithfulness": scores["faithfulness"], "answer_relevancy": scores["answer_relevancy"],
                "platform_hallucination": platform_hallucination(rag["answer"], retrieved_platforms),
                "latency_sec": rag["latency_sec"],
            })
            # Larger k stuffs more retrieved reviews into context -- more
            # tokens per call and per RAGAS sub-call. Scale the pause with k
            # instead of dropping k=20 from the sweep entirely: k=20 gets an
            # extra ~6s on top of the baseline, k=3 barely any.
            time.sleep(SLEEP_BETWEEN_CALLS + k * 0.3)
    return save_csv(rows, "2_topk_comparison.csv")


# =============================================================================
# Experiment 3 — Prompt engineering comparison (old vs current)
# =============================================================================

def experiment_3_prompt_comparison(llm, vectorstore, ragas_llm, ragas_emb, questions):
    print("\n=== Experiment 3: Prompt engineering comparison ===")
    rows = []
    for label, prompt in (("old", OLD_RAG_PROMPT), ("current", RAG_PROMPT)):
        chain = build_chain(llm, vectorstore, k=5, search_type="similarity", prompt=prompt, document_prompt=DOCUMENT_PROMPT)
        for q in questions:
            print(f"  [{label}] [{q['id']}] {q['question'][:50]}")
            rag = run_rag(chain, q["question"])
            contexts = [d.page_content for d in rag["source_documents"]]
            retrieved_platforms = get_retrieved_platforms(rag["source_documents"])
            scores = score_with_ragas(ragas_llm, ragas_emb, q["question"], rag["answer"], contexts=contexts)
            rows.append({
                "question_id": q["id"], "question": q["question"], "prompt_version": label,
                "answer": rag["answer"],
                "faithfulness": scores["faithfulness"], "answer_relevancy": scores["answer_relevancy"],
                "platform_hallucination": platform_hallucination(rag["answer"], retrieved_platforms),
                "latency_sec": rag["latency_sec"],
            })
            time.sleep(SLEEP_BETWEEN_CALLS)
    return save_csv(rows, "3_prompt_comparison.csv")


# =============================================================================
# Experiment 4 — Hallucination analysis (broad question set incl. edge cases)
# =============================================================================

def experiment_4_hallucination_analysis(llm, vectorstore, ragas_llm, ragas_emb, questions):
    print("\n=== Experiment 4: Hallucination analysis ===")
    chain = build_chain(llm, vectorstore, k=5, search_type="similarity", prompt=RAG_PROMPT, document_prompt=DOCUMENT_PROMPT)
    rows = []
    for q in questions:
        print(f"  [{q['id']}] ({q.get('category', '?')}) {q['question'][:50]}")
        rag = run_rag(chain, q["question"])
        contexts = [d.page_content for d in rag["source_documents"]]
        retrieved_platforms = get_retrieved_platforms(rag["source_documents"])
        scores = score_with_ragas(ragas_llm, ragas_emb, q["question"], rag["answer"], contexts=contexts)
        fab_risk = judge_fabrication_risk(llm, q["question"], rag["answer"])
        rows.append({
            "question_id": q["id"], "question": q["question"], "category": q.get("category", "in_scope"),
            "answer": rag["answer"], "is_refusal": is_refusal(rag["answer"]),
            "platform_hallucination": platform_hallucination(rag["answer"], retrieved_platforms),
            "faithfulness": scores["faithfulness"], "answer_relevancy": scores["answer_relevancy"],
            "fabrication_risk": fab_risk, "latency_sec": rag["latency_sec"],
        })
        time.sleep(SLEEP_BETWEEN_CALLS)

    df = save_csv(rows, "4_hallucination_analysis.csv")

    thin = df[df["category"] == "thin_evidence"]
    answerable = df[df["category"] == "in_scope"]
    summary = [{
        "n_questions": len(df),
        "platform_hallucination_rate": df["platform_hallucination"].mean(),
        "mean_faithfulness": df["faithfulness"].dropna().mean() if df["faithfulness"].notna().any() else None,
        "mean_fabrication_risk": df["fabrication_risk"].dropna().mean() if df["fabrication_risk"].notna().any() else None,
        "thin_evidence_refusal_rate": thin["is_refusal"].mean() if len(thin) else None,
        "answerable_false_refusal_rate": (
            answerable["is_refusal"].mean() if len(answerable) else None
        ),
    }]
    save_csv(summary, "4_hallucination_summary.csv")
    return df


# =============================================================================
# Experiment 5 — Retriever comparison (similarity vs MMR)
# =============================================================================

def experiment_5_retriever_comparison(llm, vectorstore, ragas_llm, ragas_emb, questions):
    print("\n=== Experiment 5: Retriever comparison (similarity vs MMR) ===")
    rows = []
    configs = [("similarity", {}), ("mmr", {"fetch_k": 20})]
    for search_type, extra in configs:
        chain = build_chain(
            llm, vectorstore, k=5, search_type=search_type,
            fetch_k=extra.get("fetch_k", 20), prompt=RAG_PROMPT, document_prompt=DOCUMENT_PROMPT,
        )
        for q in questions:
            print(f"  [{search_type}] [{q['id']}] {q['question'][:50]}")
            rag = run_rag(chain, q["question"])
            contexts = [d.page_content for d in rag["source_documents"]]
            retrieved_platforms = get_retrieved_platforms(rag["source_documents"])
            scores = score_with_ragas(ragas_llm, ragas_emb, q["question"], rag["answer"], contexts=contexts)
            rows.append({
                "question_id": q["id"], "question": q["question"], "retriever_type": search_type,
                "answer": rag["answer"], "num_retrieved": len(rag["source_documents"]),
                "unique_platforms_retrieved": len(retrieved_platforms),
                "faithfulness": scores["faithfulness"], "answer_relevancy": scores["answer_relevancy"],
                "platform_hallucination": platform_hallucination(rag["answer"], retrieved_platforms),
                "latency_sec": rag["latency_sec"],
            })
            time.sleep(SLEEP_BETWEEN_CALLS)
    return save_csv(rows, "5_retriever_comparison.csv")


# =============================================================================
# Experiment 6 — Conversation memory evaluation
# =============================================================================

def experiment_6_memory_evaluation(llm, vectorstore, ragas_llm, ragas_emb, pairs):
    print("\n=== Experiment 6: Conversation memory evaluation ===")
    rows = []
    # One shared chain is fine here -- there's no stateful memory object in
    # the LCEL chain anymore (see rag/pipeline.py's ask()); chat_history is
    # just a plain text argument passed per call, threaded manually below.
    chain = build_chain(llm, vectorstore, k=5, prompt=RAG_PROMPT, document_prompt=DOCUMENT_PROMPT)
    for pair in pairs:
        print(f"  [{pair['id']}] turn1='{pair['turn1'][:40]}' turn2='{pair['turn2'][:40]}'")

        # --- turn 1, no history yet ---
        turn1 = run_rag(chain, pair["turn1"])
        time.sleep(SLEEP_BETWEEN_CALLS)

        # --- with memory: turn 2 sees turn 1's Q&A as chat_history text ---
        hist_text = _format_chat_history_text(_build_history_string([
            {"role": "user", "content": pair["turn1"]},
            {"role": "assistant", "content": turn1["answer"]},
        ]))
        turn2_with_memory = run_rag(chain, pair["turn2"], chat_history_text=hist_text)
        contexts_mem = [d.page_content for d in turn2_with_memory["source_documents"]]
        scores_mem = score_with_ragas(ragas_llm, ragas_emb, pair["turn2"], turn2_with_memory["answer"], contexts=contexts_mem)

        # --- without memory: turn 2 asked cold, chat_history="" ---
        time.sleep(SLEEP_BETWEEN_CALLS)
        turn2_no_memory = run_rag(chain, pair["turn2"], chat_history_text="")
        contexts_nomem = [d.page_content for d in turn2_no_memory["source_documents"]]
        scores_nomem = score_with_ragas(ragas_llm, ragas_emb, pair["turn2"], turn2_no_memory["answer"], contexts=contexts_nomem)

        # crude grounding-in-history heuristic: does turn2's answer share any
        # platform tag with turn1's retrieved context?
        turn1_platforms = get_retrieved_platforms(turn1["source_documents"])

        for cond, turn2_result, scores in (
            ("with_memory", turn2_with_memory, scores_mem),
            ("without_memory", turn2_no_memory, scores_nomem),
        ):
            mentioned = extract_mentioned_platforms(turn2_result["answer"])
            rows.append({
                "pair_id": pair["id"], "turn1_question": pair["turn1"], "turn2_question": pair["turn2"],
                "memory_condition": cond, "turn2_answer": turn2_result["answer"],
                "shares_platform_with_turn1": bool(mentioned & turn1_platforms) if turn1_platforms else None,
                "faithfulness": scores["faithfulness"], "answer_relevancy": scores["answer_relevancy"],
                "latency_sec": turn2_result["latency_sec"],
            })
        time.sleep(SLEEP_BETWEEN_CALLS)
    return save_csv(rows, "6_memory_evaluation.csv")


# =============================================================================
# Experiment 7a — Metadata-tagging ablation
# =============================================================================

def experiment_7a_metadata_tagging_ablation(llm, vectorstore, ragas_llm, ragas_emb, questions):
    print("\n=== Experiment 7a: Metadata-tagging ablation ===")
    rows = []
    for label, doc_prompt in (("tagged", DOCUMENT_PROMPT), ("untagged", UNTAGGED_DOCUMENT_PROMPT)):
        chain = build_chain(llm, vectorstore, k=5, prompt=RAG_PROMPT, document_prompt=doc_prompt)
        for q in questions:
            print(f"  [{label}] [{q['id']}] {q['question'][:50]}")
            rag = run_rag(chain, q["question"])
            contexts = [d.page_content for d in rag["source_documents"]]
            retrieved_platforms = get_retrieved_platforms(rag["source_documents"])
            scores = score_with_ragas(ragas_llm, ragas_emb, q["question"], rag["answer"], contexts=contexts)
            rows.append({
                "question_id": q["id"], "question": q["question"], "tagging": label,
                "answer": rag["answer"],
                "platform_hallucination": platform_hallucination(rag["answer"], retrieved_platforms),
                "faithfulness": scores["faithfulness"], "latency_sec": rag["latency_sec"],
            })
            time.sleep(SLEEP_BETWEEN_CALLS)
    return save_csv(rows, "7a_metadata_tagging_ablation.csv")


# =============================================================================
# Experiment 7b — Out-of-scope guard accuracy
# =============================================================================

def experiment_7b_scope_guard_accuracy(llm, labeled_questions):
    print("\n=== Experiment 7b: Out-of-scope guard accuracy ===")
    rows = []
    for item in labeled_questions:
        q, true_label = item["question"], item["label"]
        keyword_pred = "in_scope" if is_complaint_question(q) else "out_of_scope"
        llm_pred = "in_scope" if is_complaint_question_llm(q, llm=llm) else "out_of_scope"
        rows.append({
            "question": q, "true_label": true_label,
            "keyword_guard_pred": keyword_pred, "keyword_correct": keyword_pred == true_label,
            "llm_guard_pred": llm_pred, "llm_correct": llm_pred == true_label,
        })
        time.sleep(SLEEP_BETWEEN_CALLS)

    df = save_csv(rows, "7b_scope_guard_accuracy.csv")

    def _metrics(pred_col):
        correct = df[pred_col.replace("_pred", "_correct")]
        pos = df[df["true_label"] == "out_of_scope"]
        false_positive = (pos[pred_col] == "in_scope").mean() if len(pos) else None  # missed an off-topic Q
        neg = df[df["true_label"] == "in_scope"]
        false_negative = (neg[pred_col] == "out_of_scope").mean() if len(neg) else None  # wrongly rejected
        return correct.mean(), false_positive, false_negative

    kw_acc, kw_fp, kw_fn = _metrics("keyword_guard_pred")
    llm_acc, llm_fp, llm_fn = _metrics("llm_guard_pred")
    summary = [
        {"method": "keyword_heuristic", "accuracy": kw_acc, "off_topic_missed_rate": kw_fp, "in_scope_wrongly_rejected_rate": kw_fn},
        {"method": "llm_classifier", "accuracy": llm_acc, "off_topic_missed_rate": llm_fp, "in_scope_wrongly_rejected_rate": llm_fn},
    ]
    save_csv(summary, "7b_scope_guard_summary.csv")
    return df


# =============================================================================
# Experiment 7c — Latency comparison across configurations
# =============================================================================

def experiment_7c_latency_comparison(topk_df: pd.DataFrame, retriever_df: pd.DataFrame):
    print("\n=== Experiment 7c: Latency comparison across configurations ===")
    rows = []
    if topk_df is not None and len(topk_df):
        for k, g in topk_df.groupby("k"):
            rows.append({"config": f"top_k={k}", "mean_latency_sec": g["latency_sec"].mean(), "n_samples": len(g)})
    if retriever_df is not None and len(retriever_df):
        for rtype, g in retriever_df.groupby("retriever_type"):
            rows.append({"config": f"retriever={rtype}", "mean_latency_sec": g["latency_sec"].mean(), "n_samples": len(g)})
    return save_csv(rows, "7c_latency_comparison.csv")


# =============================================================================
# Consolidated summary — headline numbers from every CSV just written
# =============================================================================

NUMERIC_SUMMARY_COLUMNS = (
    "faithfulness", "answer_relevancy", "fabrication_risk",
    "platform_hallucination", "is_refusal", "latency_sec",
    "accuracy", "off_topic_missed_rate", "in_scope_wrongly_rejected_rate",
    "mean_latency_sec",
)


def build_overall_summary():
    """Scan every CSV just written to OUTPUT_DIR and pull mean values for the
    numeric columns that matter, into one small reference table."""
    rows = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if not fname.endswith(".csv") or fname == "summary.csv":
            continue
        try:
            df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
        except Exception:
            continue
        row = {"file": fname, "n_rows": len(df)}
        for col in NUMERIC_SUMMARY_COLUMNS:
            if col in df.columns:
                row[f"mean_{col}"] = pd.to_numeric(df[col], errors="coerce").mean()
        rows.append(row)
    return save_csv(rows, "summary.csv")


# =============================================================================
# Main orchestration
# =============================================================================

EXPERIMENTS = {
    "1": "LLM-only vs RAG",
    "2": "Top-K comparison",
    "3": "Prompt comparison",
    "4": "Hallucination analysis",
    "5": "Retriever comparison",
    "6": "Memory evaluation",
    "7a": "Metadata-tagging ablation",
    "7b": "Scope guard accuracy",
    "7c": "Latency comparison",
}


def main():
    parser = argparse.ArgumentParser(description="FeedbackIQ RAG evaluation suite")
    parser.add_argument("--fast", action="store_true", help="Use a tiny question subset for a quick smoke test")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated experiment ids to run, e.g. 1,4,7b")
    args = parser.parse_args()

    selected = set(args.only.split(",")) if args.only else set(EXPERIMENTS.keys())

    core_qs = CORE_QUESTIONS[:3] if args.fast else CORE_QUESTIONS
    full_qs = FULL_QUESTIONS[:6] if args.fast else FULL_QUESTIONS
    mem_pairs = MEMORY_PAIRS[:1] if args.fast else MEMORY_PAIRS
    scope_set = SCOPE_GUARD_SET[:6] if args.fast else SCOPE_GUARD_SET
    k_values = (3, 5) if args.fast else (3, 5, 10, 20)

    print("Loading LLM, vectorstore, and RAGAS judge components...")
    llm = _get_llm()
    if llm is None:
        raise SystemExit("No LLM configured (GROQ_API_KEY missing) — cannot run evaluation.")
    vectorstore = _get_vectorstore()
    # Deliberately a *separate* ChatGroq instance from `llm` above (not
    # reused) -- see the comment in _get_ragas_llm for why sharing one
    # instance between the judge and the generator corrupts both.
    judge_llm = _get_llm()
    ragas_llm = _get_ragas_llm(judge_llm)
    ragas_emb = _get_ragas_embeddings()

    topk_df, retriever_df = None, None

    if "1" in selected:
        experiment_1_llm_only_vs_rag(llm, vectorstore, ragas_llm, ragas_emb, [q for q in full_qs if q.get("category") == "in_scope"])
    if "2" in selected:
        topk_df = experiment_2_topk_comparison(llm, vectorstore, ragas_llm, ragas_emb, core_qs, k_values=k_values)
    if "3" in selected:
        experiment_3_prompt_comparison(llm, vectorstore, ragas_llm, ragas_emb, core_qs)
    if "4" in selected:
        experiment_4_hallucination_analysis(llm, vectorstore, ragas_llm, ragas_emb, full_qs)
    if "5" in selected:
        retriever_df = experiment_5_retriever_comparison(llm, vectorstore, ragas_llm, ragas_emb, core_qs)
    if "6" in selected:
        experiment_6_memory_evaluation(llm, vectorstore, ragas_llm, ragas_emb, mem_pairs)
    if "7a" in selected:
        experiment_7a_metadata_tagging_ablation(llm, vectorstore, ragas_llm, ragas_emb, core_qs)
    if "7b" in selected:
        experiment_7b_scope_guard_accuracy(llm, scope_set)
    if "7c" in selected:
        experiment_7c_latency_comparison(topk_df, retriever_df)

    build_overall_summary()
    print(f"\nAll results written to {OUTPUT_DIR}/")


if __name__ == "__main__" :
    main()
