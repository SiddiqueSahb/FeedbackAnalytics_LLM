"""
RAG Pipeline — Conversational Complaint Exploration
=====================================================
USE CASE: Business analysts ask natural language questions about
customer complaint patterns. The system retrieves relevant reviews
from the FAISS index and generates grounded answers using an LLM.

This is the SPECIFIC use case defined for the dissertation:
  "Conversational exploration of customer complaints across platforms"

  Evaluated via RAGAS in scripts/evaluate_rag.py
"""

from __future__ import annotations
import sys, os, re, time

# Must be set before torch/faiss are imported anywhere below. torch and
# faiss-cpu each bundle their own OpenMP runtime; loading both in one
# process on macOS causes a hard segfault (no Python traceback) the first
# time they both do parallel work (e.g. embedding a query + FAISS search).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
from typing import Optional, Any
from logger import get_logger
from functools import lru_cache
from config import settings
from rag.prompts import (
    RAG_PROMPT,
    CONDENSE_QUESTION_PROMPT,
    is_complaint_question,
    is_complaint_question_llm,
)
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

log = get_logger("rag.pipeline")

_vectorstore_lock = threading.Lock()

# Minimum relevance score (0-1, 1 = most similar) a retrieved review must
# clear to be handed to the LLM at all. Without this, FAISS always returns
# its top-k documents regardless of how irrelevant they are — for a
# question like "Who is the CEO of Apple?", the retriever would still hand
# back 5 customer reviews, and the only thing preventing a hallucinated
# answer would be whether the LLM correctly judges those reviews as
# insufficient and follows RAG_PROMPT's refusal instruction. This constant
# makes "irrelevant" a fact the code checks before generation, not a
# judgement the model makes every time.
#
# 0.35 is a starting point, not a validated constant — this index uses
# Euclidean distance on normalised all-MiniLM-L6-v2 embeddings, converted
# to a [0, 1] score via 1 - distance/sqrt(2) (see langchain_core's
# VectorStore._euclidean_relevance_score_fn). Tune this empirically using
# the evaluation plan's Group D questions (docs/RAG_Hallucination_Audit.md)
# before treating it as final.
SIMILARITY_THRESHOLD = 0.35

# Safety cap on how much of a single review's text can enter the prompt.
# Reviews in this corpus are short (median 19-85 words depending on
# platform), so this is not expected to bind under current data — it's a
# guard against context overflow if the index is ever extended with
# longer-form content.
MAX_REVIEW_CHARS = 1000

# Bounded retry for transient Groq rate-limit errors on the live request
# path. Deliberately much smaller than evaluate/evaluate_llm_vs_rag.py's
# retry budget (6 retries, 8s apart) -- that script runs unattended, this
# runs while a person is waiting for a chat response, so it trades
# completeness for not making them wait a long time for an error message.
# It only retries rate-limit errors; anything else propagates immediately,
# so it changes nothing about how genuine failures (bad key, network,
# malformed request) are surfaced.
MAX_LIVE_RETRIES = 2
RETRY_BASE_WAIT = 2.0


def _extract_retry_after_seconds(exc: Exception, default: float = RETRY_BASE_WAIT) -> float:
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 0.5
        except ValueError:
            pass
    return default


def _invoke_with_retry(fn, *args, max_retries: int = MAX_LIVE_RETRIES, **kwargs):
    """Call fn(*args, **kwargs), retrying on rate-limit errors only.

    Used to wrap the two LLM call sites on the live request path (the main
    generation chain in ask(), and the scope-classifier call in
    is_complaint_question_llm()) so a brief, transient Groq rate limit gets
    absorbed instead of surfacing immediately as "An error occurred: ...".
    Not a general-purpose retry-everything wrapper -- a non-rate-limit
    exception raises on the first attempt, same as before this existed.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            is_rate_limit = "rate_limit" in str(exc).lower() or "429" in str(exc)
            if not is_rate_limit or attempt >= max_retries:
                raise
            wait = _extract_retry_after_seconds(exc)
            log.warning(
                "Rate limit hit, retrying in %.1fs (%d/%d)",
                wait, attempt + 1, max_retries,
            )
            time.sleep(wait)


# create_stuff_documents_chain's default document formatter only inserts
# doc.page_content into {context} — none of the metadata (platform, rating,
# sentiment) that build_index.py stores on every review ever reaches the LLM.
# RAG_PROMPT's grounding rules (only cite platforms present in context) are
# unenforceable without this: tag each retrieved review with its real
# metadata before it is stuffed into {context}. All four keys are guaranteed
# on every document (see scripts/build_index.py meta_cols).
DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content", "platform", "rating", "sentiment_label"],
    template="[Platform: {platform} | Rating: {rating} | Sentiment: {sentiment_label}]\n{page_content}",
)

# ── Out-of-scope response 
OUT_OF_SCOPE_MSG: str = (
    "I'm designed specifically for exploring customer complaint patterns across "
    "Amazon, Yelp, and Twitter Airline reviews. Please ask me about complaints, "
    "sentiment trends, or customer feedback. For example: "
    "'What are the top complaints on Amazon?' or "
    "'Why are airline customers most dissatisfied?'"
)

#Load LLM
def _get_llm() -> Optional[Any]:
    """Return LangChain LLM wrapper (Groq or OpenAI). Returns None if no key configured."""
    if settings.USE_GROQ and settings.GROQ_API_KEY:
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            # Without this, a slow/stuck Groq call can run well past the
            # frontend's own 60s request timeout — the backend keeps working
            # on a request the user has already been told failed. 30s keeps
            # the backend's own failure surfacing before that happens.
            timeout=30,
        )
    return None


#Load FAISS Vector DB
@lru_cache(maxsize=1)
def _get_vectorstore_uncached() -> Any:
    """Load LangChain FAISS vectorstore (cached after first load)."""
    from langchain_community.vectorstores import FAISS as LangFAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    if not os.path.exists(settings.LANGCHAIN_INDEX_PATH):
        raise FileNotFoundError(
            f"LangChain index not found at {settings.LANGCHAIN_INDEX_PATH}. "
            "Run: python scripts/build_index.py"
        )
    # Loads the embedding model into memory.
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # It does NOT calculate embeddings.,It simply reads the vectors that already exist.
    vs = LangFAISS.load_local(
        settings.LANGCHAIN_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    log.info("LangChain FAISS vectorstore loaded")
    return vs


def _get_vectorstore() -> Any:
    # Lock closes the gap lru_cache leaves open on its own — see the same
    # pattern/comment in nlp/embedding_service.py._load_resources.
    with _vectorstore_lock:
        return _get_vectorstore_uncached()


def _make_grounded_retriever(
    vectorstore: Any,
    platform_filter: Optional[dict[str, str]],
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    threshold: float = SIMILARITY_THRESHOLD,
) -> RunnableLambda:
    """
    Build a retriever that only ever returns documents clearing
    `threshold` on relevance — the hard gate that a plain
    `as_retriever(search_type="mmr", ...)` does not provide, since MMR only
    controls *diversity* among whatever top-k FAISS finds, not whether any
    of it is actually relevant.

    Two passes over the same candidate pool:
      1. `similarity_search_with_relevance_scores` scores every candidate
         and keeps only those at or above `threshold`. If nothing clears
         it, an empty list is returned — the caller (ask()) is expected to
         treat that as "refuse", not "answer from nothing".
      2. `max_marginal_relevance_search` reruns MMR over the same
         (query, filter) to pick a diverse k from the pool, then that
         selection is intersected with step 1's passing set — so MMR's
         diversity re-ranking can never smuggle in a sub-threshold review
         just because it was topically distinct from the others.

    Each returned Document is a fresh copy (never the FAISS docstore's own
    object) with its relevance score attached at metadata["similarity_score"]
    and its text capped at MAX_REVIEW_CHARS — see the module-level comments
    on both.
    """

    def _retrieve(query: str) -> list[Document]:
        scored = vectorstore.similarity_search_with_relevance_scores(
            query, k=fetch_k, filter=platform_filter,
        )
        passing_scores = {
            doc.metadata.get("review_id"): score
            for doc, score in scored
            if score >= threshold
        }
        if not passing_scores:
            return []

        mmr_docs = vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=platform_filter,
        )

        grounded_docs: list[Document] = []
        for doc in mmr_docs:
            score = passing_scores.get(doc.metadata.get("review_id"))
            if score is None:
                continue

            content = doc.page_content
            if len(content) > MAX_REVIEW_CHARS:
                content = content[:MAX_REVIEW_CHARS] + "…"

            meta = dict(doc.metadata)
            meta["similarity_score"] = round(float(score), 4)
            grounded_docs.append(Document(page_content=content, metadata=meta))

        return grounded_docs

    return RunnableLambda(_retrieve)

#It converts the raw LangChain Document objects returned by FAISS into a clean, structured format that your UI (Streamlit/FastAPI) 
#can easily display.
def _format_sources(source_docs: list) -> list[dict[str, object]]:
    """
    Convert LangChain source documents into structured source dicts.
    Returns platform, sentiment, text snippet — for UI citation display.

    "text" stays truncated to 200 chars — that's a UI preview, not meant to
    be a complete record. "full_text" carries the untruncated review, for
    anything (e.g. RAGAS evaluation) that needs the same content the LLM
    actually saw rather than a display snippet.
    """
    sources: list[dict[str, object]] = []
    for doc in source_docs:
        meta: dict = doc.metadata or {}
        sources.append({
            "text":             doc.page_content[:200],
            "full_text":        doc.page_content,
            "platform":         meta.get("platform",        "unknown"),
            "sentiment_label":  meta.get("sentiment_label", "unknown"),
            "rating":           meta.get("rating",          0),
            "review_id":        meta.get("review_id",       ""),
            "similarity_score": meta.get("similarity_score"),
        })
    return sources

# Convert API chat history into the format expected by LangChain memory
def _build_history_string(chat_history: list[dict]) -> list[tuple[str, str]]:
    """
    Convert chat history from API format:
    [{"role": ..., "content": ...}]
    to LangChain conversation pairs:
    [(user_message, assistant_reply), ...]
    """

    # Store the converted user-assistant conversation pairs
    pairs: list[tuple[str, str]] = []

    # Keep only valid dictionary messages and handle empty chat history
    history: list[dict] = [m for m in (chat_history or []) if isinstance(m, dict)]

    # Start from the first message
    i: int = 0

    # Process the conversation two messages at a time
    while i < len(history) - 1:

        # Check if the current message is from the user
        # and the next message is the assistant's reply
        if history[i].get("role") == "user" and history[i + 1].get("role") == "assistant":

            # Store the conversation as a (user, assistant) tuple
            pairs.append((
                history[i].get("content", ""),
                history[i + 1].get("content", ""),
            ))

            # Move to the next user-assistant pair
            i += 2

        else:
            # Skip messages that do not form a valid pair
            i += 1

    # Return the formatted conversation history
    return pairs



# Recognised platform names, checked against the raw question text.
_PLATFORM_KEYWORDS: dict[str, str] = {
    "amazon": "amazon",
    "yelp": "yelp",
    "twitter": "twitter_airline",
    "airline": "twitter_airline",
}


def _detect_platform_filter(question: str) -> Optional[dict[str, str]]:
    """
    If the question names a specific platform, return a FAISS metadata
    filter restricting retrieval to that platform's reviews.

    Without this, "What are the top complaints on Amazon?" and "What are
    the top complaints on Yelp?" hit the exact same unfiltered similarity
    search — the word "Amazon" is just more text for the embedding to
    match against, not a constraint, so the retriever is free to return
    Yelp or Twitter Airline reviews for an Amazon-specific question. This
    checks the question for a platform name and, if found, narrows
    retrieval to reviews tagged with that platform in build_index.py's
    stored metadata.

    Only looks at the current question, not chat_history — a follow-up
    like "give me an example of one of those" won't carry the filter over
    from an earlier turn. Good enough to fix the reproduced bug (asking
    about one named platform returning another platform's reviews)
    without adding history-scanning complexity.
    """
    text = question.lower()
    for keyword, platform in _PLATFORM_KEYWORDS.items():
        if keyword in text:
            return {"platform": platform}
    return None


def _format_chat_history_text(pairs: list[tuple[str, str]]) -> str:
    """
    Render (human, ai) pairs as a plain text block for CONDENSE_QUESTION_PROMPT.

    Returns "" when there's no history. That matters beyond formatting:
    create_history_aware_retriever treats an empty/falsy chat_history as "no
    history" and skips question reformulation entirely, querying the
    retriever with the raw question directly — so a first-turn question
    isn't needlessly rewritten by an LLM pass it doesn't need.
    """
    if not pairs:
        return ""
    lines: list[str] = []
    for human, ai in pairs:
        lines.append(f"Human: {human}")
        lines.append(f"AI: {ai}")
    return "\n".join(lines)


# Main function that handles the complete RAG workflow
# Main entry point of the RAG pipeline.
# This function receives the user's question and conversation history,
# retrieves relevant customer reviews from the FAISS vector database,
# sends the retrieved context to the LLM, and returns a grounded response.
def ask(question: str,chat_history: Optional[list[dict]] = None,) -> dict[str, object]:
    """
    Answer customer feedback questions using Retrieval-Augmented Generation (RAG).

    Args:
        question: User's question about customer reviews or complaints.
        chat_history: Previous conversation in API format
                      [{"role": "...", "content": "..."}].

    Returns:
        A dictionary containing:
            answer           - Response generated by the LLM.
            sources          - Retrieved reviews used to generate the answer.
            use_case         - Current application ("complaint exploration").
            retrieval_count  - Number of retrieved reviews.
    """

    # Record the incoming question and conversation length
    # to help with debugging and monitoring the application.
    log.info("RAG ask | question='%s' history_len=%d",question[:60], len(chat_history or []))


    # Fixed refusal string returned whenever retrieval finds nothing above
    # SIMILARITY_THRESHOLD. Deliberately distinct from OUT_OF_SCOPE_MSG:
    # this fires when a question *is* answerable in principle but the
    # index has no relevant evidence for it, not when the question is
    # about a different subject entirely.
    NO_EVIDENCE_MSG = "I couldn't find enough relevant customer reviews to answer this question."

    # Check whether the LLM API key is available before processing the request
    llm = _get_llm()
    if llm is None:
        return {
            "answer":          "No LLM API key configured. Add GROQ_API_KEY to .env (free at console.groq.com).",
            "sources":         [],
            "use_case":        "complaint_exploration",
            "retrieval_count": 0,
            "grounded":        False,
        }

    # Ensure the question is related to customer reviews or complaints.
    # Cheap keyword check first — catches named off-topic subjects (weather,
    # sports, coding, ...) with no LLM call.
    if not is_complaint_question(question):
        log.info("Out-of-scope question rejected by keyword guard: %s", question[:60])
        return {
            "answer":          OUT_OF_SCOPE_MSG,
            "sources":         [],
            "use_case":        "complaint_exploration",
            "retrieval_count": 0,
            "grounded":        False,
        }

    # Second-stage guard: the keyword list above only names nine specific
    # off-topic subjects, so "Who is the CEO of Apple?" or "What is quantum
    # computing?" pass through it untouched. This LLM classifier catches
    # what the keyword list misses, at the cost of one extra small LLM call
    # on every question that clears the first gate.
    if not is_complaint_question_llm(question, llm=llm):
        log.info("Out-of-scope question rejected by LLM guard: %s", question[:60])
        return {
            "answer":          OUT_OF_SCOPE_MSG,
            "sources":         [],
            "use_case":        "complaint_exploration",
            "retrieval_count": 0,
            "grounded":        False,
        }

    try:
        # Load the FAISS vector database
        vectorstore = _get_vectorstore()

        platform_filter = _detect_platform_filter(question)
        if platform_filter:
            log.info("Platform filter applied: %s", platform_filter)

        fetch_k = 50 if platform_filter else 20
        retriever = _make_grounded_retriever(
            vectorstore, platform_filter, k=5, fetch_k=fetch_k,
        )
        log.debug("Grounded retriever ready (threshold=%.2f)", SIMILARITY_THRESHOLD)

        # Fast path for single-turn questions: the retrieval query is the
        # raw question with no reformulation involved, so it can be probed
        # directly, before building or invoking the chain at all. A
        # clearly out-of-domain question that both guards above missed is
        # refused here without spending an LLM call on generation.
        if not chat_history:
            probe_docs = retriever.invoke(question)
            if not probe_docs:
                log.info("No documents cleared the similarity threshold (fast path): %s", question[:60])
                return {
                    "answer":          NO_EVIDENCE_MSG,
                    "sources":         [],
                    "use_case":        "complaint_exploration",
                    "retrieval_count": 0,
                    "grounded":        False,
                }

        # LCEL replacement for the deprecated ConversationalRetrievalChain +
        # ConversationBufferMemory pair. No memory object: chat_history is
        # just passed straight through on each call as plain text.
        #   1. history_aware_retriever reformulates `question` into a
        #      standalone query when chat_history is non-empty (via
        #      CONDENSE_QUESTION_PROMPT), then retrieves; with no history it
        #      queries the retriever with the raw question directly.
        #   2. combine_docs_chain answers from the retrieved + tagged
        #      context using RAG_PROMPT.
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, CONDENSE_QUESTION_PROMPT
        )
        combine_docs_chain = create_stuff_documents_chain(
            llm, RAG_PROMPT, document_prompt=DOCUMENT_PROMPT
        )
        chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
        log.debug("LCEL retrieval chain built")

        chat_history_text = _format_chat_history_text(_build_history_string(chat_history or []))

        # Retrieve relevant reviews and generate the final response.
        # _invoke_with_retry absorbs a brief Groq rate limit here instead of
        # letting it fall straight through to the except Exception below and
        # surfacing as "An error occurred" on the user's first request.
        result = _invoke_with_retry(
            chain.invoke, {"input": question, "chat_history": chat_history_text}
        )

        retrieved_docs = result.get("context", [])

        if not retrieved_docs:
            # Hard backstop, not a repeat of the fast-path check above: this
            # covers multi-turn questions, where the retrieval query is an
            # LLM-reformulated standalone question the fast path above
            # never sees. If nothing cleared the relevance threshold for
            # that reformulated query, the answer the model generated is
            # discarded outright — the model was still asked to answer from
            # empty context, and RAG_PROMPT's refusal rule is an
            # instruction to it, not a guarantee. This is the guarantee.
            log.info(
                "No grounded context after chain execution, overriding generated answer: %s",
                question[:60],
            )
            return {
                "answer":          NO_EVIDENCE_MSG,
                "sources":         [],
                "use_case":        "complaint_exploration",
                "retrieval_count": 0,
                "grounded":        False,
            }

        # Format the retrieved reviews for display in the application
        sources = _format_sources(retrieved_docs)

        # Log the response for debugging and monitoring
        log.info(
            "RAG answer | sources=%d answer_len=%d",
            len(sources),
            len(result.get("answer", "")),
        )

        # Return the generated answer along with its supporting reviews
        return {
            "answer":          result["answer"],
            "sources":         sources,
            "use_case":        "complaint_exploration",
            "retrieval_count": len(sources),
            "grounded":        True,
        }

    # Handle the case where the FAISS index has not been created
    except FileNotFoundError as e:
        log.error("Index not found: %s", e)
        return {
            "answer":          str(e),
            "sources":         [],
            "use_case":        "complaint_exploration",
            "retrieval_count": 0,
            "grounded":        False,
        }

    # Handle any unexpected errors during the RAG pipeline
    except Exception as e:
        log.error("RAG pipeline error: %s", e, exc_info=True)
        return {
            "answer":          f"An error occurred: {str(e)}",
            "sources":         [],
            "use_case":        "complaint_exploration",
            "retrieval_count": 0,
            "grounded":        False,
        }