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
import sys, os

# Must be set before torch/faiss are imported anywhere below. torch and
# faiss-cpu each bundle their own OpenMP runtime; loading both in one
# process on macOS causes a hard segfault (no Python traceback) the first
# time they both do parallel work (e.g. embedding a query + FAISS search).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Any
from logger import get_logger
from functools import lru_cache
from config import settings
from rag.prompts import RAG_PROMPT, CONDENSE_QUESTION_PROMPT, is_complaint_question
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

log = get_logger("rag.pipeline")

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
        )
    return None


#Load FAISS Vector DB
@lru_cache(maxsize=1)
def _get_vectorstore() -> Any:
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


    # Check whether the LLM API key is available before processing the request
    llm = _get_llm()
    if llm is None:
        return {
            "answer":          "No LLM API key configured. Add GROQ_API_KEY to .env (free at console.groq.com).",
            "sources":         [],
            "use_case":        "complaint_exploration",
            "retrieval_count": 0,
        }

    # Ensure the question is related to customer reviews or complaints
    if not is_complaint_question(question):
        log.info("Out-of-scope question rejected: %s", question[:60])
        return {
            "answer":          OUT_OF_SCOPE_MSG,
            "sources":         [],
            "use_case":        "complaint_exploration",
            "retrieval_count": 0,
        }

    try:
        # Load the FAISS vector database and create a retriever
        vectorstore = _get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "fetch_k": 20},
        )
        log.debug("Vectorstore + retriever ready")

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

        # Retrieve relevant reviews and generate the final response
        result = chain.invoke({"input": question, "chat_history": chat_history_text})

        # Format the retrieved reviews for display in the application
        sources = _format_sources(result.get("context", []))

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
        }

    # Handle the case where the FAISS index has not been created
    except FileNotFoundError as e:
        log.error("Index not found: %s", e)
        return {
            "answer":          str(e),
            "sources":         [],
            "use_case":        "complaint_exploration",
            "retrieval_count": 0,
        }

    # Handle any unexpected errors during the RAG pipeline
    except Exception as e:
        log.error("RAG pipeline error: %s", e, exc_info=True)
        return {
            "answer":          f"An error occurred: {str(e)}",
            "sources":         [],
            "use_case":        "complaint_exploration",
            "retrieval_count": 0,
        }