from __future__ import annotations

import json
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from config import settings

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from logger import get_logger

log = get_logger("nlp.summariser")

EMPTY_ANALYSIS = {
    "summary": "",
    "keywords": [],
    "business_insight": "",
    "severity": "",
    "priority": "",
    "department": "",
    "executive_summary": "",
}

# Minimum relevance score a retrieved "similar review" must clear before it is
# allowed into this prompt as supporting evidence. Same basis as
# rag.pipeline.SIMILARITY_THRESHOLD — both scores come from the same FAISS
# index over normalised embeddings (nlp/embedding_service.py). Without this,
# semantic_search always returns its top-5 nearest neighbours regardless of
# how weak the match is, and the LLM has no signal that "Similar Review 3" is
# only 0.28 similar — it just gets told to use it as evidence. See
# docs/Analyse_LLM_Audit.md Issue 1.
ANALYSE_SIMILARITY_THRESHOLD = 0.35

NO_SIMILAR_REVIEWS_MARKER = "(No sufficiently similar historical reviews found.)"


class ReviewAnalysisLLM(BaseModel):
    """
    Schema the LLM's JSON response must conform to. Validated explicitly after
    parsing — JsonOutputParser on its own only guarantees valid JSON, not that
    the keys/types/enums match what the rest of the app expects (a missing
    key or an out-of-enum severity would otherwise pass through untouched).
    See docs/Analyse_LLM_Audit.md Issue 4.
    """

    summary: str
    keywords: list[str] = Field(min_length=5, max_length=5)
    business_insight: str
    severity: Literal["Low", "Medium", "High", "Critical"]
    priority: Literal["Low", "Medium", "High", "Urgent"]
    department: Literal[
        "Engineering", "Quality Assurance", "Customer Support",
        "Logistics", "Finance", "Marketing",
    ]
    executive_summary: str


# Prompt Template
prompt = ChatPromptTemplate.from_template(
"""
You are a Senior Customer Experience Analyst.

IMPORTANT

Return ONLY ONE valid JSON object.

Do NOT include:
- explanations
- markdown
- notes
- reasoning
- analysis
- code fences

Analyse the customer review using BOTH:

1. The current customer review.
2. Similar historical customer reviews retrieved from the review database, if any are listed below.

Current Review
--------------
{review}

Predicted Sentiment
-------------------
{sentiment}

Complaint Category
------------------
{category}

Retrieved Similar Reviews (RAG)
-------------------------------
{rag_context}


{{
    "summary": "",
    "keywords": [],
    "business_insight": "",
    "severity": "",
    "priority": "",
    "department": "",
    "executive_summary": ""
}}

Instructions

1. Summary
- Maximum 2 sentences.
- Describe ONLY the current customer review.
- Preserve important product or service details.
- Do not invent information.

2. Keywords
- Return exactly FIVE business-related keywords.
- Each keyword should contain 1-3 words.
- Avoid duplicates.

3. Business Insight
- Maximum 25 words.
- Start with an action verb.
- If "Retrieved Similar Reviews" lists actual reviews, ground the insight in both the current review and that evidence.
- If "Retrieved Similar Reviews" states that none were found, base the insight only on the current review — do not reference or imply the existence of similar cases.

4. Severity
Choose ONE:
Low
Medium
High
Critical

5. Priority
Choose ONE:
Low
Medium
High
Urgent

6. Department

Engineering
- Software defects
- Hardware design issues

Quality Assurance
- Product failures
- Manufacturing defects
- Reliability issues

Customer Support
- Communication
- Complaint handling

Logistics
- Shipping
- Delivery
- Packaging

Finance
- Billing
- Refunds

Marketing
- Advertising
- Product information

7. Executive Summary
Maximum 30 words.

IMPORTANT

Use retrieved reviews ONLY as supporting evidence, and only if any are actually listed above.

Never invent information.

Return ONLY JSON.
"""
)


# Loaded lazily on first use, not at import time — importing this module
# (which happens transitively the moment backend.main starts) used to raise
# RuntimeError immediately if GROQ_API_KEY was blank, crashing the entire
# API before it could bind to a port, not just the one endpoint that needs
# the LLM. Now a missing key only fails the call that actually needs it,
# and that failure is already caught below and turned into EMPTY_ANALYSIS.
@lru_cache(maxsize=1)
def _get_chain():
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        timeout=30,
    )

    # Bind the schema at the model-call level (Groq tool-calling) rather than
    # relying on the prompt's written-out JSON example plus a free-text
    # JsonOutputParser. On llama-3.1-8b-instant specifically, the prompt-only
    # approach was observed to silently drop a required key (executive_summary)
    # in ~2/3 of calls when the retrieved-reviews context was short/empty —
    # each one degrading to EMPTY_ANALYSIS via the post-hoc validation added
    # for docs/Analyse_LLM_Audit.md Issue 4. Structured output constrains
    # generation itself, not just validates it after the fact, so incomplete
    # responses should become rare rather than being caught-and-discarded.
    return prompt | llm.with_structured_output(ReviewAnalysisLLM)


# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------

def analyse_review_with_llm(
    text: str,
    sentiment: str,
    category: str,
    similar_reviews: list[dict],
) -> dict:

    # Drop weak matches before they ever reach the prompt — the LLM should
    # never be told a "similar review" is supporting evidence when it isn't
    # actually similar. See ANALYSE_SIMILARITY_THRESHOLD above.
    grounded_reviews = [
        r for r in similar_reviews
        if r.get("similarity_score", 0) >= ANALYSE_SIMILARITY_THRESHOLD
    ]

    if grounded_reviews:
        rag_context = ""
        for i, review in enumerate(grounded_reviews, start=1):
            rag_context += (
                f"\nSimilar Review {i}\n"
                f"Platform: {review.get('platform', 'unknown')}\n"
                f"Rating: {review.get('rating', 'N/A')}\n"
                f"Sentiment: {review.get('sentiment_label', 'N/A')}\n"
                f"Similarity Score: {review.get('similarity_score', 0):.3f}\n"
                f"Review: {review.get('text', '')[:250]}\n"
            )
    else:
        # Explicit absence marker, not just an empty string — the prompt's
        # instruction 3 branches on this text being present. An empty string
        # left the model free to quietly fabricate what "similar reviews"
        # might have said; this makes the absence unambiguous. See
        # docs/Analyse_LLM_Audit.md Issue 3.
        rag_context = NO_SIMILAR_REVIEWS_MARKER

    log.info("Generating LLM business analysis")
    try:
        # with_structured_output(ReviewAnalysisLLM) means `result` is already
        # a validated ReviewAnalysisLLM instance (or raises) — the model's
        # tool-call arguments are schema-constrained at generation time, not
        # just checked after the fact. See _get_chain()'s comment for why
        # this replaced the plain JsonOutputParser + post-hoc validate.
        result = _get_chain().invoke(
        {
            "review": text[:600],
            "sentiment": sentiment,
            "category": category,
            "rag_context": rag_context,
        }
        )
        log.info("LLM analysis completed")
        return result.model_dump()

    except ValidationError:
        log.exception("LLM returned a malformed or out-of-enum analysis shape.")
        return EMPTY_ANALYSIS.copy()

    except Exception:
        log.exception("LLM request failed.")
        return EMPTY_ANALYSIS.copy()


# Test

if __name__ == "__main__":

    demo = analyse_review_with_llm(
        text="The battery stopped charging after one week and customer support ignored my emails.",
        sentiment="negative",
        category="Product Performance Failures",
        similar_reviews=[],
    )

    print(json.dumps(demo, indent=4))
