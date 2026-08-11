"""
Prompt templates used by the Retrieval-Augmented Generation (RAG) pipeline.

These prompts guide the language model to answer business analyst questions
using only the customer reviews retrieved from the FAISS vector database.

The prompts also support multi-turn conversations and prevent responses to
questions outside the customer complaint analysis use case.
"""
from typing import Optional, Any

# LangChain ≥1.0 moved PromptTemplate to langchain_core; fall back to old path for <0.2
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate  # type: ignore[no-redef]


# Prompt that instructs the LLM how to analyse the retrieved customer reviews
# and generate an evidence-based response.
#
# NOTE: each entry in {context} is tagged with its real metadata
# (see DOCUMENT_PROMPT in pipeline.py) — [Platform: ... | Rating: ... | Sentiment: ...]
# followed by the review text. That tagging is what makes rules 2 and 5 below
# actually enforceable instead of asking the model to guess.
RAG_PROMPT_TEMPLATE = """You are FeedbackIQ, an analyst assistant that helps business analysts understand customer complaint patterns from real customer reviews.

SCOPE
Answer using only the reviews under "Retrieved Reviews" below. You have no knowledge of Amazon, Yelp, or Twitter Airline beyond what is written in those reviews.

STRICT RULES:
1. Base every statement only on the Retrieved Reviews below — never on outside knowledge or assumption.
2. Mention a platform (Amazon, Yelp, or Twitter Airline) only if it actually appears in the "Platform:" tag of a retrieved review. Never reference a platform absent from the Retrieved Reviews, even if the question names it.
3. Never invent statistics, counts, percentages, complaint categories, or quotes that are not explicitly present in the Retrieved Reviews.
4. If the Retrieved Reviews are empty, off-topic, or insufficient to answer confidently, respond with exactly: "The retrieved reviews do not contain enough evidence to answer this question." Do not attempt a partial answer.
5. Every complaint pattern you report must be traceable to at least one specific retrieved review.
6. Be concise and analytical — no filler, no generic advice that isn't tied to the evidence above.

Retrieved Reviews:
{context}

Business Analyst Question: {input}

Respond in exactly this structure:

Platform(s) covered: [only platforms whose tag appears above; if one platform, name only that one]

Key complaint patterns:
- [pattern grounded in a specific review]
- [add more only if the evidence supports it — do not pad]

Recommendation: [one specific, measurable action tied directly to the patterns above]"""


# input_variables use "input" (not "question") to match what
# create_stuff_documents_chain / create_retrieval_chain pass through —
# LangChain's LCEL retrieval-chain helpers hardcode "input" as the query
# key, unlike the deprecated ConversationalRetrievalChain this replaced.
RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "input"],
)


# Prompt used to rewrite follow-up questions into standalone questions
# before retrieving relevant customer reviews.
CONDENSE_QUESTION_TEMPLATE = """Given the conversation history and a follow-up question, rewrite the follow-up as a single standalone question about customer complaints, so it can be used on its own to search a review database.

Rules:
- Do not add facts, platforms, numbers, or assumptions not implied by the conversation.
- If the follow-up question is already standalone, return it unchanged.
- Output only the rewritten question — no preamble, labels, or explanation.

Conversation History:
{chat_history}

Follow-up Question: {input}

Standalone question:"""

# input_variables use "input" (not "question") — required by
# create_history_aware_retriever, which raises ValueError if "input" isn't
# present in the prompt's variables. See RAG_PROMPT's comment above for why.
CONDENSE_QUESTION_PROMPT = PromptTemplate(
    template=CONDENSE_QUESTION_TEMPLATE,
    input_variables=["chat_history", "input"],
)


# ── Out-of-scope guard ────────────────────────────────────────────────────────
# Two layers are provided:
#   1. is_complaint_question()      - free, instant, regex keyword filter.
#      Use as the default first-pass gate before touching the retriever/LLM.
#   2. is_complaint_question_llm()  - one small LLM call, more precise on
#      ambiguous questions (e.g. "what do people think about delivery speed").
#      Swap it in inside rag/pipeline.py's ask() if the keyword filter proves
#      too imprecise during evaluation (a defensible thing to report in the
#      dissertation's error analysis).
import re

OUT_OF_SCOPE_TOPICS = [
    "weather", "politics", "sports", "recipe", "coding",
    "write code", "poem", "story", "joke",
]

# Word-boundary regex — plain substring matching (the previous approach)
# false-positives on things like "history" (contains "story") or "sporty",
# which would wrongly reject legitimate complaint questions.
_OUT_OF_SCOPE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in OUT_OF_SCOPE_TOPICS) + r")\b",
    re.IGNORECASE,
)


def is_complaint_question(question: str) -> bool:
    """
    Fast heuristic check for whether a question is in-scope.

    Returns:
        True  - question does not match a known off-topic keyword.
        False - question clearly matches an off-topic keyword (whole-word match).

    This is a blunt, zero-cost first-pass filter, not a semantic classifier —
    it will not catch every off-topic question (e.g. "recommend me a good book"),
    only the ones the current topic list explicitly names. See
    is_complaint_question_llm() for a more precise alternative.
    """
    return not bool(_OUT_OF_SCOPE_PATTERN.search(question))


# ── Optional: LLM-based scope classifier ──────────────────────────────────────
SCOPE_CLASSIFIER_TEMPLATE = """Decide if the question below is asking about customer complaints, feedback, reviews, sentiment, or ratings on Amazon, Yelp, or Twitter Airline.

Question: {question}

Answer with exactly one word: YES or NO."""

SCOPE_CLASSIFIER_PROMPT = PromptTemplate(
    template=SCOPE_CLASSIFIER_TEMPLATE,
    input_variables=["question"],
)


def is_complaint_question_llm(question: str, llm: Optional[Any] = None) -> bool:
    """
    LLM-based out-of-scope guard. More accurate than the keyword heuristic on
    ambiguous questions, at the cost of one extra small LLM call.

    Falls back to is_complaint_question() if no LLM is configured or the call
    fails, so this can never crash the pipeline or leave a question unclassified.

    Not wired into ask() by default — swap it in there if you want the
    stricter guard:
        from rag.prompts import is_complaint_question_llm as is_complaint_question
    """
    from rag.pipeline import _get_llm  # local import: avoids a circular import
                                        # with pipeline.py, which imports from
                                        # this module at load time.

    llm = llm or _get_llm()
    if llm is None:
        return is_complaint_question(question)

    try:
        prompt_text = SCOPE_CLASSIFIER_PROMPT.format(question=question)
        response = llm.invoke(prompt_text)
        text = getattr(response, "content", str(response)).strip().upper()
        return text.startswith("YES")
    except Exception:
        return is_complaint_question(question)