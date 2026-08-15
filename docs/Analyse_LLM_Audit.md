# Analyse Page LLM Audit — `nlp/summariser.py`

Scope: the Analyse page's business-analysis LLM call — `analyse_review_with_llm()`
in `nlp/summariser.py`, and its retrieval step, `semantic_search()` in
`nlp/embedding_service.py`, as invoked from `backend/services/sentiment_service.py::analyse_review()`.

This is a narrower, lower-stakes pipeline than the Chatbot's RAG system: input is
always a single pasted review (not an open question over the whole corpus), so
there is no "off-topic question" surface the way there is on the Chatbot page.
The risks here are about **fabrication inside the generated analysis** and
**unfiltered low-relevance retrieval**, not out-of-scope refusal.

Not part of this audit: `nlp/categoriser.py` (complaint category) is a zero-shot
HuggingFace classification pipeline, not a free-text LLM call — no hallucination
surface there.

---

## Headline finding

There is no hard gate anywhere in this pipeline. Retrieval always returns its
top-k reviews regardless of relevance (no `SIMILARITY_THRESHOLD`, unlike the
now-fixed Chatbot retriever), and the LLM's output is trusted verbatim — nothing
checks that `summary` / `business_insight` / `executive_summary` actually stay
within what the pasted review and retrieved reviews say. Grounding here rests
entirely on prompt instructions ("Never invent information," "Use retrieved
reviews ONLY as supporting evidence"), which is exactly the soft-gate pattern
the Chatbot audit flagged as insufficient on its own.

---

## Issue 1 — No relevance threshold on retrieval for this pipeline

**Severity:** Medium
**File:** `nlp/embedding_service.py`
**Function:** `semantic_search()`

**Problem:** Called from `sentiment_service.py` with no relevance floor. It
always returns its top-5 nearest reviews by cosine/L2 similarity, however weak
the match, exactly like the Chatbot's retriever did before today's fix.

**Why it matters:** If a user pastes text that has no good match in the corpus
(e.g. a review about a product category not covered, or non-review text
entirely), `semantic_search` still hands back 5 "similar reviews" — just weakly
similar ones — and the prompt tells the model to use them as supporting
evidence for `business_insight` and `executive_summary`. The model has no signal
that these matches are actually weak; a `similarity_score` is present in the
data passed to the prompt, but the prompt never instructs the model to
discount low scores or omit weak evidence.

**Exact fix:** Filter `similar_reviews` before they reach the prompt, dropping
any below a relevance floor (reuse `rag.pipeline.SIMILARITY_THRESHOLD` or a
locally-defined constant, since `similarity_score` here is computed the same
way — FAISS `IndexFlatIP`/L2 via `embedding_service.py`).

**Replacement code** (`nlp/summariser.py`, top of `analyse_review_with_llm`):
```python
ANALYSE_SIMILARITY_THRESHOLD = 0.35  # same basis as rag/pipeline.py; tune together

def analyse_review_with_llm(
    text: str,
    sentiment: str,
    category: str,
    similar_reviews: list[dict],
) -> dict:

    grounded_reviews = [
        r for r in similar_reviews
        if r.get("similarity_score", 0) >= ANALYSE_SIMILARITY_THRESHOLD
    ]

    rag_context = ""
    for i, review in enumerate(grounded_reviews, start=1):
        ...
```

**Expected behaviour:** Weak matches are silently excluded from the prompt
rather than presented as supporting evidence; `rag_context` becomes empty (not
just short) when nothing in the corpus is genuinely close, which also matters
for Issue 3 below.

---

## Issue 2 — LLM output trusted with no post-hoc grounding check

**Severity:** Medium
**File:** `nlp/summariser.py`
**Function:** `analyse_review_with_llm()`

**Problem:** The prompt instructs "Never invent information," "Describe ONLY
the current customer review," but nothing in code verifies this after the
model responds. `result` from `_get_chain().invoke(...)` is returned as-is on
success.

**Why it matters:** This is the same soft-gate weakness the Chatbot audit
flagged as Issue 1 — a prompt instruction is not a guarantee. Unlike the
Chatbot, there's no clean equivalent of "empty context → hard refusal" here,
because the primary input (the pasted review) is never empty by the time this
function runs. A full hard gate isn't really applicable to this pipeline the
way it was for the Chatbot. The practical mitigation is narrower: make sure the
model can't cite retrieved evidence that doesn't exist (Issue 1) and keep the
generated claims checkable against source text (already partly true, since the
UI shows the original review alongside the analysis, letting a human catch
fabrication) rather than trying to build an automatic verifier, which would be
disproportionate for a 2-sentence summary and a 25-word insight.

**Exact fix:** No code fix recommended beyond Issue 1 and Issue 4 below —
flagging this as a known, accepted limitation rather than something to
over-engineer a solution for, since the output is short, human-reviewed in the
UI, and not used for autonomous decisions.

**Expected behaviour:** N/A — documented limitation, not a code change.

---

## Issue 3 — Prompt doesn't handle the empty-retrieval case

**Severity:** Low-Medium
**File:** `nlp/summariser.py`
**Function:** prompt template + `analyse_review_with_llm()`

**Problem:** Instruction 3 ("Business Insight... Use BOTH the review and
retrieved reviews") is unconditional. If `similar_reviews` is `[]` (already
possible today if semantic search fails or the corpus has nothing close; more
likely once Issue 1's threshold is added), the model is still told to use
"both," with nothing in `rag_context` to use.

**Why it matters:** An LLM told to use evidence that isn't there will often
either (a) quietly fabricate what the retrieved reviews might say, or (b)
produce a stilted business_insight that vaguely gestures at "similar cases"
that don't actually exist in the context. Neither is desirable, and it's the
same class of problem `RAG_PROMPT_TEMPLATE`'s rule 5 was written to prevent on
the Chatbot side.

**Exact fix:** Make the instruction conditional, mirroring the Chatbot prompt's
handling of empty context.

**Replacement code** (`nlp/summariser.py`, template + call site):
```python
# In analyse_review_with_llm(), before building rag_context:
has_similar_reviews = bool(grounded_reviews)  # from Issue 1's filtered list
```
```text
# In the prompt template, replace instruction 3 with:
3. Business Insight
- Maximum 25 words.
- Start with an action verb.
- If "Retrieved Similar Reviews" is non-empty, ground the insight in both the
  current review and that evidence.
- If "Retrieved Similar Reviews" is empty, base the insight only on the current
  review — do not reference or imply the existence of similar cases.
```
And pass an explicit marker when there's nothing retrieved, e.g. set
`rag_context = "(No sufficiently similar historical reviews found.)"` when
`grounded_reviews` is empty, so the model sees an explicit absence rather than
an empty string it might paper over.

**Expected behaviour:** With no relevant retrieved reviews, the model produces
an insight grounded only in the pasted review, and doesn't gesture at
comparison data that isn't actually present.

---

## Issue 4 — No schema validation on the parsed JSON output

**Severity:** Low
**File:** `nlp/summariser.py` / `backend/services/sentiment_service.py`
**Function:** `analyse_review_with_llm()` → `analyse_review()`

**Problem:** `JsonOutputParser()` (no Pydantic schema attached) parses
whatever JSON shape the model returns. If the model omits a key, uses the
wrong type (e.g. `keywords` as a string instead of a list), or returns a
`severity`/`priority`/`department` value outside the closed set the prompt
defines, nothing catches it here. The `except Exception` in
`analyse_review_with_llm` only catches failures in the LLM call itself
(timeout, API error, invalid JSON that fails parsing) — a syntactically valid
but wrong-shaped JSON object passes through untouched and is merged into the
response dict at `sentiment_service.py:114` (`**analysis`). It's Pydantic's
`ReviewAnalysis` model (`backend/models/schemas.py`) that would eventually
raise on a bad shape — but that surfaces as a generic 500 from FastAPI's
response validation, not a clean fallback to `EMPTY_ANALYSIS`.

**Why it matters:** Correctness/robustness, not hallucination per se — but a
malformed field ships to the user as a confusing 500 error instead of a
graceful partial result, and a wrong-enum `severity`/`department` value
would look plausible in the UI even though it violates the app's own defined
categories.

**Exact fix:** Use `JsonOutputParser(pydantic_object=ReviewAnalysisLLM)` with a
local Pydantic schema (mirroring `backend.models.schemas.ReviewAnalysis`,
including `Literal` constraints on `severity`/`priority`/`department`), and
catch `ValidationError` alongside the existing `Exception` handler to fall
back to `EMPTY_ANALYSIS` instead of letting a bad shape propagate to the API
response layer.

**Replacement code sketch:**
```python
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

class ReviewAnalysisLLM(BaseModel):
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

parser = JsonOutputParser(pydantic_object=ReviewAnalysisLLM)
# ... in analyse_review_with_llm():
except (Exception, ValidationError):
    log.exception("LLM request failed or returned an invalid shape.")
    return EMPTY_ANALYSIS.copy()
```

**Expected behaviour:** A malformed or out-of-enum LLM response falls back
cleanly to `EMPTY_ANALYSIS` (same graceful-degradation path already used for
LLM call failures) instead of surfacing as an unhandled 500.

---

## Issue 5 — Duplicated fields in the retrieved-review context block

**Severity:** Low (cosmetic / token-waste, not a hallucination risk)
**File:** `nlp/summariser.py`
**Function:** `analyse_review_with_llm()`, lines 190-201

**Problem:** The per-review block built for `rag_context` repeats `Platform:`
and `Similarity Score:` twice each:
```python
rag_context += (
    f"\nSimilar Review {i}\n"
    f"Platform: {review['platform']}\n"
    f"Similarity Score: {review['similarity_score']:.3f}\n"
    f"Platform: {review['platform']}\n"          # duplicate
    f"Rating: {review.get('rating','N/A')}\n"
    f"Sentiment: {review.get('sentiment_label','N/A')}\n"
    f"Similarity Score: {review['similarity_score']:.3f}\n"  # duplicate
    f"Review: {review['text'][:250]}\n"
)
```

**Why it matters:** Not a correctness bug — just wasted prompt tokens and
mildly confusing context. Also uses `review['platform']` /
`review['similarity_score']` (bracket access, not `.get()`) while other fields
use `.get()` with a fallback — inconsistent, and would raise `KeyError` if
either key were ever missing from a `similar_reviews` dict (currently they
never are, per `embedding_service.py`, but it's the only two fields not
defended the way `rating`/`sentiment_label` are).

**Exact fix:**
```python
rag_context += (
    f"\nSimilar Review {i}\n"
    f"Platform: {review.get('platform', 'unknown')}\n"
    f"Rating: {review.get('rating', 'N/A')}\n"
    f"Sentiment: {review.get('sentiment_label', 'N/A')}\n"
    f"Similarity Score: {review.get('similarity_score', 0):.3f}\n"
    f"Review: {review.get('text', '')[:250]}\n"
)
```

**Expected behaviour:** Same information, no duplication, and consistent
`.get()` defensive access across all fields.

---

## Priority order if you implement these

1. **Issue 1** (relevance threshold) — the closest analogue to the Chatbot's
   core fix; meaningfully reduces the chance of a business-insight built on
   irrelevant "similar" reviews.
2. **Issue 3** (empty-retrieval prompt handling) — directly depends on Issue 1
   existing (there's no empty-context case to handle until weak matches are
   filtered out).
3. **Issue 4** (schema validation) — robustness/UX fix, unrelated to the other
   two, can be done independently.
4. **Issue 5** (duplicate fields) — trivial cleanup, do anytime.
5. **Issue 2** — no action recommended; documented as an accepted limitation.

Issues 1, 3, 4, 5 are additive/isolated — each can be implemented and verified
independently without touching the Chatbot's `rag/pipeline.py`.
