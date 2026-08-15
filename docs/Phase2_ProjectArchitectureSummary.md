# Phase 2 — Project Architecture Summary

Verified against the actual codebase (not the dissertation's description of
it). This is what was **implemented** — Phase 3/6/7 will separately establish
what was **executed** and what has **saved evidence**.

## Pipeline overview

```
Raw datasets (Amazon / Yelp / Twitter Airline)
        |
scripts/preprocess.py          -> cleaned, unified review corpus (parquet)
        |
        +-- Sentiment classification (5 models, compared on a shared split)
        |     VADER (nlp/sentiment.py, rule-based)
        |     Naive Bayes (scripts/train_classical_models.py, TF-IDF features)
        |     Logistic Regression (scripts/train_classical_models.py, TF-IDF features)
        |     RoBERTa (pre-trained, cardiffnlp twitter-roberta-base-sentiment)
        |     DistilBERT (fine-tuned; notebooks/3.DistillBERTFineTuning.ipynb,
        |                 70/15/15 split, trained on an EARLIER corpus version)
        |     -> scripts/evaluate_models.py: shared 80/20 held-out comparison
        |     -> evaluate/evaluate_sentiment_diagnostics.py: per-platform,
        |        binary-vs-3-class, VADER threshold sweep, McNemar tests
        |
        +-- Complaint category discovery
        |     BERTopic + sentence-transformer embeddings + HDBSCAN clustering
        |     -> topic merging into a manual taxonomy
        |     -> zero-shot classification (nlp/categoriser.py) assigns each
        |        review to a taxonomy category at inference time
        |     -> evaluate/evaluate_category_discovery.py: coherence, frequency,
        |        per-platform/per-sentiment distribution
        |
        +-- Retrieval
        |     TF-IDF baseline (scripts/evaluate_tf_idf.py)
        |     Semantic search: SentenceTransformer (all-MiniLM-L6-v2) embeddings
        |     + FAISS index (scripts/build_index.py, scripts/evaluate_semantic_search.py)
        |
        +-- RAG chatbot (rag/pipeline.py)
        |     LangChain LCEL: create_history_aware_retriever + create_retrieval_chain
        |     + create_stuff_documents_chain, Groq-hosted llama-3.1-8b-instant
        |     Custom retriever (_make_grounded_retriever): similarity threshold
        |     (0.35) intersected with MMR diversity selection
        |     Platform-name detection narrows retrieval by metadata filter
        |     Two-stage scope guard (keyword + LLM classifier) + hard post-chain
        |     backstop that discards ungrounded generations
        |     -> evaluate/evaluate_llm_vs_rag.py: RAGAS (faithfulness, answer
        |        relevancy, context precision/recall) comparing RAG vs standalone LLM
        |
        +-- Statistical testing
              McNemar's exact test for pairwise model comparison (scripts or
              evaluate/ scripts producing data/results/metrics_summary/significance_tests.csv)
```

## Serving layer (built during this engagement, not part of the dissertation's core research contribution)

- **FastAPI backend** (`backend/`) — REST endpoints wrapping the above:
  `/api/sentiment`, `/api/search`, `/api/rag/chat`, `/api/evaluation/*`.
  All blocking ML calls wrapped in `asyncio.to_thread`.
- **Streamlit frontend** (`frontend/`) — six pages: Dashboard, Analyse (single
  review → sentiment + category + similar reviews + LLM business summary),
  Search (semantic search UI), Chatbot (the RAG system), Upload, Evaluation
  (reads `data/results/` and renders RAGAS/sentiment/retrieval comparisons).
- This layer is an application built *on top of* the research artefacts to
  make them usable/demonstrable — it is not itself a research contribution
  and should not be presented as one in the dissertation. Where it's
  mentioned at all, it belongs as evidence of applied engineering competence
  (relevant to the "technically challenging" / "professionalism" language in
  the Presentation-of-Results and Reproducibility criteria), not as a
  research result.

## Known architectural facts relevant to the audit ahead

1. **DistilBERT's training corpus and the shared comparison corpus are not
   the same file.** Confirmed earlier via `evaluate/measure_split_overlap.py`
   and the notebook's own saved cell outputs: DistilBERT trained on a
   592,354-row corpus (27.8% neutral) that no longer exists on disk; the
   current corpus used for the shared 80/20 comparison is 642,692 rows
   (33.1% neutral). This is a **Problem analysis & design choices** and
   **Reproducibility** issue, not just a numbers issue — it affects whether
   the shared-comparison DistilBERT figure can be presented as a clean,
   controlled comparison.

2. **RAG evaluation sample is small** — `evaluate/evaluate_llm_vs_rag.py`
   runs 5 hand-authored questions with hand-authored `GROUND_TRUTHS`. This
   must be stated plainly wherever RAGAS numbers are reported; it directly
   affects how the "Critical review of results" and "Presentation of
   results" criteria will read the RAG evidence.

3. **RAG ablation status needs verification against actual saved outputs**
   before Chapter 4/5 claim it — this is Phase 7's job specifically, but
   flagging now since it's the rubric's own named example of what
   Distinction-band Presentation of Results looks for.

4. **Two separate LLM-generation pipelines exist**: the RAG chatbot
   (`rag/pipeline.py`, the dissertation's actual defined use case —
   "conversational exploration of customer complaints across platforms")
   and a secondary single-review business-insight generator
   (`nlp/summariser.py`) used by the Analyse page, which also retrieves
   similar reviews but is not the system evaluated with RAGAS. Chapter
   3/4/5 should be careful not to conflate the two — only the first is
   "the RAG system" for dissertation purposes.

5. **Five sentiment models, not a single pipeline** — VADER (lexicon),
   Naive Bayes + Logistic Regression (classical, TF-IDF), RoBERTa
   (pre-trained transformer), DistilBERT (fine-tuned transformer). This
   breadth is itself relevant evidence for "technically challenging" under
   Problem Analysis & Design Choices, provided each is justified against
   literature rather than just listed.

Next: Phase 3, the Master Results Register, which requires opening every
file under `data/results/` (roughly 40 files across 8 subfolders — sentiment
comparison/reports/confusion matrices, sentiment diagnostics, category
discovery, TF-IDF and semantic-search retrieval evaluation, embedding-model
comparison, and the RAG vs LLM comparison). That's the next piece of work.
