# Structured Project Understanding — Verification Baseline

Purpose: an audited record of what this project *actually contains*, established by reading the
source code, configuration and evaluation outputs directly. Every figure below was read from a
file in the repository. This document is the factual baseline for all dissertation chapters, so
that no chapter asserts something the artefacts do not support.

Verified on: 11 August 2026.

---

## 1. Research problem

Organisations hold large volumes of unstructured customer feedback but lack an auditable route
from raw review text to a decision. Sentiment classification quantifies *how much*
dissatisfaction exists without identifying *what it concerns*; topic models surface keyword
clusters that are not business-ready categories; and large language models can answer analyst
questions fluently while providing no verifiable evidential basis for their claims.

## 2. Research gap

The gap is integrative, not algorithmic. The constituent techniques (transformer sentiment
classification, embedding-based topic modelling, dense retrieval, retrieval-augmented
generation) are mature. What is absent is an end-to-end configuration that (a) operates across
heterogeneous feedback domains rather than a single benchmark, (b) consolidates per-platform
topics into one cross-domain complaint taxonomy, (c) grounds generated insight in retrievable
source reviews, and (d) is evaluated at every stage with metrics appropriate to that stage —
including where results are unfavourable.

## 3. Contribution

1. A controlled five-model sentiment comparison on one multi-domain corpus (642,692 reviews),
   spanning lexicon, classical ML, pretrained transformer and fine-tuned transformer families.
2. An automated cross-platform taxonomy consolidation stage: per-platform BERTopic topics are
   merged by agglomerative clustering over topic centroid embeddings, then named by an LLM.
   This is automated, not manual.
3. A grounded complaint-exploration interface: FAISS retrieval + metadata-tagged context +
   prompt-level grounding constraints, with source review IDs returned for traceability.
4. A stage-appropriate evaluation suite: supervised metrics for sentiment, unsupervised
   coherence/distribution metrics for taxonomy quality, and RAGAS for generation faithfulness.

## 4. Methodology (as implemented)

**Stage 1 — Data preparation** (`scripts/preprocess.py`)
Three sources unified into `data/processed/reviews_unified.parquet`: 642,692 reviews
(Yelp 449,858; Amazon 178,883; Twitter Airline 13,951). Schema: `review_id, platform,
product_category, text, cleaned_text, rating, sentiment_label, date`.
Sentiment labels are **derived from star ratings** (`rating <= 2` → negative, `== 3` → neutral,
`>= 4` → positive); Twitter uses its native `airline_sentiment` field, back-mapped to a
pseudo-rating. Classes were deliberately balanced during sampling: negative 218,799,
neutral 212,503, positive 211,390.

**Stage 2 — Sentiment classification** (`scripts/train_classical_models.py`,
`scripts/evaluate_models.py`, `nlp/sentiment.py`, `notebooks/2`, `notebooks/3`)
Five models evaluated on a common held-out split: 20% stratified, `random_state=42`,
128,539 reviews. Metrics: accuracy, macro/weighted F1, macro precision/recall. MLflow tracking
configured. 5-fold StratifiedKFold used in classical training.

**Stage 3 — Complaint discovery** (`scripts/discover_categories.py`)
BERTopic per platform on negative-sentiment reviews only, English-filtered via langdetect.
Custom UMAP / HDBSCAN / CountVectorizer sub-models; `min_topic_size` auto-derived (0.15% of
corpus, capped 30–300). Cross-platform merge via agglomerative clustering on
`topic_embeddings_`, target 24 categories, then LLM naming per merged cluster.
Outputs: `complaint_categories_all_negative.json`, `taxonomy_merge_info_negative.json`,
`bertopic_info_<platform>_negative.json`.

**Stage 4 — Zero-shot categorisation** (`nlp/categoriser.py`)
Two-stage: MiniLM embedding shortlist (top 6 of 24) → DeBERTa-v3-base-zeroshot-v2.0 NLI rerank
using each category's description as the hypothesis. Confidence threshold 0.35; below it,
returns "Unclassified / Emerging Complaint".

**Stage 5 — Retrieval-Augmented Generation** (`rag/pipeline.py`, `rag/prompts.py`,
`scripts/build_index.py`)
FAISS index over all-MiniLM-L6-v2 embeddings (384-d, normalised). LangChain LCEL chain:
`create_history_aware_retriever` → `create_stuff_documents_chain` → `create_retrieval_chain`.
Retriever k=5, fetch_k=20. Each retrieved review is tagged with platform / rating / sentiment
before entering the prompt, which makes the prompt's grounding rules enforceable. Source
`review_id`s are returned alongside every answer. **LLM: Groq `llama-3.1-8b-instant`,
temperature 0.2.** A keyword out-of-scope guard plus an optional LLM scope classifier are
implemented.

## 5. Evaluation results (all read from repository files)

### 5.1 Sentiment classification — test set n = 128,539

| Model | Accuracy | Macro F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| VADER (lexicon) | 0.5142 | 0.4381 | 0.621 | **0.086** | 0.608 |
| Naive Bayes | 0.7558 | 0.7580 | 0.761 | 0.678 | 0.835 |
| Logistic Regression | 0.8034 | 0.8033 | 0.824 | 0.718 | 0.868 |
| RoBERTa (pretrained) | 0.6327 | 0.5694 | 0.743 | **0.225** | 0.740 |
| DistilBERT (fine-tuned) | **0.8243** | **0.8244** | 0.837 | 0.747 | 0.889 |

The headline finding is not simply that fine-tuning wins. It is that both *off-the-shelf*
approaches collapse specifically on the neutral class (VADER 0.086, RoBERTa 0.225), which is
what drags their aggregate scores below a plain TF-IDF logistic regression. This is a
label-semantics mismatch, not general model weakness: neither was built for a three-class
rating-derived scheme in which "neutral" means a 3-star review.

### 5.2 Complaint taxonomy

- Raw BERTopic topics: Amazon 69, Yelp 18, Twitter Airline 34 → **121 total**
- Documents clustered: Amazon 57,931; Yelp 149,838; Twitter Airline 8,890
- Merged taxonomy: **24 categories**; mean pairwise centroid cosine similarity **0.2831**
- Held-out evaluation (450 reviews, 150/platform, seed 42): 21 of 24 categories populated;
  mean within-category coherence **0.6334**; largest category "Product Performance Failures"
  127 (28.2%); **124 reviews (27.6%) unclassified**; three categories received a single review.

### 5.3 Retrieval

| Metric (n = 30 queries) | Semantic (MiniLM + FAISS) | TF-IDF |
|---|---|---|
| Mean retrieval time | 21.71 ms | 127.61 ms |
| Mean similarity score | 0.6833 | 0.5593 |
| Precision@3 | 0.90 | 0.90 |

Dense retrieval is ~5.9× faster and returns higher-similarity matches, but Precision@3 is
identical — so the advantage claimed must be latency and semantic matching, not accuracy.

### 5.4 RAG (RAGAS)

- Main run, n = 24: faithfulness **0.7862**, answer relevancy **0.7323**, fabrication risk
  **0.0000**, mean latency **6.18 s**
- MMR retriever run, n = 5: faithfulness 0.7551, answer relevancy 0.6339,
  context precision 0.6061, context recall 0.5333
- LLM-only baseline, n = 5: answer relevancy **0.9202** (no retrieved context, so faithfulness
  is undefined)

The LLM-only/RAG relevancy inversion is a genuine finding and should be reported, not buried:
unconstrained generation produced answers judged *more* relevant while having no verifiable
evidential basis. It demonstrates that answer relevancy alone cannot validate a RAG system.

## 6. Strengths

- Multi-domain corpus at scale with a genuinely large, stratified, seeded test split.
- Complete family coverage in the sentiment comparison (lexicon → classical → pretrained →
  fine-tuned), which is what makes RQ1 answerable rather than rhetorical.
- Two retrieval baselines (TF-IDF, embedding-model comparison) — the marking scheme explicitly
  credits baseline comparison.
- Metadata-tagged retrieval context, which converts prompt grounding rules from aspiration
  into something enforceable.
- Reproducibility discipline: fixed seeds throughout, MLflow tracking, pinned requirements,
  unit tests for the merge and shortlist logic.
- Ablation-shaped experiments already exist in `scripts/evaluate_rag.py` (top-k sweep,
  retriever comparison, prompt comparison, metadata-tagging ablation) — directly addressing the
  distinction criterion "ablation study is also done where appropriate".

## 7. Weaknesses (to be stated, not concealed)

1. **Labels are rating-derived, not human-annotated.** A 4-star review containing a complaint
   is labelled positive. This caps achievable accuracy and partly explains the neutral-class
   difficulty across all five models.
2. **Twitter labels come from a different annotation process** than Amazon/Yelp, so the corpus
   is not label-homogeneous.
3. **Taxonomy coherence of 0.2831 is low**, and 27.6% of held-out reviews fall below the
   classification threshold. One category absorbs 28% of reviews while three take a single
   review each — the taxonomy's granularity is uneven.
4. **No human-labelled gold set** for complaint categorisation, so no accuracy/F1 can be
   reported for Stage 4. This is a scope decision, and it must be defended explicitly.
5. **Context precision/recall computed on n = 5.** Too small to generalise from.
6. **Judge LLM is the same model as the generator** (Groq Llama 3.1 8B), so RAGAS scores carry
   self-grading bias.
7. **No standard topic-model metrics** (C_v / NPMI coherence, topic diversity) were computed.
8. **The configured zero-shot baseline is unused.** `ZEROSHOT_BASELINE_MODEL`
   (`facebook/bart-large-mnli`) is defined in `config.py` but never referenced — an ablation
   that is one config change away from existing.
9. **No user study or expert evaluation**, so business usefulness cannot be evidenced.
10. **API keys are hardcoded in `config.py`.** A reproducibility and research-ethics problem
    that must be fixed before submission or repository sharing.

## 8. Originality — mapped to the marking scheme's MSc novelty criteria

- *Identification of a unique problem/gap*: cross-domain complaint taxonomy construction with
  no ground truth, and its downstream effect on grounded generation.
- *Creative application of methods*: automated centroid-based cross-platform topic merging
  followed by LLM naming; two-stage embedding-shortlist-then-NLI categorisation, which bounds
  NLI cost as the taxonomy grows.
- *Exploration of emerging technologies*: RAGAS-based faithfulness evaluation of an LCEL RAG
  pipeline — recent tooling, thinly covered in existing literature.
- *Integration of multiple disciplines*: sentiment analysis, topic modelling, information
  retrieval and explainable AI combined in one pipeline.
- *Validation of novelty through results*: two findings that contradict the naive expectation
  — a pretrained transformer losing to logistic regression, and LLM-only scoring higher on
  answer relevancy than RAG.
- *Acknowledgment of limitations*: section 7 above.

## 9. Missing / not yet supplied

**Not uploaded** (referenced in the brief but absent): dissertation draft, literature review,
methodology chapter, notebook outputs, figures/tables document. Only the marking scheme PDF was
received. All citations therefore remain APA 7 placeholders.

**Not implemented in the codebase**: any dashboard or UI. `streamlit`, `fastapi` and `uvicorn`
appear in `requirements.txt`, but there is no `app.py`, `dashboard.py` or equivalent. The
"Business Insight Generation" stage exists as source-attributed generated answers, not as a
user-facing application.

**DistilBERT fine-tuning — now verified** (read from `notebooks/3.DistillBERTFineTuning.ipynb`
and `models/distilbert-finetuned-final/`):

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased`, 3-class head |
| Split | 70 / 15 / 15 stratified, `random_state=42` |
| Split sizes at training time | train 414,647 · val 88,854 · test 88,853 |
| Epochs | 4, with `EarlyStoppingCallback(patience=2)` |
| Batch size | 32 train / 64 eval |
| Learning rate | 2e-5, `warmup_steps=1000`, `weight_decay=0.01` |
| Precision | fp16 mixed precision, `dataloader_num_workers=2` |
| Class imbalance | balanced class weights via `compute_class_weight`, applied through a custom `WeightedTrainer` overriding `compute_loss` with a weighted `CrossEntropyLoss` |
| Model selection | `load_best_model_at_end`, `metric_for_best_model="f1_macro"` |
| Seed | 42 |
| Training runtime | 10,544 s (≈2 h 56 m), final training loss 0.3350 |

Per-epoch validation (from the notebook): epoch 1 acc 83.48 / macro F1 82.59; epoch 2 acc 84.08 /
macro F1 83.28 (best); epoch 3 acc 83.98 / 83.12; epoch 4 acc 83.73 / 82.80. Validation loss rose
after epoch 2, so early stopping and best-checkpoint restore were doing real work.

Clean test-set result (notebook, 88,854 held-out reviews): **accuracy 0.8437, macro F1 0.8375,
weighted F1 0.8454**. Per class F1: negative 0.8620, neutral 0.7393, positive 0.9113.
Per platform: Yelp acc 0.8552 / macro F1 0.8437 · Amazon 0.8188 / 0.8209 ·
Twitter Airline 0.8271 / 0.7714.

### Three issues found while verifying this

1. **Test-set contamination in the five-model comparison.** DistilBERT was trained on a 70/15/15
   split of the corpus. `scripts/evaluate_models.py` builds its own independent 80/20 split and
   evaluates all five models on it. Reproducing both splits shows roughly **25% of that
   comparison's test set was seen by DistilBERT during training or validation**. The DistilBERT
   figure in `model_comparison.json` (accuracy 0.8243) is therefore not a clean held-out result,
   while the other four models are unaffected. The clean figure is the notebook's 0.8437.
   Note the classical models were trained on yet another 80/20 split, so no single existing test
   set is uncontaminated for every model.

2. **The corpus was regenerated after training.** The notebook's splits total 592,354 reviews;
   the current `reviews_unified.parquet` holds 642,692. The exact rows DistilBERT trained on
   are therefore not recoverable, which means the 25% figure above is a close estimate and not
   an exact count.

3. **A tokenisation bug creating a train/inference mismatch.** The tokenise function passes
   `maxlength = 128` (cell 24), which is not a valid Hugging Face argument — the correct keyword
   is `max_length`. The misspelled argument is silently ignored, so with `padding='max_length'`
   and no valid length set, sequences were padded and truncated to the tokenizer's
   `model_max_length`, confirmed as **512** in the saved `tokenizer_config.json`. Inference in
   `nlp/sentiment.py` then uses `max_length=128`. The model was trained on up to 512 tokens but
   is served only the first 128, which plausibly contributes to the gap between the notebook
   result and the comparison-script result on long Amazon and Yelp reviews.

**Corrected from the brief**: the LLM is Groq `llama-3.1-8b-instant`, not Google Gemini. The
taxonomy merge is automated, not manual. The raw topic count entering the merge is 121, not 69.
