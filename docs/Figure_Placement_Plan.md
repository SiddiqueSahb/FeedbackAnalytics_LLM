# Figure Placement Plan

Every figure below already exists in the repository. The plan assigns each one a number, a
chapter section, an exact insertion point, and a ready-to-use caption.

Two source directories are involved:

- `data/results/…` — figures written by the evaluation scripts
- `docs/figures/…` — figures produced for the dissertation

**Insertion points are given as "immediately after the paragraph ending …" with the actual
closing words of that paragraph**, so there is no ambiguity about placement.

Total: **16 figures** — 1 in Chapter 3, 15 in Chapter 4. Chapter 5 reuses them by reference
rather than reprinting, which avoids the duplication the marking scheme penalises.

---

## Chapter 3 — Research Approach

### Figure 3.1 — Research workflow

| | |
|---|---|
| **Source file** | `docs/figures/fig_architecture_publication.png` |
| **Section** | 3.3 Research Process |
| **Insert after** | The numbered five-stage list, immediately after item 5 ending "…diagnostic experiments that test the cause of unexpected results." — this replaces the existing `*(Figure 3.1 — Research workflow diagram …)*` placeholder |
| **Caption** | **Figure 3.1** — End-to-end research workflow. Three raw review sources are unified into a single corpus, which feeds three independently evaluated stages: sentiment classification, complaint discovery and taxonomy consolidation, and retrieval-augmented question answering. Evaluation outputs branch from each stage. |
| **Introduced by** | The sentence already in place: "The workflow runs in five stages, each evaluated before the next depends on it." |
| **Why needed** | The Problem Analysis criterion (20%) rewards clear design documentation. A reader cannot hold a five-stage pipeline with three evaluation regimes in mind from prose alone. |

> **If `fig_architecture_publication.png` is the cleaner of the two**, use it here and keep
> `fig_architecture_full.png` for an appendix. Do not print both in the main text — they show
> the same system at different detail levels, and printing both invites the "unbalanced
> presentation" criticism in the Report Style band.

**The five `fig_sub*.png` stage diagrams** (`fig_sub1_data_pipeline`, `fig_sub2_sentiment`,
`fig_sub3_discovery`, `fig_sub4_categorisation`, `fig_sub5_rag`) are **not** recommended for the
main text. Five additional architecture diagrams alongside 15 results figures would unbalance
the report. Place them in an appendix titled "Component architecture diagrams" and reference
that appendix once, in Section 3.3.

---

## Chapter 4 — Data Analysis

### Figure 4.1 — Review length by platform

| | |
|---|---|
| **Source file** | `data/results/figures/fig_4_1_review_length_by_platform.png` |
| **Section** | 4.2.2 Register differences |
| **Insert after** | The paragraph ending "…and it is what makes the per-platform analysis in Section 4.5.4 meaningful." |
| **Caption** | **Figure 4.1** — Review length distribution by platform, on a log scale, with median word counts. Seeded sample of up to 40,000 reviews per platform. Yelp reviews are substantially longer at the median; Twitter Airline reviews are both the shortest and the most uniform. |
| **Introduced by** | Already introduced: "Figure 4.1 reports word counts from a seeded sample of up to 40,000 reviews per platform." |
| **Why needed** | Establishes empirically that the corpus is multi-domain, which the entire cross-platform argument depends on. |

### Figure 4.2 — Overall model performance

| | |
|---|---|
| **Source file** | `docs/figures/fig_metrics_overall.png` |
| **Section** | 4.5.1 Aggregate sentiment results |
| **Insert after** | The paragraph ending "…no model is benefiting from an uneven class prior." |
| **Caption** | **Figure 4.2** — Sentiment classification performance for all five models on the shared held-out set (n = 128,539): accuracy with macro-averaged precision, recall and F1. The gap between accuracy and macro F1 is widest for VADER and RoBERTa, indicating uneven performance across classes. |
| **Introduced by** | Add this sentence at the end of that paragraph: "Figure 4.2 shows the same comparison graphically, where the divergence between accuracy and macro F1 for the two off-the-shelf models is immediately visible." |
| **Why needed** | The accuracy-versus-macro-F1 gap is the visual signature of the study's central anomaly, and it reads faster from a chart than from Table 4.5. |

### Figure 4.3 — Confusion matrices

| | |
|---|---|
| **Source file** | `data/results/figures/fig_4_2_confusion_matrices.png` |
| **Section** | 4.5.2 Per-class analysis |
| **Insert after** | The paragraph ending "…This is the systematic positive skew visible in the confusion matrices." |
| **Caption** | **Figure 4.3** — Row-normalised confusion matrices for all five models on the shared test set (n = 128,539). VADER and RoBERTa both push the majority of genuinely neutral reviews to the polar classes, behaving as effectively binary classifiers on a three-class task. |
| **Introduced by** | Amend the existing sentence to: "This is the systematic positive skew visible in the confusion matrices of Figure 4.3." |
| **Why needed** | This is the single most diagnostic figure in the dissertation. It shows *where* the failure occurs, not merely that it occurs. |
| **Note** | The filename says `fig_4_2` but this is now Figure 4.3. Either rename the file or accept the mismatch — the caption in the document is what the examiner reads. Renaming is cleaner. |

### Figure 4.4 — Per-class performance

| | |
|---|---|
| **Source file** | `docs/figures/fig_metrics_per_class.png` |
| **Section** | 4.5.2 Per-class analysis |
| **Insert after** | Table 4.7 (destination of true-neutral reviews) and the paragraph ending "…it labels 40.4% of genuinely negative reviews as positive." |
| **Caption** | **Figure 4.4** — Precision, recall and F1 by sentiment class for all five models. The neutral class separates the models far more sharply than the negative or positive classes, and the precision–recall split shows the failure to be one of recall rather than precision. |
| **Introduced by** | Add: "Figure 4.4 shows the same per-class breakdown across all three classes, where the isolation of the failure to the neutral class is clearest." |
| **Why needed** | Supports the precision-versus-recall diagnosis that underpins the label-mismatch argument in Chapter 5. |

### Figure 4.5 — Pairwise significance

| | |
|---|---|
| **Source file** | `docs/figures/fig_metrics_significance.png` |
| **Section** | 4.5.3 Statistical significance of the differences |
| **Insert after** | The final paragraph of the section, ending "…and Naive Bayes a bag-of-words classifier from a much earlier generation of methods." |
| **Caption** | **Figure 4.5** — Pairwise model comparison under McNemar's exact test on the balanced diagnostic sample (n = 1,800). Nine of ten comparisons are significant at p < 0.05; the Naive Bayes / RoBERTa pair is the single exception at p = 0.82. |
| **Introduced by** | Add: "Figure 4.5 presents these comparisons visually, with the single non-significant pair marked." |
| **Why needed** | Significance testing is a strength of this study and worth making visible rather than leaving buried in a ten-row table. |

### Figure 4.6 — Per-platform performance

| | |
|---|---|
| **Source file** | `data/results/sentiment_diagnostics/per_platform_metrics.png` |
| **Section** | 4.5.4 Diagnostic experiments, "Per-platform performance" |
| **Insert after** | The paragraph ending "…Only the fine-tuned model is stable across all three platforms, with a spread of two points." |
| **Caption** | **Figure 4.6** — Macro F1 by model and platform, 600 reviews per platform. RoBERTa outperforms both classical models on Twitter Airline while falling well below them on Yelp; Naive Bayes shows the inverse pattern. Only the fine-tuned model is stable across all three. |
| **Introduced by** | The existing text already leads into it. Amend to: "…with a spread of two points, as Figure 4.6 shows." |
| **Why needed** | **This is the most important figure in the dissertation.** The reversal of ranking by platform is the central finding, and it is far more persuasive as a chart than as Table 4.9. |

### Figure 4.7 — Three-class versus binary

| | |
|---|---|
| **Source file** | `data/results/sentiment_diagnostics/binary_comparison.png` |
| **Section** | 4.5.4 Diagnostic experiments, "Binary re-evaluation" |
| **Insert after** | The paragraph ending "…so once the neutral class is removed the ranking implied by the literature is restored." |
| **Caption** | **Figure 4.7** — Three-class against forced-binary macro F1 by model, diagnostic sample. The gain from removing the neutral class is largest for the two models never trained on this corpus, and smallest for the fine-tuned model. |
| **Introduced by** | Add: "Figure 4.7 makes the ordering of those gains apparent." |
| **Why needed** | Shows that the size of the gain is itself patterned — the more task-specific training a model has had, the less it gains. That pattern is hard to see in a table. |

### Figure 4.8 — VADER threshold sweep

| | |
|---|---|
| **Source file** | `data/results/sentiment_diagnostics/vader_threshold_sweep.png` |
| **Section** | 4.5.4 Diagnostic experiments, "VADER threshold sensitivity" |
| **Insert after** | The paragraph ending "…Threshold choice explains part of VADER's weakness on the neutral class, not most of it." |
| **Caption** | **Figure 4.8** — VADER macro F1, neutral F1 and accuracy across neutral-band widths, diagnostic sample. Macro F1 peaks at ±0.25 and declines thereafter; the improvement over the default band is 0.026, insufficient to account for the gap to the classical models. |
| **Introduced by** | Add: "Figure 4.8 shows the full sweep, including the decline beyond the optimum." |
| **Why needed** | This figure supports a *negative* result — it shows the intuitive explanation being ruled out, which is the strongest kind of diagnostic evidence. |

### Figure 4.9 — Topic reduction

| | |
|---|---|
| **Source file** | `data/results/category_discovery/discovery_topic_reduction.png` |
| **Section** | 4.4.2 Complaint discovery |
| **Insert after** | The final paragraph, ending "…each named by a language model given its constituent topic labels and keywords." |
| **Caption** | **Figure 4.9** — Reduction of 121 raw per-platform topics to 24 merged categories by agglomerative clustering over topic centroid embeddings. The compression ratio differs sharply by platform: Amazon's 69 topics are consolidated substantially, Yelp's 18 barely at all. |
| **Introduced by** | Add: "Figure 4.9 shows the reduction and how unevenly it falls across the three platforms." |
| **Why needed** | The asymmetric compression is the structural cause of the catch-all category discussed in Chapter 5. Showing it here sets up that argument. |

### Figure 4.10 — Category frequency

| | |
|---|---|
| **Source file** | `data/results/category_discovery/category_frequency.png` |
| **Section** | 4.5.5 Taxonomy evaluation |
| **Insert after** | The paragraph ending "…The remaining 199 reviews are distributed across 20 categories, three of which hold a single review." |
| **Caption** | **Figure 4.10** — Category frequency on the held-out sample (n = 450). Two categories — the largest substantive category and the unclassified bucket — account for 55.8% of assignments, while three categories hold a single review each. |
| **Introduced by** | Add: "Figure 4.10 shows the full distribution, in which the imbalance is stark." |
| **Why needed** | The long-tailed distribution is the evidence for the catch-all criticism. A table of 21 rows does not communicate the shape; this does. |

### Figure 4.11 — Coherence against category size

| | |
|---|---|
| **Source file** | `data/results/category_discovery/category_coherence.png` |
| **Section** | 4.5.5 Taxonomy evaluation |
| **Insert after** | Table 4.13 and the paragraph ending "…These are excluded from interpretation." |
| **Caption** | **Figure 4.11** — Mean within-category coherence by category. Coherence falls as category size rises: the two largest categories record the lowest coherence of any category holding more than three reviews, the signature of catch-all behaviour. |
| **Introduced by** | Add: "Figure 4.11 shows this inverse relationship across all categories." |
| **Why needed** | Makes the inverse size–coherence relationship visible, which is the quantitative basis for calling the largest category a catch-all rather than a legitimate large category. |

### Figure 4.12 — Platform distribution by category

| | |
|---|---|
| **Source file** | `data/results/category_discovery/platform_category_distribution.png` |
| **Section** | 4.5.5 Taxonomy evaluation |
| **Insert after** | Figure 4.11, before the "Three defects are visible…" paragraph |
| **Caption** | **Figure 4.12** — Distribution of categories across the three platforms. Some categories draw from all three sources; others are effectively single-platform, indicating that cross-domain transferability varies considerably between categories. |
| **Introduced by** | Add: "Figure 4.12 shows how categories distribute across platforms, and how unevenly." |
| **Why needed** | Directly evidences the "populated across platforms" success criterion set in Section 4.1.3, and shows it is only partly met. |

### Figure 4.13 — Sentiment distribution by category

| | |
|---|---|
| **Source file** | `data/results/category_discovery/sentiment_by_category.png` |
| **Section** | 4.5.5 Taxonomy evaluation |
| **Insert after** | The paragraph ending "…regardless of whether it expresses a complaint at all." |
| **Caption** | **Figure 4.13** — Sentiment composition of each category. Several complaint categories contain predominantly positive reviews — one receives 19 positive and no negative — demonstrating that the taxonomy has no mechanism for rejecting non-complaints. |
| **Introduced by** | Amend the existing text to end: "…regardless of whether it expresses a complaint at all. Figure 4.13 shows the extent of this across the taxonomy." |
| **Why needed** | This is the visual proof of the design error identified in Chapter 5. It is the single most damaging piece of evidence against the taxonomy, and printing it demonstrates the willingness to report unfavourable results that the marking scheme rewards. |

### Figure 4.14 — Retrieval latency

| | |
|---|---|
| **Source file** | `data/results/comparison_embedding_models/average_retrieval_time.png` |
| **Section** | 4.5.6 Retrieval evaluation |
| **Insert after** | The paragraph ending "…On this evidence the case for dense retrieval rests on latency, not on retrieval quality." |
| **Caption** | **Figure 4.14** — Mean retrieval latency, dense semantic search against TF-IDF, over 30 queries on 642,692 indexed reviews. Dense retrieval is approximately 5.9 times faster; its maximum latency remains below the TF-IDF minimum. |
| **Introduced by** | Add: "Figure 4.14 shows the latency difference, which is the one dimension on which the two methods clearly diverge." |
| **Why needed** | Latency is the only defensible advantage found for dense retrieval, so it is the one worth showing. |

> **Do not print** `average_precision_at_3.png` or `average_similarity_score.png` in the main
> text. Precision@3 is identical between methods, so a chart of two equal bars communicates
> nothing that the sentence does not. Similarity scores are not comparable across sparse and
> dense representations, so charting them side by side would imply a comparison the text
> explicitly disclaims. Both belong in an appendix if anywhere.

### Figure 4.15 — RAG against standalone LLM: relevancy

| | |
|---|---|
| **Source file** | `data/results/llm_vs_rag/answer_relevancy.png` |
| **Section** | 4.5.7 Generation evaluation |
| **Insert after** | Table 4.16 and the paragraph ending "…the only question in the set without a specific topic to retrieve against." |
| **Caption** | **Figure 4.15** — Answer relevancy per question, retrieval-augmented pipeline against standalone LLM (n = 5). The baseline achieves higher scores on four questions and zero on the fifth; the grounded pipeline varies within a much narrower band. |
| **Introduced by** | Add: "Figure 4.15 shows the per-question relevancy scores for both systems, where the difference in failure mode is clearer than in the means." |
| **Why needed** | This is the figure that carries the study's most subtle finding — that the two systems differ in the *shape* of their performance, not merely its level. |

### Figure 4.16 — Generation latency

| | |
|---|---|
| **Source file** | `data/results/llm_vs_rag/latency_comparison.png` |
| **Section** | 4.5.7 Generation evaluation |
| **Insert after** | Figure 4.15 |
| **Caption** | **Figure 4.16** — Mean end-to-end response latency, retrieval-augmented pipeline against standalone LLM (n = 5 questions). Grounding roughly triples response time; retrieval itself accounts for approximately one per cent of the total. |
| **Introduced by** | Add: "Figure 4.16 shows the corresponding latency cost." |
| **Why needed** | Supports the cost argument in Section 5.5.2 with measured evidence rather than assertion. |

> **`faithfulness.png`, `context_precision.png` and `context_recall.png`** are better left out of
> the main text. With n = 5 and no baseline to compare against, each is a five-bar chart of
> numbers already given exactly in Table 4.16. Printing three of them would overstate the weight
> of the weakest evidence in the study. If you want one, use `context_precision.png` in an
> appendix, since the 0.000 score is the finding that matters.

---

## Chapter 5 — Discussion

**No new figures.** Every argument in Chapter 5 refers back to evidence already presented in
Chapter 4, and the marking scheme's Report Style criterion rewards balanced presentation.
Reprinting results figures in the discussion is the most common way dissertations become
unbalanced.

Where Chapter 5 needs to point at evidence, use a cross-reference in the text — for example,
in Section 5.1.2: "as Figure 4.6 showed". Four such references are worth adding, at
Sections 5.1.1 (Figure 4.3), 5.1.2 (Figure 4.6), 5.2.3 (Figures 4.10 and 4.13) and
5.3.2 (Figure 4.15).

---

## Figures referenced in the drafts that do not exist

Three placeholders currently in the chapter text have no corresponding file. Each needs either
creating or removing — leaving a reference to a figure that is not printed is a presentation
error an examiner will notice.

| Placeholder | Location | Recommendation |
|---|---|---|
| Figure 2.1 — taxonomy of approaches (three-column diagram) | Chapter 2, §2.1.1 | **Create it.** It is a simple three-column diagram and Chapter 2 currently has no figure at all, which is unusual for a 20%-weighted literature chapter. Column 1 Sentiment classification: lexicon / classical ML / pretrained transformer / fine-tuned transformer. Column 2 Complaint discovery: LDA / NMF / BERTopic. Column 3 Feedback exploration: standalone LLM / RAG. Beneath each: supervised metrics, coherence measures, reference-free generation metrics. |
| DistilBERT training curves | Chapter 4, §4.4.1 | **Remove the placeholder.** Table 4.3 already gives all four epochs exactly, and the overfitting signature is described in the text. A four-point line chart adds nothing. |
| Category coherence as a separate figure | Chapter 4, §4.5.5 | Already covered by Figure 4.11 above; no additional figure required. |

---

## Summary of numbering changes to apply in Chapter 4

The chapter text currently references figures under an older numbering. Apply these
replacements:

| Currently in text | Becomes | File |
|---|---|---|
| Figure 4.1 | Figure 4.1 (unchanged) | `figures/fig_4_1_review_length_by_platform.png` |
| Figure 4.4 (overall performance) | **Figure 4.2** | `docs/figures/fig_metrics_overall.png` |
| Figure 4.2 (confusion matrices) | **Figure 4.3** | `figures/fig_4_2_confusion_matrices.png` |
| Figure 4.5 (per-class) | **Figure 4.4** | `docs/figures/fig_metrics_per_class.png` |
| Figure 4.6 (McNemar) | **Figure 4.5** | `docs/figures/fig_metrics_significance.png` |
| Figure 4.7 (per-platform) | **Figure 4.6** | `sentiment_diagnostics/per_platform_metrics.png` |
| Figure 4.8 (binary) | **Figure 4.7** | `sentiment_diagnostics/binary_comparison.png` |
| Figure 4.9 (VADER sweep) | **Figure 4.8** | `sentiment_diagnostics/vader_threshold_sweep.png` |
| — (new) | **Figure 4.9** | `category_discovery/discovery_topic_reduction.png` |
| Figure 4.10 (category frequency) | Figure 4.10 (unchanged) | `category_discovery/category_frequency.png` |
| Figure 4.13 (coherence) | **Figure 4.11** | `category_discovery/category_coherence.png` |
| Figure 4.11 (platform dist.) | **Figure 4.12** | `category_discovery/platform_category_distribution.png` |
| Figure 4.12 (sentiment by cat.) | **Figure 4.13** | `category_discovery/sentiment_by_category.png` |
| Figure 4.14 (retrieval) | Figure 4.14 (unchanged) | `comparison_embedding_models/average_retrieval_time.png` |
| Figure 4.15 (RAG) | Figure 4.15 (relevancy) | `llm_vs_rag/answer_relevancy.png` |
| — (new) | **Figure 4.16** | `llm_vs_rag/latency_comparison.png` |
| Figure 4.3 (training curves) | **Delete** | does not exist |
