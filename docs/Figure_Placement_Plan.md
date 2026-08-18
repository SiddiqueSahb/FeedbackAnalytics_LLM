# Figure Placement Plan

**Status: all figures are now placed inline in the chapter files.** This document is the
reference record of what sits where and why — the chapters themselves are authoritative if the
two ever disagree.

Every figure below is embedded directly in its chapter with a bold caption beneath it. Paths in
the chapters are relative, so the markdown renders correctly from the `docs/` directory.

Source directories:

- `docs/figures/…` — diagrams produced for the dissertation
- `data/results/…` — figures written by the evaluation scripts

---

## Summary

| Chapter | Figures | Numbering |
|---|---:|---|
| 2 — Literature Review | 1 | 2.1 |
| 3 — Research Approach | 3 | 3.1 – 3.3 |
| 4 — Data Analysis | 15 | 4.1 – 4.15 |
| 5 — Discussion | 0 | cross-references only |
| 6 — Conclusion | 0 | — |
| **Total** | **19** | |

Nineteen figures across roughly 1,300 lines of chapter text is a defensible density — heavy
enough that the results chapter is genuinely visual, light enough that no chapter reads as
padded. Chapter 5 deliberately carries none, for the reason given at the end of this document.

---

## Chapter 2 — Literature Review

| | |
|---|---|
| **Figure 2.1** | `docs/figures/fig_2_1_taxonomy.png` |
| **Section** | 2.1.1 Research context |
| **Placed after** | The sentence "…and where each method family sits." |
| **Caption** | Taxonomy of approaches to customer feedback analysis. The three tasks are analytically distinct, draw on different method families, and are judged by different evaluation regimes — which is why this study compares methods stage by stage rather than in a single combined table. |
| **Why it earns its place** | Chapter 2 carries 20% of the mark and would otherwise contain no figure at all. This one also does argumentative work rather than decoration: it establishes the stage-wise structure that Chapter 3 then justifies and Chapter 4 follows. A paragraph beneath it draws out the point that reading *down* a column gives a rankable sequence while reading *across* does not. |
| **Produced by** | Standalone matplotlib script; regenerate from `docs/figures/` if the method families change. |

---

## Chapter 3 — Research Approach

| Number | File | Section | Placed after |
|---|---|---|---|
| **3.1** | `docs/figures/research_workflow.png` | 3.3 Research Process | The five-stage numbered list, before the paragraph on stage ordering |
| **3.2** | `docs/figures/rag_architecture.png` | 3.6.4 Retrieval and generation | The paragraph introducing the two guards and the standalone baseline |
| **3.3** | `docs/figures/arch_evaluation_framework.png` | 3.6.5 Evaluation metrics | The opening sentence of the section |

**Captions**

- **Figure 3.1** — End-to-end research workflow. Three raw review sources are unified into a
  single corpus feeding three independently evaluated stages: sentiment classification, complaint
  discovery and taxonomy consolidation, and retrieval-augmented question answering.
- **Figure 3.2** — Retrieval-augmented question-answering architecture. The scope guard runs
  before retrieval is attempted; the similarity threshold gates retrieval output before
  generation; a post-generation check catches multi-turn queries whose reformulated question
  bypassed the pre-check.
- **Figure 3.3** — Evaluation framework across the three stages. Each stage uses measures
  appropriate to what it permits: supervised metrics where labels exist, unsupervised indicators
  where they do not, and reference-free generation metrics for the question-answering stage.

**Note on Figure 3.2.** This one matters more than it did in earlier drafts. The order of the two
guards relative to retrieval is exactly what Chapter 4 had to trace to diagnose the unanswered
customer-service question correctly — the scope guard fires *before* retrieval, which is why a
zero on three retrieval metrics did not mean what it first appeared to mean. The figure makes
that ordering visible rather than leaving it buried in prose.

### Deliberately not included in Chapter 3

| File | Reason |
|---|---|
| `arch_data_preparation.png`, `stage_data_preparation.png` | Section 3.5 describes the cleaning and unification steps completely in prose and in Table form. A diagram would restate, not add. |
| `arch_sentiment_classification.png`, `stage_sentiment_classification.png` | Same — Section 3.6.1 already specifies every model and hyperparameter. |
| `arch_complaint_discovery.png`, `stage_complaint_discovery.png` | Same for Section 3.6.2. |
| `pipeline_overview.png` | Duplicates `research_workflow.png` at lower detail. Use one or the other, never both. |
| `fig_architecture_full.png`, `fig_architecture_publication.png`, `fig_sub1`–`fig_sub5` | Earlier iterations superseded by the figures above. |

Three figures in a methodology chapter is the right number. Six would invite the "unbalanced
presentation" criticism under Report Style, and the four excluded diagrams carry no information
the prose does not already give precisely.

---

## Chapter 4 — Data Analysis

All fifteen are embedded inline, numbered in order of first appearance.

| Number | Source file | Section |
|---|---|---|
| **4.1** | `data/results/figures/fig_4_1_review_length_by_platform.png` | 4.2.2 Register differences |
| **4.2** | `docs/figures/fig_metrics_overall.png` | 4.5.1 Overall performance |
| **4.3** | `data/results/figures/fig_4_2_confusion_matrices.png` | 4.5.2 Per-class behaviour |
| **4.4** | `docs/figures/fig_metrics_per_class.png` | 4.5.2, immediately after 4.3 |
| **4.5** | `docs/figures/fig_metrics_significance.png` | 4.5.3 Significance testing |
| **4.6** | `data/results/sentiment_diagnostics/per_platform_metrics.png` | 4.5.4 Diagnostics |
| **4.7** | `data/results/sentiment_diagnostics/binary_comparison.png` | 4.5.4 |
| **4.8** | `data/results/sentiment_diagnostics/vader_threshold_sweep.png` | 4.5.4, after Table 4.11 |
| **4.9** | `data/results/category_discovery/category_frequency.png` | 4.5.5 Taxonomy |
| **4.10** | `data/results/category_discovery/platform_category_distribution.png` | 4.5.5 |
| **4.11** | `data/results/category_discovery/sentiment_by_category.png` | 4.5.5 |
| **4.12** | `data/results/category_discovery/category_coherence.png` | 4.5.5 |
| **4.13** | `data/results/comparison_embedding_models/average_retrieval_time.png` | 4.5.6 Retrieval |
| **4.14** | `data/results/llm_vs_rag/answer_relevancy.png` | 4.5.7 Generation |
| **4.15** | `data/results/ablation_metadata_tagging/metadata_tagging_ablation.png` | 4.5.8 Ablation |

**Figure 4.15 is the one that changed the chapter's standing against the marking scheme.**
"Ablation study is also done where appropriate" appears explicitly in the Distinction band for
Presentation of Results, and until this experiment ran, Chapter 4 had no ablation to show. It
now has one, with its result — a trade-off rather than a confirmation — reported honestly.

### Deliberately excluded from Chapter 4

| File | Reason |
|---|---|
| `discovery_topic_reduction.png` | The platform asymmetry it shows is given exactly in Table 4.4. Adding it would require renumbering everything after it for no gain. |
| `llm_vs_rag/latency_comparison.png` | Latency is stated exactly in Table 4.15 and discussed in §5.5.2. A two-bar chart adds nothing. |
| `comparison_embedding_models/average_precision_at_3.png` | Precision@3 is identical between methods — two equal bars communicate nothing. Stated in Table 4.14 instead. |
| `comparison_embedding_models/average_similarity_score.png` | Implies a cross-method comparison the text explicitly disclaims as invalid (dense and sparse cosine similarity measure different quantities). |
| `llm_vs_rag/faithfulness.png`, `context_precision.png`, `context_recall.png` | Three five-bar charts would give the study's weakest evidence its heaviest visual weight. Exact values are in Table 4.16. |
| `evaluation/tfidf/*`, `evaluation_semantic_search/*` | Single-method figures superseded by the direct comparison in Figure 4.13. |

The excluded set is as much a part of the presentation argument as the included one. Every figure
above was dropped because a table states the same thing more precisely, or because the chart
would imply a comparison the text disclaims.

---

## Chapter 5 — Discussion

**No new figures, by design.** Reprinting Chapter 4's figures in the discussion is the most
common cause of the "unbalanced presentation" criticism, and it also blurs the division of labour
the two chapters are built on: Chapter 4 presents evidence, Chapter 5 argues about it.

Cross-references to Chapter 4 figures are used instead, at these points:

| Section | Reference |
|---|---|
| 5.1.1 | Confusion matrices (Figure 4.3) when discussing the neutral-class collapse |
| 5.1.2 | Per-platform macro F1 (Figure 4.6) when discussing the ranking reversal |
| 5.2.3 | Category frequency and sentiment composition (Figures 4.9, 4.11) |
| 5.3.3 | Context precision per question (Figure 4.14, Table 4.16) |
| 5.4.3 | Metadata-tagging ablation (Figure 4.15) when discussing the trade-off finding |

---

## Chapter 6 — Conclusion

No figures. The chapter is a verdict on the objectives and a discussion of future work; it
reports no new evidence and should not appear to.

---

## Regenerating the figures

| Figure(s) | Script |
|---|---|
| 2.1 | Standalone matplotlib script (see `docs/figures/`) |
| 3.1 – 3.3 | Architecture diagram scripts |
| 4.1, 4.3 | `evaluate/generate_figures.py` |
| 4.2, 4.4, 4.5 | `evaluate/generate_all_metrics.py` |
| 4.6 – 4.8 | `evaluate/evaluate_sentiment_diagnostics.py` |
| 4.9 – 4.12 | `evaluate/evaluate_category_discovery.py` |
| 4.13 | `scripts/compare_embedding.py` |
| 4.14 | `evaluate/evaluate_llm_vs_rag.py` |
| 4.15 | `evaluate/ablation_metadata_tagging.py` |

Every figure in Chapters 4 is written to disk by the script that computed the underlying numbers,
so no figure in the results chapter was drawn by hand or transcribed. That is the property the
reproducibility criterion (10% of the mark) is asking about, and it is worth stating plainly in
the chapter text as well as here.
