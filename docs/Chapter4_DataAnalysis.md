# Chapter 4 — Data Analysis

Chapter 3 set out a methodology built against four criticisms of the existing literature. This chapter reports what that methodology produced. Its purpose is to present the evidence in enough detail for the results to be reproduced and checked; the argument about what the evidence means is held back for Chapter 5. Where a result contradicts expectation, this chapter reports the contradiction and the diagnostic experiment that probes it, but leaves the explanation for the discussion.

Every figure quoted here is produced by a script that writes to `data/results/`, and the producing script is named alongside each table so that any number can be traced to its source.

The chapter follows the CRISP-DM phases adopted in Chapter 3. Section 4.1 restates the problem and defines the success criteria against which the work is judged. Section 4.2 reports the exploratory analysis of the three sources and the data quality issues it exposed. Section 4.3 covers preparation, feature extraction and splitting. Section 4.4 documents model development at each of the three pipeline stages. Section 4.5 reports evaluation, which supplies the evidence for all four research questions. Section 4.6 addresses deployment status, and Section 4.7 summarises.

---

## 4.1 Business Understanding

Chapter 1 established the problem: organisations hold more feedback than they can read, and the practical question is which method to trust to extract signal from it. Three analytically distinct sub-problems follow — supervised classification, unsupervised discovery, and grounded generation, where the risk is a fluent answer with no basis in evidence. This chapter delivers Objectives 2 to 6.

Success is defined per stage rather than as a single accuracy target. For classification, a controlled comparison in which differences between models cannot be attributed to differences in data, reported at a granularity that reveals class-level and platform-level behaviour; a ranking alone would not satisfy it. For discovery, with no ground truth, a taxonomy whose categories are separable, internally coherent, populated across platforms, and honest about what it fails to cover. For question answering, measurable grounding: answers traceable to identifiable source reviews, scored against a standalone baseline on the same questions.

---

## 4.2 Data Understanding

### 4.2.1 Register differences

The comparison in this study depends on the three sources differing in register, so that claim is measured rather than assumed. Figure 4.1 reports word counts from a seeded sample of up to 40,000 reviews per platform.

The differences are substantial and not in the direction intuition suggests. Yelp reviews are the longest by a wide margin, with a median of 85 words against Amazon's 29 and Twitter Airline's 19. Amazon reviews are shorter than expected at the median but carry a long tail: their 95th percentile reaches 138 words against Twitter's 27, so the platform mixes very brief and very detailed reviews. Twitter Airline is both the shortest and by far the most uniform, which follows from the platform's character limit — its interquartile range spans a narrower band than either of the others.

Two properties therefore separate the sources: central length, on which Yelp stands apart, and dispersion, on which Twitter stands apart. Amazon sits between them on both. This is the empirical basis for treating the corpus as multi-domain, and it is what makes the per-platform analysis in Section 4.5.4 meaningful.

![Figure 4.1](../data/results/figures/fig_4_1_review_length_by_platform.png){width=4.6in}

**Figure 4.1** — Review length by platform: distribution on a log scale, with median word counts. Seeded sample of up to 40,000 reviews per platform. *(Produced by `evaluate/generate_figures.py`.)*

### 4.2.2 Composition

After preparation the corpus holds 642,692 reviews. Table 4.1 gives the composition.

**Table 4.1** — Corpus composition by platform

| Platform | Reviews | Share | Label origin | Register |
|---|---:|---:|---|---|
| Yelp | 449,858 | 70.0% | Star rating | Extended service narrative |
| Amazon | 178,883 | 27.8% | Star rating | Variable-length product review |
| Twitter Airline | 13,951 | 2.2% | Human annotation | Short, abbreviated, public |
| **Total** | **642,692** | **100%** | | |

Sentiment classes are close to balanced: negative 218,799 (34.0%), neutral 212,503 (33.1%), positive 211,390 (32.9%). This is a consequence of balanced sampling during preparation, and it is what makes macro F1 a fair headline metric.

Platform representation, by contrast, is heavily skewed. Yelp contributes seventy per cent and Twitter Airline barely two. This asymmetry drives two later design decisions: topic discovery runs per platform (Section 4.4.2), and the diagnostic evaluation samples equally from each platform (Section 4.5.4).

### 4.2.3 Data quality issues

Four issues carry forward into the interpretation of results, and the first two are the consequential ones. Rating-derived labels mean a four-star review containing a specific complaint is labelled positive, placing a ceiling on achievable accuracy that no model can exceed. Label provenance is also not uniform: Twitter labels come from human annotation, so "neutral" is a three-star rating on two platforms and a judgement of no clear sentiment on the third — not one construct but two, which matters once the neutral class turns out to be where most models fail. Twitter text also contains handles, abbreviations and truncation absent elsewhere, and some non-English reviews are retained in the corpus but filtered at the discovery stage, where mixed languages cluster by language rather than subject.

---

## 4.3 Data Preparation

Preparation followed the procedure specified in Section 3.5: schema unification, removal of records missing text or labels, case and whitespace normalisation, rule-based labelling from star ratings, and class balancing during sampling. A four-record sample of the resulting corpus, showing the schema and the effect of cleaning, is given in Appendix B.2.

### 4.3.1 Feature extraction

Feature representation differs by model family, which is itself part of the comparison.

Classical models use TF-IDF vectors over unigrams and bigrams. VADER requires no features, operating directly on raw text through its lexicon. Transformer models use their own subword tokenisers, with DistilBERT truncating at its tokeniser maximum. Topic discovery and retrieval use 384-dimensional sentence-transformer embeddings from `all-MiniLM-L6-v2`, L2-normalised so that inner product equals cosine similarity.

No manual feature selection or scaling was applied. TF-IDF weighting handles term importance implicitly, and embedding dimensions are not individually interpretable, so selecting among them would not be meaningful.

### 4.3.2 Splitting

The sentiment comparison uses an 80/20 stratified split with `random_state=42`, producing a held-out test set of **128,539 reviews** with the class distribution preserved.

The fine-tuned model was trained on a separately generated 70/15/15 stratified split, also seeded at 42. Its validation partition drove early stopping and checkpoint selection; its test partition of 88,854 reviews was held out entirely.

As disclosed in Section 3.5.1, these two partitions are not identical, so a portion of the 128,539-review test set overlaps data the fine-tuned model saw in training. The uncontaminated figure for that model is the one from its own held-out partition, and it is that figure which is treated as definitive. Every comparison in Section 4.5.1 involving DistilBERT is annotated accordingly.

---

## 4.4 Model Development

### 4.4.1 Sentiment classifiers

Five models span four families, selected on the evidence reviewed in Section 2.3.1.

VADER serves as the lexicon baseline, requiring no training, with compound scores mapped to three classes at the conventional ±0.05 thresholds. Naive Bayes and logistic regression are trained over TF-IDF features, the latter using the lbfgs solver with `max_iter=3000`, with five-fold stratified cross-validation during development. RoBERTa is used pretrained and unmodified in its `twitter-roberta-base-sentiment-latest` form; it is deliberately not fine-tuned, since its role is to represent what transfer learning delivers without task-specific training.

DistilBERT was fine-tuned from `distilbert-base-uncased` with a three-class head. The full configuration is given in Appendix B.


Two choices warrant note. Class weights were applied through a custom trainer overriding the loss function, because the standard trainer ignores them; without this the neutral class would be under-served despite corpus-level balance. And checkpoint selection used macro F1 rather than validation loss, so that the quantity optimised matches the quantity reported.

Per-epoch training behaviour is reported in Appendix B.


Validation loss fell to epoch 2 then rose while training loss continued to decline — the standard signature of overfitting — so best-checkpoint restoration selected epoch 2. Training took 10,545 seconds.


### 4.4.2 Complaint discovery

BERTopic was applied to the negative-sentiment reviews of each platform separately, after English-language filtering. Running per platform rather than over the pooled corpus prevents Yelp's seventy per cent share from dominating the topic space. Table 4.2 records configuration and outcome.

**Table 4.2** — BERTopic discovery by platform *(source: `data/results/category_discovery/discovery_platform_topics.csv`)*

| Platform | Documents clustered | Minimum topic size | Topics discovered |
|---|---:|---:|---:|
| Amazon | 57,931 | 86 | 69 |
| Yelp | 149,838 | 224 | 18 |
| Twitter Airline | 8,890 | 30 | 34 |
| **Total** | **216,659** | — | **121** |

Minimum topic size was derived from each platform's document count rather than fixed, which is why it varies. The topic counts are themselves informative: Yelp produced only 18 topics from 149,838 documents while Amazon produced 69 from roughly a third as many. Product categories span a far wider semantic range than restaurant and service complaints, and the discovery stage reflects that.

The 121 raw topics were then merged by agglomerative clustering over topic centroid embeddings to a target of 24 categories, each named by a language model given its constituent topic labels and keywords.

### 4.4.3 Category assignment, retrieval and question answering

Assignment, retrieval and generation were implemented as specified in Sections 3.6.2 and 3.6.3: a six-category embedding shortlist scored by zero-shot entailment with a 0.35 confidence threshold; FAISS retrieval of five reviews per query by maximal marginal relevance, each tagged with platform, rating and sentiment before entering the prompt; and generation by Llama 3.1 8B at temperature 0.2 with source identifiers returned. The same questions were then answered with retrieval disabled, providing the RQ2 baseline.

---

## 4.5 Model Evaluation

### 4.5.1 Aggregate sentiment results

Table 4.3 reports all five models on the shared test set.

**Table 4.3** — Sentiment classification results, shared held-out set (n = 128,539) *(source: `data/results/metrics_summary/overall_metrics.csv`)*

| Model | Family | Accuracy | Balanced acc. | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| VADER | Lexicon baseline | 0.5142 | 0.5148 | 0.5213 | 0.5148 | 0.4381 | 0.4396 |
| Naive Bayes | Classical ML | 0.7558 | 0.7557 | 0.7621 | 0.7557 | 0.7580 | 0.7579 |
| Logistic Regression | Classical ML | 0.8034 | 0.8033 | 0.8033 | 0.8033 | 0.8033 | 0.8033 |
| RoBERTa (pretrained) | Pretrained transformer | 0.6327 | 0.6314 | 0.6056 | 0.6314 | 0.5694 | 0.5709 |
| DistilBERT (fine-tuned)† | Fine-tuned transformer | **0.8243** | **0.8245** | **0.8247** | **0.8245** | **0.8244** | **0.8244** |

† Partially contaminated: a portion of this test set overlaps DistilBERT's training data (Section 4.3.2). The clean figure follows below.

Balanced accuracy tracks accuracy closely for every model, confirming that the class balancing described in Section 4.2.3 worked: no model is benefiting from an uneven class prior.

On its own uncontaminated held-out partition of 88,854 reviews, the fine-tuned model achieved accuracy 0.8437, macro F1 0.8375 and weighted F1 0.8454. This is the figure treated as definitive for DistilBERT throughout the remainder of this dissertation. Its closeness to the shared-set figure of 0.8244 suggests the contamination did not materially inflate the shared-set result, but the clean figure is used regardless, since the argument should not depend on that judgement.


Two results contradict the ordering the literature in Section 2.1.2 would predict.

The first is VADER's weakness. Its accuracy of 0.5142 places just over half of reviews in the right class, but its macro F1 of 0.4381 is far lower, and the gap between the two is the finding: performance is unevenly distributed across classes in a way that accuracy alone conceals. For reference, a uniform random classifier on this balanced three-class problem would score approximately 0.333 on both measures, so VADER is well above chance — the point is not that it fails outright but that it fails asymmetrically.

The second is more consequential. The pretrained transformer falls below both classical models, with a macro F1 of 0.5694 against logistic regression's 0.8033 — a gap of 23 points in the direction opposite to the one the literature predicts.

### 4.5.2 Per-class analysis

The aggregates conceal where the failure occurs. The full per-class report is given in Appendix B; the pattern it exposes is what makes the anomaly legible.


Both off-the-shelf models perform acceptably on negative and positive while collapsing on neutral, and separating precision from recall shows the nature of that collapse. VADER's neutral precision is 0.377 while its neutral recall is 0.048; RoBERTa's are 0.522 and 0.143. Precision is moderate in both cases and recall is near zero. Neither model is systematically wrong about what neutral text looks like when it does apply the label — both simply almost never apply it.

A second pattern appears in the positive column. VADER records 0.963 recall against 0.444 precision, so it captures nearly every genuinely positive review while being wrong about most of what it calls positive. This is the systematic positive skew visible in the confusion matrices.


The destinations of misclassified neutral reviews, given in full in Appendix B, sharpen this. VADER assigns the neutral label to only 4.8% of genuinely neutral reviews, pushing 78.3% to positive; RoBERTa splits them between both poles, 36.2% negative and 49.5% positive. Neither fails uniformly — both behave as binary classifiers confronted with a three-class task. VADER's skew extends further: it labels 40.4% of genuinely negative reviews as positive.

### 4.5.3 Statistical significance of the differences

McNemar's exact test was applied to every model pair on the balanced diagnostic sample, since a difference in macro F1 is only meaningful if it exceeds what sampling variation would produce.

**An important scope note.** The test requires paired per-review predictions, which the saved confusion matrices do not retain. It was therefore computed on the balanced diagnostic sample of 1,800 reviews described in Section 4.5.4, not on the full test set. The accuracies reported in Appendix B are consequently those observed on that sample and differ from Table 4.3; the test speaks to pairwise correctness on the balanced sample, and is not a direct significance test of the macro F1 differences reported in Table 4.3.


Nine of the ten comparisons are significant at p < 0.05, most by very wide margins. Two deserve specific note.

The gap between DistilBERT and logistic regression is significant at p = 1.7 × 10⁻²¹, so on this sample the advantage of fine-tuning is not attributable to sampling variation.

Naive Bayes and RoBERTa, by contrast, cannot be separated. Of the 675 reviews on which they disagree, each is correct on almost exactly half, giving p = 0.82. This does not establish that the two models are equivalent — a non-significant result is not evidence of equality — but the available paired evidence does not support ranking either above the other on this sample, despite RoBERTa being a transformer and Naive Bayes a bag-of-words classifier from a much earlier generation of methods.


### 4.5.4 Error analysis and diagnostic experiments

Three diagnostic experiments test the mechanism behind these results, using a balanced sample of 1,800 reviews drawn equally across platform and class so that no platform dominates. Because the sample is balanced rather than representative, the three-class figures in this section differ from Table 4.3 and should be compared only within the section. A qualitative inspection of individual misclassified reviews follows the three experiments.

**Per-platform performance.**

**Table 4.4** — Macro F1 (neutral F1 in parentheses) by platform, 600 reviews per platform *(source: `data/results/sentiment_diagnostics/per_platform_metrics.csv`)*

| Model | Amazon | Yelp | Twitter Airline |
|---|---:|---:|---:|
| VADER | 0.4811 (0.152) | 0.4252 (0.019) | 0.5880 (0.480) |
| Naive Bayes | 0.7952 (0.728) | 0.7573 (0.693) | 0.4838 (0.154) |
| Logistic Regression | 0.7908 (0.724) | 0.7948 (0.721) | 0.6434 (0.509) |
| RoBERTa (pretrained) | 0.6218 (0.302) | 0.5544 (0.198) | **0.7853 (0.713)** |
| DistilBERT (fine-tuned) | 0.8448 (0.782) | 0.8301 (0.748) | 0.8505 (0.836) |

The result is decisive. RoBERTa scores 0.7853 on Twitter Airline against 0.5544 on Yelp — a spread of 23 points, with neutral F1 moving from 0.713 to 0.198 — and on Twitter it **outperforms both classical models**, beating logistic regression by 14 points. Naive Bayes shows the mirror pattern, 0.7952 on Amazon collapsing to 0.4838 on Twitter. The ranking reverses depending on which platform produces it. Only the fine-tuned model is stable across all three, with a spread of two points.

**Binary re-evaluation.** Removing the neutral class and forcing a choice between negative and positive from each model's own class scores gives the following.

**Table 4.5** — Effect of removing the neutral class, diagnostic sample *(source: `data/results/sentiment_diagnostics/binary_comparison.csv`)*

| Model | Three-class macro F1 | Binary macro F1 (forced) | Gain |
|---|---:|---:|---:|
| VADER | 0.5094 | 0.7974 | +0.2880 |
| RoBERTa (pretrained) | 0.6669 | **0.9341** | +0.2672 |
| Naive Bayes | 0.6959 | 0.8803 | +0.1844 |
| Logistic Regression | 0.7449 | 0.9216 | +0.1766 |
| DistilBERT (fine-tuned) | 0.8426 | 0.9533 | +0.1107 |

The ordering of the gains is itself the pattern: the more a model depends on task-specific training, the less it gains from removing neutral. RoBERTa's binary macro F1 of 0.9341 exceeds logistic regression's 0.9216, so once the class is removed the ranking implied by the literature is restored.

**VADER threshold sensitivity.** Widening the neutral band from the default ±0.05 raises macro F1 from 0.5094 to a maximum of 0.5353 at ±0.25, with neutral F1 rising from 0.253 to 0.349, after which macro F1 declines as the band swallows genuinely polar reviews. The full sweep is in Appendix B. The improvement is real but modest — 0.026 macro F1 at the optimum — so threshold choice explains part of VADER's weakness on the neutral class, not most of it.


**Qualitative error analysis.** The three experiments establish *where* errors concentrate but not what they look like, so individual misclassified reviews were inspected by joining the cached per-review predictions back to the source text (`evaluate/extract_error_examples.py`). Three patterns account for most failures, each corresponding to one of the quantitative findings above. Representative examples are reproduced verbatim in Appendix B.

*Pattern 1 — mixed evaluative content in three-star reviews.* Of the 600 true-neutral reviews in the balanced sample, 300 were assigned a polar label by RoBERTa while the fine-tuned model assigned neutral correctly. These are not uninformative texts but reviews carrying clear praise and clear criticism together, where a three-star rating encodes a net judgement across both clauses. This is the concrete form of the label-definition mismatch that Section 5.1.1 argues explains RoBERTa's neutral-class collapse.

*Pattern 2 — positive vocabulary in negative reviews.* VADER assigns positive compound scores to one-star reviews whose complaint is carried by temporal or concessive contrast rather than negative words. These errors sit confidently on the wrong side of the boundary rather than near it, which is why widening the neutral band recovers so little.

*Pattern 3 — the labelling ceiling.* A residual group is misclassified by all five models. Inspection shows reviews whose rating and text genuinely diverge, which is the ceiling imposed by rating-derived labelling rather than a modelling failure — the constraint recorded in Section 3.2.2 appearing in the data.

### 4.5.5 Taxonomy evaluation

Two distinct measures are used and should not be confused. *Centroid separation* is the mean pairwise cosine similarity **between** category centroids, where low is desirable. *Within-category coherence* is the mean similarity of reviews **to** their own centroid, where high is desirable. The implementation names the first `taxonomy_coherence`, which is misleading, and the distinction is drawn explicitly here so that the ambiguity is not carried into the results.

The merged taxonomy has a centroid separation of **0.2831** across its 24 categories, indicating well-separated categories. Had the 121 raw topics merged into 24 groups that still overlapped heavily, the consolidation would have been a relabelling exercise rather than a genuine reduction.

Applying the taxonomy to a held-out stratified sample of 450 reviews, 150 per platform, gives Table 4.6.

**Table 4.6** — Taxonomy evaluation on held-out data (n = 450) *(source: `data/results/category_discovery/evaluation_summary.csv`, `category_frequency.csv`)*

| Measure | Value |
|---|---|
| Categories used (of 24 available) | 21 |
| Mean within-category coherence | 0.6334 |
| Largest category | Product Performance Failures — 127 (28.2%) |
| Unclassified (below 0.35 confidence) | 124 (27.6%) |
| Categories with a single review | 3 |
| Average category size | 21.4 |

Two categories between them absorb 251 of 450 reviews (55.8%): the largest substantive category and the unclassified bucket. The remaining 199 reviews are distributed across 20 categories, three of which hold a single review.

Coherence varies systematically with category size, and the relationship is inverse; representative values are given in Appendix B.


The three categories holding a single review each score 1.0, which is an artefact of measuring a review's similarity to a centroid it alone defines, not a quality signal. These are excluded from interpretation.

Three defects are visible in the assignment output, and all three are reported rather than smoothed.

The largest category absorbs over a quarter of the sample while its internal coherence, 0.3997, is among the lowest recorded — the signature of a catch-all rather than a well-defined complaint type.

One category retained an un-renamed raw BERTopic label: "Spray Bottle, Continuous Spray, Dryer Diffuser, Fit Dryer". The language-model naming step described in Section 4.4.2 failed on at least one cluster, and the failure passed through into the deployed taxonomy.

The sentiment breakdown shows the taxonomy assigning positive reviews to complaint categories. That same mis-named category received 19 positive reviews and zero negative; the unclassified group holds 60 positive against 34 negative. The taxonomy was built from negative reviews only and has no mechanism for rejecting non-complaints, so any review presented to it is assigned somewhere or left unclassified regardless of whether it expresses a complaint at all.

![Figure 4.2](../data/results/category_discovery/category_frequency.png){width=4.6in}

**Figure 4.2** — Category frequency on the held-out sample (n = 450). Two categories account for 55.8% of assignments; three hold a single review each.


![Figure 4.3](../data/results/category_discovery/sentiment_by_category.png){width=4.6in}

**Figure 4.3** — Sentiment composition of each category. Several complaint categories contain predominantly positive reviews — the defect discussed above.


### 4.5.6 Retrieval evaluation

**Table 4.7** — Retrieval comparison, 30 queries over 642,692 indexed reviews *(source: `data/results/comparison_embedding_models/retrieval_comparison.csv`)*

| Metric | Semantic (MiniLM + FAISS) | TF-IDF |
|---|---:|---:|
| Mean retrieval time | 21.71 ms | 127.61 ms |
| Min / max retrieval time | 15.90 / 36.01 ms | 121.20 / 150.46 ms |
| Mean similarity score | 0.6833 | 0.5593 |
| Precision@3 | 0.90 | 0.90 |
| Query success rate | 100% | 100% |

Dense retrieval is roughly 5.9 times faster, and its latency is also more variable in relative terms while remaining well below the TF-IDF minimum. Precision@3 is identical at 0.90.

Two cautions apply. The similarity scores are not comparable across methods, since cosine similarity over dense embeddings and over sparse TF-IDF vectors measure different quantities; the higher dense figure should not be read as better retrieval. And with 30 queries the comparison is small enough that a modest precision difference could pass undetected. On this evidence the case for dense retrieval rests on latency, not on retrieval quality.


### 4.5.7 Generation evaluation

Table 4.8 compares the retrieval-augmented pipeline against the standalone baseline on the same five questions, with the same generator, temperature and prompt structure. These figures were regenerated after a hard grounding threshold (`SIMILARITY_THRESHOLD = 0.35`, `rag/pipeline.py`) was added to the retriever, which discards a question outright — returning a fixed refusal rather than an answer — when nothing retrieved clears the threshold. An earlier run of the same script, before that threshold existed, produced higher aggregate scores; those figures are superseded and are not reported here, since they describe a pipeline configuration that is no longer the one in `rag/pipeline.py`. Section 4.6 describes the guard; this section is the evaluation of the system with it in place. One caveat belongs here rather than buried in a footnote: as the per-question data below shows, the threshold itself was not what produced the one refusal observed in this sample — a separate part of the pipeline was responsible, and the difference matters enough to walk through properly rather than gloss over.

**Table 4.8** — RAG against standalone LLM (n = 5 questions) *(source: `data/results/llm_vs_rag/overall_results.csv`, `comparison_summary.csv`)*

| Metric | RAG pipeline | Standalone LLM |
|---|---:|---:|
| Faithfulness | 0.540 | not computable |
| Answer relevancy | 0.610 | **0.753** |
| Context precision | 0.416 | not applicable |
| Context recall | 0.400 | not applicable |
| Mean latency | 1,582.7 ms | 734.0 ms |
| Mean retrieved reviews | 3.4 | 0 |

Faithfulness cannot be computed for the standalone condition because the metric measures whether an answer follows from retrieved context, and no context exists. That absence is itself the finding: the baseline produces answers with no verifiable evidential basis, so the property the pipeline exists to provide is undefined for the alternative.

On the two metrics that can be compared, the standalone model scored higher on answer relevancy — 0.753 against 0.610 — while taking under half the time. Mean retrieved reviews falls to 3.4 because one question returns zero reviews (rejected before retrieval by the scope guard, not the grounding threshold — see the note under Table 4.9) and another returns only two, rather than the fixed five the earlier, un-gated retriever always returned.

**Per-question variance is severe, and the aggregate means conceal a pattern that reverses part of this reading.** Table 4.9 gives the full per-question breakdown.

**Table 4.9** — Per-question RAGAS scores (n = 5) *(source: `data/results/llm_vs_rag/rag_ragas_scores.csv`, `llm_ragas_scores.csv`)*

| Question | RAG faithfulness | RAG relevancy | RAG context precision | RAG context recall | LLM relevancy |
|---|---:|---:|---:|---:|---:|
| Why are customers unhappy with battery life? | 0.417 | 0.442 | 0.325 | 1.000 | 0.988 |
| What delivery problems are customers reporting? | 1.000 | 0.740 | 0.756 | 1.000 | **0.000** |
| Why are airline passengers dissatisfied? | 0.400 | n/a¹ | **0.000** | **0.000** | 0.961 |
| What customer service issues appear most often? | **0.000**² | 0.437 | **0.000**² | **0.000**² | 0.964 |
| Summarise the major customer complaints. | 0.882 | 0.819 | 1.000 | **0.000** | 0.853 |
| **Mean** | **0.540** | **0.610**¹ | **0.416** | **0.400** | **0.753** |

¹ The RAGAS judge raised an `OutputParserException` while scoring this question's answer relevancy — the free-tier judge model returned prose instead of the expected structured output — so no score exists for that cell. The reported mean is computed over the remaining four questions, not five, which narrows an already small sample further. This is a concrete instance of the judge-model risk discussed in Section 5.3.4.

² This one needs unpacking rather than a one-line note, because the obvious explanation turns out to be the wrong one. It's tempting to read this row as the grounding threshold doing its job — no evidence, so no answer — but the text returned doesn't say that. It matches `OUT_OF_SCOPE_MSG` verbatim, the message the pipeline's *scope guard* returns, not `NO_EVIDENCE_MSG`, the message the similarity threshold returns. That distinction matters: the scope guard runs first, before retrieval is even attempted, so this question never reached the grounding mechanism at all. Somewhere in the two-stage guard — most likely its LLM classifier, since the keyword list doesn't contain anything this question would trigger — "What customer service issues appear most often?" was judged out of scope. It plainly isn't; this is a legitimate complaint-exploration question of exactly the kind the system is built to answer. The same rejection, identical wording, recurred in both conditions of the metadata-tagging ablation described in Section 4.5.8, which rules out coincidence and points to a real, reproducible weak spot in the scope classifier rather than a one-off fluke. Faithfulness, context precision and context recall are mechanically zero here because no retrieval or generation ever happened for this question — not because the pipeline judged its own evidence insufficient.

Four observations follow directly from the table, and are recorded here as evidence rather than interpretation.

The standalone model's relevancy advantage is driven by the same single anomalous score as before. Its per-question relevancy is 0.988, 0.000, 0.961, 0.964 and 0.853 — four high scores and one complete failure on the delivery question. Excluding that single zero would raise its mean to approximately 0.941; including it gives 0.753. The RAG pipeline's four scored questions range from 0.437 to 0.819, a spread of 0.38 — wider than the earlier, un-gated run, but still narrower than the baseline's spread of 0.99.

The false rejection on the customer-service question is the most consequential single result in this table, but it credits the wrong mechanism if read carelessly. It is not the grounding threshold working as designed — no retrieval was even attempted — it's the scope guard misfiring on a question it should have passed straight through. The practical effect looks similar to a deliberate refusal (no answer, rather than a possibly wrong one), which limits the damage, but the cause is different, and Section 4.5.8's ablation results depend on getting this distinction right rather than folding it into the grounding-threshold story. It also mechanically depresses three of the four aggregate metrics for this question, since a rejection scores zero on faithfulness, context precision and context recall by construction, the same way a grounding refusal would.

Retrieval quality is still the pipeline's weakest measured component among the questions that actually reached it. Context precision fell to zero on two of five questions — airline and, by construction, the rejected customer-service question — and on the airline question, faithfulness fell with it, to 0.400, rather than remaining high as it did in the earlier, un-gated run, where a zero-precision question still produced an answer scored 0.917 on faithfulness. That's a genuine improvement: the pipeline no longer generates a confident, well-formed answer from evidence its own retrieval judged irrelevant. Whether the grounding threshold itself deserves the credit for that, though, is a separate question from whether the scope guard behaved correctly, and this sample doesn't actually contain a case of the threshold refusing a question with retrieved-but-irrelevant evidence — that's a gap worth flagging for a future, larger evaluation, not something this table can settle either way.

Context recall is now 0.000 on three of five questions and 1.000 on the remaining two, and the three zero cases have different causes that should not be collapsed into one. On the summarisation question it repeats the earlier finding — an open-ended request with no specific topic to retrieve against, a structural limitation of similarity-based retrieval rather than a tuning failure. On the airline question it reflects genuinely thin retrieval: only two reviews cleared the threshold, and neither covers the reference answer's specific topics (delays, cancellations, lost luggage). On the customer-service question it is the mechanical consequence of the scope guard's false rejection, not of anything retrieval did.

Three constraints qualify all of these figures. Sample size follows from free-tier API rate limits, which capped the available evaluation call volume; with n = 5 (n = 4 for relevancy) the means carry very wide uncertainty and no confidence interval is reported because none would be meaningful at this sample size. The judge model is the model that generated the answers, so the self-enhancement bias documented in Section 2.1.2 applies — though that bias would act to inflate the pipeline's scores rather than the baseline's, so it cannot explain the relevancy result. And the judge's own output failed to parse on one question, discussed above, which is itself evidence for that same judge-reliability concern rather than a separate issue.


**A note on evaluation scope.** Four ablation conditions were designed for the retrieval stage (Section 3.6.5). Three of them — retrieval depth, retriever type and prompt version — were specified but never built, so the contribution of those design decisions is still argued from reasoning rather than measured. The fourth, metadata tagging, was built and run, and Section 4.5.8 reports what it found.

### 4.5.8 Ablation: does metadata tagging actually help?

Section 3's design chapter argues that tagging each retrieved review with its platform, rating and sentiment before it reaches the generator (`DOCUMENT_PROMPT` in `rag/pipeline.py`) is what makes `RAG_PROMPT`'s grounding rules enforceable rather than aspirational. Until now that was a claim defended by reasoning, not evidence. `evaluate/ablation_metadata_tagging.py` tests it directly: the same five questions, run twice through the same pipeline, once with the tags left in place and once with them stripped so the generator sees only the raw review text. Retrieval is identical in both runs — same threshold, same MMR selection — so any difference in the RAGAS scores can only come from what the generator was shown, not from what was retrieved.

**Table 4.10** — Metadata tagging ablation (n = 5 questions) *(source: `data/results/ablation_metadata_tagging/comparison_summary.csv`)*

| Metric | Tagged | Untagged | Delta |
|---|---:|---:|---:|
| Faithfulness | 0.667 | 0.549 | **+0.118** |
| Answer relevancy | 0.452 | 0.654 | **−0.202** |
| Context precision | 0.416 | 0.416 | 0.000 |
| Context recall | 0.400 | 0.400 | 0.000 |

The clean part of the result first: context precision and context recall are identical, to three decimal places, between conditions. That's exactly what the design predicts, since neither condition touches retrieval, and it's reassuring in a small way — it confirms the ablation is actually isolating what it claims to isolate, rather than some confound in how the two runs were set up.

The interesting part is that tagging doesn't win cleanly. Faithfulness goes up with tags, by a solid margin — 0.667 against 0.549 — which does support the argument that citing each review's platform and rating gives the model something concrete to stay anchored to. But answer relevancy goes the other way, and by more: 0.452 tagged against 0.654 untagged, a drop of 0.202. That's not a rounding artefact; it's the largest single delta in the table. A plausible explanation is that the tagged answers spend more of their length on citation-style attribution — "(Yelp, Rating: 1.0, Sentiment: negative)" — while the untagged answers read more like a direct response to the question, which is closer to what an answer-relevancy judge is actually scoring. If that's right, the tagging design buys grounding at a real cost to how responsive the answer feels, and the honest conclusion is a trade-off rather than a validation.

One concrete example is worth quoting directly, because it's exactly the failure mode the tagging is meant to prevent, caught in the act. For the battery-life question, the untagged run's answer opens with "Platform(s) covered: Twitter Airline, Amazon" — but none of the five retrieved reviews for that question mention an airline; they're about product batteries and an in-store replacement service, plausibly Amazon and Yelp. The tagged run's answer for the same question and the same retrieved reviews says "Platform(s) covered: Yelp, Amazon," which matches what was actually retrieved. That's a real, observed instance of `RAG_PROMPT` rule 3 — never reference a platform absent from the retrieved reviews — being violated without the tags and respected with them. It's one example, not a pattern established across many questions, but it's the kind of concrete evidence the design argument in Section 3 didn't have before this ablation ran.

The latency figures (1,704.6 ms tagged against 501.3 ms untagged) are reported for completeness but shouldn't be read as an effect of tagging itself — the two conditions ran as separate sequential API calls to the same free-tier Groq endpoint, and a gap this large is more consistent with ordinary variance in queueing and network latency between calls than with tag text meaningfully changing generation time. No claim rests on it.

Taken together: this is the one piece of the planned ablation suite that got measured rather than argued, and it converts a two-line design justification into a genuine, if narrow, empirical finding — metadata tagging measurably improves faithfulness and produces at least one avoided platform hallucination, at a measurable cost to answer relevancy. The other three planned conditions (retrieval depth, retriever type, prompt version) remain unexecuted, and this single result should not be read as standing in for them.

![Figure 4.4](../data/results/ablation_metadata_tagging/metadata_tagging_ablation.png){width=4.6in}

**Figure 4.4** — Metadata-tagging ablation: tagged against untagged RAGAS scores across all four metrics. Context precision and recall are identical by construction; faithfulness and answer relevancy move in opposite directions. *(Produced by `evaluate/ablation_metadata_tagging.py`.)*

---

## 4.6 Deployment

The system was not deployed to production. What exists is a programmatic interface together with a local application layer built to make the components usable and demonstrable.

The `ask()` function in `rag/pipeline.py` accepts a question and optional conversation history, and returns the generated answer, the source reviews supporting it with their platform, rating, sentiment and review identifier, a retrieval count, and a flag indicating whether the answer is grounded. A similarity threshold and a two-stage scope guard reject out-of-domain questions and insufficient evidence before generation, and a post-generation check discards any answer produced without retrieved context. This is the form in which the system is used, and it is what the evaluation in Section 4.5.7 exercises.

A local FastAPI service and Streamlit interface expose these components for inspection. This layer is an engineering artefact supporting demonstration and manual checking; it is not a research result and no claim rests on it. Production deployment is identified as future work in Chapter 6.

---

## 4.7 Chapter Summary

The sentiment comparison produced a clear ranking, with the fine-tuned transformer highest at 0.8375 macro F1 on clean held-out data and the lexicon baseline lowest at 0.4381 on the shared test set. It also produced two results contradicting the expected ordering: a pretrained transformer 23 points of macro F1 below a linear model, and both off-the-shelf methods failing specifically on the neutral class while performing acceptably on the other two. McNemar's exact test found nine of ten pairwise differences significant on the diagnostic sample, the exception being Naive Bayes against RoBERTa at p = 0.82.

The diagnostic experiments established where the anomaly comes from. RoBERTa's macro F1 varies by 23 points across platforms and it outperforms both classical models on the Twitter data resembling its pretraining corpus, while removing the neutral class raises its binary macro F1 to 0.9341, above logistic regression's 0.9216. VADER's threshold sweep improved macro F1 by 0.026 at best, so decision-boundary choice accounts for part but not most of its weakness.

Complaint discovery produced 121 raw topics consolidated into 24 well-separated categories at 0.2831 centroid separation. Applied to held-out data the taxonomy used 21 of them, achieved 0.6334 mean within-category coherence, and left 27.6% of reviews unclassified. Two categories absorbed 55.8% of the sample between them. Three defects were recorded: a catch-all largest category with low internal coherence, a failed language-model naming step leaving a raw topic label in the taxonomy, and no mechanism for rejecting non-complaints, with positive reviews assigned to complaint categories.

Retrieval showed dense embeddings to be 5.9 times faster than TF-IDF at identical Precision@3 of 0.90. On five questions the retrieval-augmented pipeline, evaluated with a hard grounding threshold in place, achieved 0.540 faithfulness and 0.416 context precision, while the standalone baseline scored higher on mean answer relevancy, 0.753 against 0.610, despite having no evidence to be faithful to — an advantage that rests substantially on a single anomalous score in a set of five and is smaller in consistency terms, since the baseline's per-question relevancy spans 0.99 against the pipeline's 0.38. One question in this sample went unanswered, but tracing it back showed the grounding threshold wasn't actually responsible: the scope guard rejected it before retrieval ran, a false positive rather than the threshold correctly declining insufficient evidence. A separate, executed ablation on metadata tagging (Section 4.5.8) found a genuine trade-off rather than a clean win — tagging lifted faithfulness by 0.118 and cost 0.202 on answer relevancy, with one directly observed case of a platform hallucination it prevented.

Chapter 5 interprets these results against the four research questions and the literature reviewed in Chapter 2.
