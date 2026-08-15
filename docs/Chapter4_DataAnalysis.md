# Chapter 4 — Data Analysis

Chapter 3 set out a methodology built against four criticisms of the existing literature. This chapter reports what that methodology produced. Its purpose is to present the evidence in enough detail for the results to be reproduced and checked; the argument about what the evidence means is held back for Chapter 5. Where a result contradicts expectation, this chapter reports the contradiction and the diagnostic experiment that probes it, but leaves the explanation for the discussion.

Every figure quoted here is produced by a script that writes to `data/results/`, and the producing script is named alongside each table so that any number can be traced to its source.

The chapter follows the CRISP-DM phases adopted in Chapter 3. Section 4.1 restates the problem and defines the success criteria against which the work is judged. Section 4.2 reports the exploratory analysis of the three sources and the data quality issues it exposed. Section 4.3 covers preparation, feature extraction and splitting. Section 4.4 documents model development at each of the three pipeline stages. Section 4.5 reports evaluation, which supplies the evidence for all four research questions. Section 4.6 addresses deployment status, and Section 4.7 summarises.

---

## 4.1 Business Understanding

### 4.1.1 The problem

Organisations hold more customer feedback than they can read, and the practical question is not whether it contains useful signal but which method should be trusted to extract it. Three sub-problems follow, and they are analytically distinct. Classifying the sentiment of a review is supervised. Identifying what customers complain about is unsupervised, because the categories are not known in advance. Answering an analyst's question about those complaints is a generation problem, where the risk is a fluent answer with no basis in evidence.

### 4.1.2 Objectives for this chapter

This chapter delivers Objectives 2 to 6 from Chapter 1: preparing the unified corpus, comparing five sentiment classifiers with diagnosis of any anomalous result, discovering and consolidating complaint topics, building the retrieval and question-answering components with a baseline to compare against, and evaluating each stage with metrics appropriate to it.

### 4.1.3 Success criteria

Success is defined per stage, and deliberately not as a single accuracy target.

For sentiment classification, the criterion is a controlled comparison in which differences between models cannot be attributed to differences in data, reported at a granularity that reveals class-level and platform-level behaviour. A ranking alone would not satisfy it.

For complaint discovery, with no ground truth available, the criterion is a taxonomy whose categories are separable from one another, internally coherent, populated across platforms, and honest about what it fails to cover.

For question answering, the criterion is measurable grounding: answers traceable to identifiable source reviews, scored against a standalone baseline on the same questions.

---

## 4.2 Data Understanding

### 4.2.1 Sources

Three public datasets are combined: Amazon product reviews, Yelp business reviews, and the Twitter US Airline Sentiment corpus. All were obtained from their public distributions. No data was scraped and none was collected from individuals.

### 4.2.2 Register differences

The comparison in this study depends on the three sources differing in register, so that claim is measured rather than assumed. Figure 4.1 reports word counts from a seeded sample of up to 40,000 reviews per platform.

The differences are substantial and not in the direction intuition suggests. Yelp reviews are the longest by a wide margin, with a median of 85 words against Amazon's 29 and Twitter Airline's 19. Amazon reviews are shorter than expected at the median but carry a long tail: their 95th percentile reaches 138 words against Twitter's 27, so the platform mixes very brief and very detailed reviews. Twitter Airline is both the shortest and by far the most uniform, which follows from the platform's character limit — its interquartile range spans a narrower band than either of the others.

Two properties therefore separate the sources: central length, on which Yelp stands apart, and dispersion, on which Twitter stands apart. Amazon sits between them on both. This is the empirical basis for treating the corpus as multi-domain, and it is what makes the per-platform analysis in Section 4.5.4 meaningful.

*(Figure 4.1 — Review length by platform: distribution on a log scale, and median word count. Produced by `evaluate/generate_figures.py`; saved as `data/results/figures/fig_4_1_review_length_by_platform.png`.)*

### 4.2.3 Composition

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

### 4.2.4 Data quality issues

Exploratory analysis identified four issues, each carried forward into the interpretation of results.

Labels for Amazon and Yelp derive from star ratings, not from reading the text. A four-star review containing a specific complaint is labelled positive. This places a ceiling on achievable accuracy that no model can exceed.

Label provenance is not uniform. Twitter labels come from human annotation, so "neutral" means a three-star rating on two platforms and a human judgement of no clear sentiment on the third. The neutral class is therefore not one construct but two — a point that becomes important when the neutral class turns out to be where most models fail.

Text quality varies with register. Twitter posts contain handles, abbreviations and truncation absent from the other sources.

Some reviews are non-English. These are retained in the corpus but filtered out at the topic discovery stage, where mixed languages degrade clustering.

---

## 4.3 Data Preparation

### 4.3.1 Cleaning and unification

Each source was mapped onto a common schema of `review_id`, `platform`, `product_category`, `text`, `cleaned_text`, `rating`, `sentiment_label` and `date`. Records missing text or a sentiment label were dropped, as were reviews of five characters or fewer, which carry no usable signal. Text was lowercased and whitespace normalised to produce `cleaned_text`; the original `text` is retained because the retrieval stage benefits from readable output while models consume the cleaned form.

Sentiment labels for Amazon and Yelp were derived by rule: ratings of two or below negative, three neutral, four or above positive. Twitter's existing annotation was mapped to the same three classes and given a pseudo-rating so the schema stays uniform. Class balancing was applied during sampling, producing the distribution in Section 4.2.3.

### 4.3.2 Feature extraction

Feature representation differs by model family, which is itself part of the comparison.

Classical models use TF-IDF vectors over unigrams and bigrams. VADER requires no features, operating directly on raw text through its lexicon. Transformer models use their own subword tokenisers, with DistilBERT truncating at its tokeniser maximum. Topic discovery and retrieval use 384-dimensional sentence-transformer embeddings from `all-MiniLM-L6-v2`, L2-normalised so that inner product equals cosine similarity.

No manual feature selection or scaling was applied. TF-IDF weighting handles term importance implicitly, and embedding dimensions are not individually interpretable, so selecting among them would not be meaningful.

### 4.3.3 Splitting

The sentiment comparison uses an 80/20 stratified split with `random_state=42`, producing a held-out test set of **128,539 reviews** with the class distribution preserved.

The fine-tuned model was trained on a separately generated 70/15/15 stratified split, also seeded at 42. Its validation partition drove early stopping and checkpoint selection; its test partition of 88,854 reviews was held out entirely.

As disclosed in Section 3.5.3, these two partitions are not identical, so a portion of the 128,539-review test set overlaps data the fine-tuned model saw in training. The uncontaminated figure for that model is the one from its own held-out partition, and it is that figure which is treated as definitive. Every comparison in Section 4.5.1 involving DistilBERT is annotated accordingly.

---

## 4.4 Model Development

### 4.4.1 Sentiment classifiers

Five models span four families, selected on the evidence reviewed in Section 2.3.1.

VADER serves as the lexicon baseline, requiring no training, with compound scores mapped to three classes at the conventional ±0.05 thresholds. Naive Bayes and logistic regression are trained over TF-IDF features, the latter using the lbfgs solver with `max_iter=3000`, with five-fold stratified cross-validation during development. RoBERTa is used pretrained and unmodified in its `twitter-roberta-base-sentiment-latest` form; it is deliberately not fine-tuned, since its role is to represent what transfer learning delivers without task-specific training.

DistilBERT was fine-tuned from `distilbert-base-uncased` with a three-class head. Table 4.2 gives the configuration.

**Table 4.2** — DistilBERT fine-tuning configuration

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Base model | `distilbert-base-uncased` | Warmup steps | 1,000 |
| Split | 70/15/15, seed 42 | Weight decay | 0.01 |
| Epochs | 4 (early stopping, patience 2) | Precision | fp16 |
| Batch size | 32 train / 64 eval | Class weights | Balanced, weighted cross-entropy |
| Learning rate | 2e-5 | Selection metric | Validation macro F1 |

Two choices warrant note. Class weights were applied through a custom trainer overriding the loss function, because the standard trainer ignores them; without this the neutral class would be under-served despite corpus-level balance. And checkpoint selection used macro F1 rather than validation loss, so that the quantity optimised matches the quantity reported.

Table 4.3 records training behaviour.

**Table 4.3** — DistilBERT validation performance by epoch

| Epoch | Training loss | Validation loss | Accuracy | Weighted F1 | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2612 | 0.4482 | 83.48% | 83.51% | 82.59% |
| 2 | 0.2810 | 0.3977 | **84.08%** | **84.13%** | **83.28%** |
| 3 | 0.2355 | 0.4508 | 83.98% | 84.00% | 83.12% |
| 4 | 0.2007 | 0.5225 | 83.73% | 83.71% | 82.80% |

Validation loss fell to epoch 2 then rose while training loss continued to decline, the standard signature of overfitting. Best-checkpoint restoration selected epoch 2. Training took 10,545 seconds with a final training loss of 0.3350.

*(Figure 4.3 — DistilBERT training and validation loss with validation macro F1 across four epochs, marking the epoch-2 checkpoint selected by early stopping.)*

### 4.4.2 Complaint discovery

BERTopic was applied to the negative-sentiment reviews of each platform separately, after English-language filtering. Running per platform rather than over the pooled corpus prevents Yelp's seventy per cent share from dominating the topic space. Table 4.4 records configuration and outcome.

**Table 4.4** — BERTopic discovery by platform *(source: `data/results/category_discovery/discovery_platform_topics.csv`)*

| Platform | Documents clustered | Minimum topic size | Topics discovered |
|---|---:|---:|---:|
| Amazon | 57,931 | 86 | 69 |
| Yelp | 149,838 | 224 | 18 |
| Twitter Airline | 8,890 | 30 | 34 |
| **Total** | **216,659** | — | **121** |

Minimum topic size was derived from each platform's document count rather than fixed, which is why it varies. The topic counts are themselves informative: Yelp produced only 18 topics from 149,838 documents while Amazon produced 69 from roughly a third as many. Product categories span a far wider semantic range than restaurant and service complaints, and the discovery stage reflects that.

The 121 raw topics were then merged by agglomerative clustering over topic centroid embeddings to a target of 24 categories, each named by a language model given its constituent topic labels and keywords.

### 4.4.3 Category assignment

Individual reviews are assigned in two stages: an embedding comparison shortlists six candidate categories, then a zero-shot entailment model scores only that shortlist, using each category's full description as the hypothesis. A confidence threshold of 0.35 leaves weak matches unclassified instead of forcing a label, so that coverage failure remains visible in the results rather than being absorbed into a nearest-neighbour assignment.

### 4.4.4 Retrieval and question answering

The corpus was embedded with `all-MiniLM-L6-v2` and indexed in FAISS with normalised vectors carrying platform, rating, sentiment and review identifier as metadata. Retrieval returns five reviews per query from a candidate pool of twenty, using maximal marginal relevance.

Each retrieved review enters the prompt tagged with its platform, rating and sentiment. The prompt forbids any statistic, category or quotation not present in the retrieved text, restricts platform references to those appearing in the evidence, and specifies a fixed refusal string when evidence is insufficient. Source identifiers are returned with every answer. Generation uses Llama 3.1 8B via Groq at temperature 0.2, with multi-turn support through a history-aware retriever that reformulates follow-up questions into standalone queries before retrieval.

The same questions are answered by the model with retrieval disabled, providing the baseline for RQ2.

---

## 4.5 Model Evaluation

### 4.5.1 Aggregate sentiment results

Table 4.5 reports all five models on the shared test set.

**Table 4.5** — Sentiment classification results, shared held-out set (n = 128,539) *(source: `data/results/metrics_summary/overall_metrics.csv`)*

| Model | Family | Accuracy | Balanced acc. | Macro P | Macro R | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| VADER | Lexicon baseline | 0.5142 | 0.5148 | 0.5213 | 0.5148 | 0.4381 | 0.4396 |
| Naive Bayes | Classical ML | 0.7558 | 0.7557 | 0.7621 | 0.7557 | 0.7580 | 0.7579 |
| Logistic Regression | Classical ML | 0.8034 | 0.8033 | 0.8033 | 0.8033 | 0.8033 | 0.8033 |
| RoBERTa (pretrained) | Pretrained transformer | 0.6327 | 0.6314 | 0.6056 | 0.6314 | 0.5694 | 0.5709 |
| DistilBERT (fine-tuned)† | Fine-tuned transformer | **0.8243** | **0.8245** | **0.8247** | **0.8245** | **0.8244** | **0.8244** |

† Partially contaminated: a portion of this test set overlaps DistilBERT's training data (Section 4.3.3). The clean figure follows below.

Balanced accuracy tracks accuracy closely for every model, confirming that the class balancing described in Section 4.2.3 worked: no model is benefiting from an uneven class prior.

On its own uncontaminated held-out partition of 88,854 reviews, the fine-tuned model achieved accuracy 0.8437, macro F1 0.8375 and weighted F1 0.8454. This is the figure treated as definitive for DistilBERT throughout the remainder of this dissertation. Its closeness to the shared-set figure of 0.8244 suggests the contamination did not materially inflate the shared-set result, but the clean figure is used regardless, since the argument should not depend on that judgement.

*(Figure 4.4 — Overall model performance: accuracy with macro-averaged precision, recall and F1 for all five models.)*

Two results contradict the ordering the literature in Section 2.1.2 would predict.

The first is VADER's weakness. Its accuracy of 0.5142 places just over half of reviews in the right class, but its macro F1 of 0.4381 is far lower, and the gap between the two is the finding: performance is unevenly distributed across classes in a way that accuracy alone conceals. For reference, a uniform random classifier on this balanced three-class problem would score approximately 0.333 on both measures, so VADER is well above chance — the point is not that it fails outright but that it fails asymmetrically.

The second is more consequential. The pretrained transformer falls below both classical models, with a macro F1 of 0.5694 against logistic regression's 0.8033 — a gap of 23 points in the direction opposite to the one the literature predicts.

### 4.5.2 Per-class analysis

The aggregates conceal where the failure occurs. Table 4.6 reports per-class performance, and this is the table that makes the anomaly legible.

**Table 4.6** — Per-class precision, recall and F1 by model, shared test set (n = 128,539) *(source: `data/results/metrics_summary/per_class_metrics.csv`)*

| Model | Neg. P | Neg. R | Neg. F1 | Neu. P | Neu. R | Neu. F1 | Pos. P | Pos. R | Pos. F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VADER | 0.743 | 0.533 | 0.621 | 0.377 | **0.048** | **0.086** | 0.444 | 0.963 | 0.608 |
| Naive Bayes | 0.757 | 0.766 | 0.761 | 0.654 | 0.704 | 0.678 | 0.876 | 0.798 | 0.835 |
| Logistic Regression | 0.818 | 0.830 | 0.824 | 0.721 | 0.715 | 0.718 | 0.871 | 0.864 | 0.868 |
| RoBERTa (pretrained) | 0.684 | 0.814 | 0.743 | 0.522 | **0.143** | **0.225** | 0.612 | 0.937 | 0.740 |
| DistilBERT (fine-tuned) | 0.856 | 0.820 | 0.837 | 0.740 | 0.755 | 0.747 | 0.879 | 0.899 | 0.889 |

Both off-the-shelf models perform acceptably on negative and positive while collapsing on neutral, and separating precision from recall shows the nature of that collapse. VADER's neutral precision is 0.377 while its neutral recall is 0.048; RoBERTa's are 0.522 and 0.143. Precision is moderate in both cases and recall is near zero. Neither model is systematically wrong about what neutral text looks like when it does apply the label — both simply almost never apply it.

A second pattern appears in the positive column. VADER records 0.963 recall against 0.444 precision, so it captures nearly every genuinely positive review while being wrong about most of what it calls positive. This is the systematic positive skew visible in the confusion matrices.

*(Figure 4.2 — Row-normalised confusion matrices for all five models. Produced by `evaluate/generate_figures.py`; saved as `data/results/figures/fig_4_2_confusion_matrices.png`.)*

*(Figure 4.5 — Precision, recall and F1 by sentiment class, one panel per class.)*

Table 4.7 shows where the misclassified neutral reviews go, as row-normalised percentages.

**Table 4.7** — Destination of true-neutral reviews, row-normalised *(source: `data/results/cm_*.json`)*

| Model | → negative | → neutral | → positive |
|---|---:|---:|---:|
| VADER | 16.9% | 4.8% | 78.3% |
| RoBERTa (pretrained) | 36.2% | 14.3% | 49.5% |
| Logistic Regression | 17.3% | 71.5% | 11.2% |
| DistilBERT (fine-tuned) | 13.3% | 75.5% | 11.3% |

VADER assigns the neutral label to under five per cent of genuinely neutral reviews, pushing 78.3% to positive. RoBERTa splits them between both poles. Neither model fails uniformly; both behave as effectively binary classifiers confronted with a three-class task. VADER's skew extends further still: it labels 40.4% of genuinely negative reviews as positive.

### 4.5.3 Statistical significance of the differences

A difference in macro F1 between two models is only meaningful if it exceeds what sampling variation would produce. McNemar's exact test was therefore applied to every pair of models. The test considers only the reviews on which two models disagree: under the null hypothesis that both are equally accurate, each disagreement is equally likely to favour either model, so an exact binomial test on the discordant counts gives the probability of observing the imbalance found.

**An important scope note.** The test requires paired per-review predictions, which the saved confusion matrices do not retain. It was therefore computed on the balanced diagnostic sample of 1,800 reviews described in Section 4.5.4, not on the full test set. The accuracies in Table 4.8 are consequently those observed on that sample and differ from Table 4.5; the test speaks to pairwise correctness on the balanced sample, and is not a direct significance test of the macro F1 differences reported in Table 4.5.

**Table 4.8** — McNemar pairwise comparisons, balanced diagnostic sample (n = 1,800) *(source: `data/results/metrics_summary/significance_tests.csv`)*

| Model A | Model B | A acc. | B acc. | Discordant pairs | p-value | Significant | Better |
|---|---|---:|---:|---:|---:|---|---|
| VADER | DistilBERT | 0.5539 | 0.8417 | 734 | 1.6 × 10⁻⁸⁹ | yes | DistilBERT |
| VADER | Logistic Regression | 0.5539 | 0.7444 | 741 | 1.5 × 10⁻³⁷ | yes | Logistic Regression |
| Naive Bayes | DistilBERT | 0.6983 | 0.8417 | 444 | 2.6 × 10⁻³⁶ | yes | DistilBERT |
| VADER | RoBERTa | 0.5539 | 0.6944 | 429 | 3.2 × 10⁻³⁶ | yes | RoBERTa |
| RoBERTa | DistilBERT | 0.6944 | 0.8417 | 505 | 2.1 × 10⁻³³ | yes | DistilBERT |
| Logistic Regression | DistilBERT | 0.7444 | 0.8417 | 349 | 1.7 × 10⁻²¹ | yes | DistilBERT |
| VADER | Naive Bayes | 0.5539 | 0.6983 | 764 | 3.0 × 10⁻²¹ | yes | Naive Bayes |
| Naive Bayes | Logistic Regression | 0.6983 | 0.7444 | 295 | 1.6 × 10⁻⁶ | yes | Logistic Regression |
| Logistic Regression | RoBERTa | 0.7444 | 0.6944 | 596 | 2.6 × 10⁻⁴ | yes | Logistic Regression |
| **Naive Bayes** | **RoBERTa** | **0.6983** | **0.6944** | **675** | **0.82** | **no** | **—** |

Nine of the ten comparisons are significant at p < 0.05, most by very wide margins. Two deserve specific note.

The gap between DistilBERT and logistic regression is significant at p = 1.7 × 10⁻²¹, so on this sample the advantage of fine-tuning is not attributable to sampling variation.

Naive Bayes and RoBERTa, by contrast, cannot be separated. Of the 675 reviews on which they disagree, each is correct on almost exactly half, giving p = 0.82. This does not establish that the two models are equivalent — a non-significant result is not evidence of equality — but the available paired evidence does not support ranking either above the other on this sample, despite RoBERTa being a transformer and Naive Bayes a bag-of-words classifier from a much earlier generation of methods.

*(Figure 4.6 — Pairwise significance of model differences under McNemar's exact test, n = 1,800.)*

### 4.5.4 Diagnostic experiments

Three diagnostic experiments test the mechanism behind these results, using a balanced sample of 1,800 reviews drawn equally across platform and class so that no platform dominates. Because the sample is balanced rather than representative, the three-class figures in this section differ from Table 4.5 and should be compared only within the section.

**Per-platform performance.** Table 4.9 reports macro F1 and neutral F1 for every model on each platform.

**Table 4.9** — Macro F1 (neutral F1 in parentheses) by platform, 600 reviews per platform *(source: `data/results/sentiment_diagnostics/per_platform_metrics.csv`)*

| Model | Amazon | Yelp | Twitter Airline |
|---|---:|---:|---:|
| VADER | 0.4811 (0.152) | 0.4252 (0.019) | 0.5880 (0.480) |
| Naive Bayes | 0.7952 (0.728) | 0.7573 (0.693) | 0.4838 (0.154) |
| Logistic Regression | 0.7908 (0.724) | 0.7948 (0.721) | 0.6434 (0.509) |
| RoBERTa (pretrained) | 0.6218 (0.302) | 0.5544 (0.198) | **0.7853 (0.713)** |
| DistilBERT (fine-tuned) | 0.8448 (0.782) | 0.8301 (0.748) | 0.8505 (0.836) |

The result is decisive. RoBERTa scores 0.7853 on Twitter Airline against 0.5544 on Yelp, a spread of 23 points, and its neutral F1 moves from 0.713 on Twitter to 0.198 on Yelp. On Twitter it also **outperforms both classical models**, beating logistic regression by 14 points and Naive Bayes by 30. The ranking of methods reverses depending on which platform is used to produce it.

Naive Bayes shows the mirror-image pattern, scoring 0.7952 on Amazon but collapsing to 0.4838 on Twitter. Only the fine-tuned model is stable across all three platforms, with a spread of two points.

*(Figure 4.7 — Per-platform macro F1 by model, grouped bars, n = 600 per platform.)*

**Binary re-evaluation.** Table 4.10 removes the neutral class. The forced-binary column requires each model to choose between negative and positive using its own class scores.

**Table 4.10** — Effect of removing the neutral class, diagnostic sample *(source: `data/results/sentiment_diagnostics/binary_comparison.csv`)*

| Model | Three-class macro F1 | Binary macro F1 (forced) | Gain |
|---|---:|---:|---:|
| VADER | 0.5094 | 0.7974 | +0.2880 |
| RoBERTa (pretrained) | 0.6669 | **0.9341** | +0.2672 |
| Naive Bayes | 0.6959 | 0.8803 | +0.1844 |
| Logistic Regression | 0.7449 | 0.9216 | +0.1766 |
| DistilBERT (fine-tuned) | 0.8426 | 0.9533 | +0.1107 |

The two off-the-shelf models gain most from removing neutral, and the ordering of the gains is itself the pattern: the more a model depends on task-specific training, the less it gains. RoBERTa's binary macro F1 of 0.9341 exceeds logistic regression's 0.9216, so once the neutral class is removed the ranking implied by the literature is restored.

*(Figure 4.8 — Three-class against forced-binary macro F1 by model.)*

**VADER threshold sensitivity.** Widening VADER's neutral band from the default ±0.05 raises macro F1 from 0.5094 to a maximum of 0.5353 at ±0.25, with neutral F1 rising from 0.253 to 0.349 and the proportion of reviews predicted neutral rising from 10.2% to 18.9%. Beyond ±0.25 macro F1 declines again. Table 4.11 gives the sweep.

**Table 4.11** — VADER neutral-band sweep, diagnostic sample *(source: `data/results/sentiment_diagnostics/vader_threshold_sweep.csv`)*

| Neutral band | Accuracy | Macro F1 | Neutral F1 | Predicted neutral (%) |
|---:|---:|---:|---:|---:|
| ±0.05 (default) | 0.5539 | 0.5094 | 0.2526 | 10.2 |
| ±0.10 | 0.5567 | 0.5179 | 0.2812 | 12.1 |
| ±0.15 | 0.5594 | 0.5258 | 0.3075 | 14.0 |
| ±0.20 | **0.5628** | 0.5332 | 0.3311 | 16.3 |
| ±0.25 | 0.5606 | **0.5353** | 0.3486 | 18.9 |
| ±0.30 | 0.5561 | 0.5335 | 0.3638 | 21.9 |
| ±0.40 | 0.5478 | 0.5297 | 0.3909 | 27.5 |
| ±0.50 | 0.5406 | 0.5259 | 0.4413 | 37.2 |
| ±0.60 | 0.5272 | 0.5099 | 0.4649 | 44.3 |

The improvement is real but modest: a gain of 0.026 macro F1 at the optimum. Threshold choice explains part of VADER's weakness on the neutral class, not most of it.

*(Figure 4.9 — VADER macro F1, neutral F1 and accuracy across neutral-band widths, marking the ±0.25 optimum. All three diagnostic figures produced by `evaluate/evaluate_sentiment_diagnostics.py`.)*

### 4.5.5 Taxonomy evaluation

Two distinct measures are used and should not be confused. *Centroid separation* is the mean pairwise cosine similarity **between** category centroids, where low is desirable. *Within-category coherence* is the mean similarity of reviews **to** their own centroid, where high is desirable. The implementation names the first `taxonomy_coherence`, which is misleading, and the distinction is drawn explicitly here so that the ambiguity is not carried into the results.

The merged taxonomy has a centroid separation of **0.2831** across its 24 categories, indicating well-separated categories. Had the 121 raw topics merged into 24 groups that still overlapped heavily, the consolidation would have been a relabelling exercise rather than a genuine reduction.

Applying the taxonomy to a held-out stratified sample of 450 reviews, 150 per platform, gives Table 4.12.

**Table 4.12** — Taxonomy evaluation on held-out data (n = 450) *(source: `data/results/category_discovery/evaluation_summary.csv`, `category_frequency.csv`)*

| Measure | Value |
|---|---|
| Categories used (of 24 available) | 21 |
| Mean within-category coherence | 0.6334 |
| Largest category | Product Performance Failures — 127 (28.2%) |
| Unclassified (below 0.35 confidence) | 124 (27.6%) |
| Categories with a single review | 3 |
| Average category size | 21.4 |

Two categories between them absorb 251 of 450 reviews (55.8%): the largest substantive category and the unclassified bucket. The remaining 199 reviews are distributed across 20 categories, three of which hold a single review.

Coherence varies systematically with category size, and the relationship is inverse. Table 4.13 gives representative values.

**Table 4.13** — Within-category coherence against category size, selected categories *(source: `data/results/category_discovery/category_coherence.csv`)*

| Category | Reviews | Mean coherence |
|---|---:|---:|
| Travel Disruption Issues | 11 | 0.6763 |
| Home Network Connectivity Issues | 7 | 0.6410 |
| Service and Wait Time Delays | 35 | 0.5678 |
| Audio Performance Failures | 22 | 0.5546 |
| Product Fit and Sizing Issues | 25 | 0.4079 |
| Product Performance Failures | 127 | 0.3997 |
| Unclassified / Emerging Complaint | 124 | 0.3767 |

The three categories holding a single review each score 1.0, which is an artefact of measuring a review's similarity to a centroid it alone defines, not a quality signal. These are excluded from interpretation.

Three defects are visible in the assignment output, and all three are reported rather than smoothed.

The largest category absorbs over a quarter of the sample while its internal coherence, 0.3997, is among the lowest recorded — the signature of a catch-all rather than a well-defined complaint type.

One category retained an un-renamed raw BERTopic label: "Spray Bottle, Continuous Spray, Dryer Diffuser, Fit Dryer". The language-model naming step described in Section 4.4.2 failed on at least one cluster, and the failure passed through into the deployed taxonomy.

The sentiment breakdown shows the taxonomy assigning positive reviews to complaint categories. That same mis-named category received 19 positive reviews and zero negative; the unclassified group holds 60 positive against 34 negative. The taxonomy was built from negative reviews only and has no mechanism for rejecting non-complaints, so any review presented to it is assigned somewhere or left unclassified regardless of whether it expresses a complaint at all.

*(Figure 4.10 — Category frequency distribution. Figure 4.11 — Platform distribution by category. Figure 4.12 — Sentiment distribution by category, showing positive reviews assigned to complaint categories. Figure 4.13 — Mean within-category coherence against category size. All produced by `evaluate/evaluate_category_discovery.py`.)*

### 4.5.6 Retrieval evaluation

**Table 4.14** — Retrieval comparison, 30 queries over 642,692 indexed reviews *(source: `data/results/comparison_embedding_models/retrieval_comparison.csv`)*

| Metric | Semantic (MiniLM + FAISS) | TF-IDF |
|---|---:|---:|
| Mean retrieval time | 21.71 ms | 127.61 ms |
| Min / max retrieval time | 15.90 / 36.01 ms | 121.20 / 150.46 ms |
| Mean similarity score | 0.6833 | 0.5593 |
| Precision@3 | 0.90 | 0.90 |
| Query success rate | 100% | 100% |

Dense retrieval is roughly 5.9 times faster, and its latency is also more variable in relative terms while remaining well below the TF-IDF minimum. Precision@3 is identical at 0.90.

Two cautions apply. The similarity scores are not comparable across methods, since cosine similarity over dense embeddings and over sparse TF-IDF vectors measure different quantities; the higher dense figure should not be read as better retrieval. And with 30 queries the comparison is small enough that a modest precision difference could pass undetected. On this evidence the case for dense retrieval rests on latency, not on retrieval quality.

*(Figure 4.14 — Retrieval latency and Precision@3, semantic against TF-IDF, n = 30 queries.)*

### 4.5.7 Generation evaluation

Table 4.15 compares the retrieval-augmented pipeline against the standalone baseline on the same five questions, with the same generator, temperature and prompt structure.

**Table 4.15** — RAG against standalone LLM (n = 5 questions) *(source: `data/results/llm_vs_rag/overall_results.csv`, `comparison_summary.csv`)*

| Metric | RAG pipeline | Standalone LLM |
|---|---:|---:|
| Faithfulness | 0.849 | not computable |
| Answer relevancy | 0.671 | **0.751** |
| Context precision | 0.503 | not applicable |
| Context recall | 0.800 | not applicable |
| Mean latency | 2,022.6 ms | 669.6 ms |
| Mean retrieved reviews | 5.0 | 0 |

Faithfulness cannot be computed for the standalone condition because the metric measures whether an answer follows from retrieved context, and no context exists. That absence is itself the finding: the baseline produces answers with no verifiable evidential basis, so the property the pipeline exists to provide is undefined for the alternative.

On the two metrics that can be compared, the standalone model scored higher on answer relevancy — 0.751 against 0.671 — while taking roughly a third of the time. Unconstrained generation produced answers judged more responsive while resting on nothing checkable.

**Per-question variance is severe, and the aggregate means conceal a pattern that reverses part of this reading.** Table 4.16 gives the full per-question breakdown.

**Table 4.16** — Per-question RAGAS scores (n = 5) *(source: `data/results/llm_vs_rag/rag_ragas_scores.csv`, `llm_ragas_scores.csv`)*

| Question | RAG faithfulness | RAG relevancy | RAG context precision | RAG context recall | LLM relevancy |
|---|---:|---:|---:|---:|---:|
| Why are customers unhappy with battery life? | 1.000 | 0.743 | 0.250 | 1.000 | 0.988 |
| What delivery problems are customers reporting? | 0.917 | 0.728 | 0.867 | 1.000 | **0.000** |
| Why are airline passengers dissatisfied? | 0.917 | 0.579 | **0.000** | 1.000 | 0.961 |
| What customer service issues appear most often? | 0.625 | 0.570 | 0.700 | 1.000 | 0.952 |
| Summarise the major customer complaints. | 0.786 | 0.737 | 0.700 | **0.000** | 0.853 |
| **Mean** | **0.849** | **0.671** | **0.503** | **0.800** | **0.751** |

Three observations follow directly from the table, and are recorded here as evidence rather than interpretation.

The standalone model's relevancy advantage rests entirely on one anomalous score. Its per-question relevancy is 0.988, 0.000, 0.961, 0.952 and 0.853 — four high scores and one complete failure on the delivery question. Excluding that single zero would raise its mean to approximately 0.938; including it gives 0.751. The RAG pipeline's relevancy, by contrast, ranges only from 0.570 to 0.743, a spread of 0.17 against the baseline's 0.99.

Retrieval quality is the pipeline's weakest measured component. Context precision ranges from 0.000 to 0.867 with a mean of 0.503, and one question retrieved no relevant context at all by this measure while still producing an answer scored 0.917 on faithfulness and 0.579 on relevancy.

Context recall is 1.000 on four of five questions and 0.000 on the fifth, the open-ended summarisation question — the only question in the set without a specific topic to retrieve against.

Two constraints qualify all of these figures. Sample size follows from free-tier API rate limits, which capped the available evaluation call volume; with n = 5 the means carry very wide uncertainty and no confidence interval is reported because none would be meaningful at this sample size. And the model judging the answers is the model that generated them, so the self-enhancement bias documented in Section 2.1.2 applies — though that bias would act to inflate the pipeline's scores rather than the baseline's, so it cannot explain the relevancy result.

*(Figure 4.15 — RAG against standalone LLM: answer relevancy per question, and mean latency. Produced by `evaluate/evaluate_llm_vs_rag.py`.)*

**A note on evaluation scope.** The ablation suite covering retrieval depth, retriever type, prompt version and metadata tagging was implemented but not executed, for the reasons given in Section 3.6.6. No ablation results are reported anywhere in this chapter, and the contribution of individual retrieval design decisions is therefore argued in Chapter 3 from design reasoning rather than measured here.

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

Retrieval showed dense embeddings to be 5.9 times faster than TF-IDF at identical Precision@3 of 0.90. On five questions the retrieval-augmented pipeline achieved 0.849 faithfulness and 0.503 context precision, while the standalone baseline scored higher on mean answer relevancy, 0.751 against 0.671, despite having no evidence to be faithful to — an advantage that rests entirely on a single anomalous score in a set of five and disappears in consistency terms, since the baseline's per-question relevancy spans 0.99 against the pipeline's 0.17.

Chapter 5 interprets these results against the four research questions and the literature reviewed in Chapter 2.
