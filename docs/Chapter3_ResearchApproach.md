# Chapter 3 — Research Approach

Chapter 2 ended with a gap and four specific complaints about how the existing literature is evaluated: comparisons run within one domain, aggregate metrics that hide class-level failure, topic models validated by a single contested coherence score, and retrieval-augmented systems adopted without being measured against the baseline they are meant to improve on. This chapter sets out a methodology designed against those four complaints rather than around them.

Its purpose is therefore twofold. It must make the work reproducible — another researcher should be able to repeat these experiments and obtain these numbers — and it must justify each decision against the limitation it responds to, so that the results in Chapter 4 can be read with the right qualifications already attached. Where a decision was constrained rather than optimal, that is stated at the point the decision is described, not deferred to a limitations section at the end.

The chapter proceeds in the order the work was actually carried out. Section 3.1 selects and justifies the research methodology and maps its phases onto the objectives. Section 3.2 describes the datasets and what they can and cannot support. Section 3.3 sets out the workflow that connects the stages. Section 3.4 covers tools and technologies. Section 3.5 describes preprocessing and, critically, splitting — the point at which the comparison is either controlled or not. Section 3.6 covers model development at each stage, with hyperparameters, evaluation metrics and an explicit statement of what was implemented but not executed. Section 3.7 addresses ethics, and Section 3.8 summarises.

---

## 3.1 Research Methodology

### 3.1.1 Selecting a methodology

Several methodologies are established for data science work, and the choice between them turns on where a project's contribution sits. CRISP-DM remains the most widely used process model, structuring work as business understanding, data understanding, data preparation, modelling, evaluation and deployment. Design Science Research is common where the output is an artefact, and organises work around building and evaluating that artefact. KDD focuses more narrowly on the sequence from raw data to extracted knowledge. Agile approaches are sometimes adapted for exploratory analytics.

This study uses **CRISP-DM, adapted for a comparative experimental design**. The reasoning follows from what this project is. Design Science suits work whose contribution is the artefact itself; here the pipeline is the instrument and the comparison is the contribution, so a methodology organised around artefact construction would misplace the emphasis and invite the writing-up to describe a system rather than evaluate a question. KDD stops short of the evaluation depth this study needs. Agile addresses project management rather than analytical validity, which is the thing at risk here.

CRISP-DM fits for two reasons that bear directly on Chapter 2's criticisms. Its evaluation phase is explicit and iterative, which suits a study whose contribution lies in evaluation rather than modelling. And its data preparation phase is treated as substantial work rather than a preliminary, which matters because preparation is precisely where a comparison becomes controlled or fails to. The adaptation is that the modelling and evaluation phases execute three times, once per pipeline stage, since each stage has its own candidate methods and its own appropriate metrics — a direct consequence of the argument in Chapter 1 that feedback analysis is not one task but three. Deployment is out of scope and is treated as future work in Chapter 6.

### 3.1.2 Applying the methodology

**Business understanding.** Established in Chapter 1: organisations need to know not only how much dissatisfaction exists but what it concerns, and need to be able to check any conclusion against evidence. This phase produced the four research questions.

**Data understanding.** Exploratory analysis of the three sources, examining review length, class balance, platform composition and text quality. This phase identified the register differences between platforms that the whole comparison depends on, and surfaced the labelling limitation discussed in Section 3.2.3.

**Data preparation.** Cleaning, schema unification, sentiment labelling, class balancing and splitting. Described in Section 3.5.

**Modelling.** Executed three times: five sentiment classifiers, BERTopic discovery with taxonomy consolidation, and the retrieval-augmented pipeline with its standalone baseline. Described in Section 3.6.

**Evaluation.** Also executed three times, with metrics matched to what each stage permits, plus diagnostic experiments that test the mechanism behind any unexpected result rather than leaving it as speculation. Described in Section 3.6.5.

Table 3.1 maps the phases onto the objectives from Chapter 1, so that every phase of work traces to a stated objective and through it to a research question.

**Table 3.1** — Mapping of CRISP-DM phases to research objectives and questions

| CRISP-DM phase | Objective | Research question | Output |
|---|---|---|---|
| Business understanding | O1 | RQ1–RQ4 | Literature review, research gap |
| Data understanding | O2 | — | Exploratory analysis, dataset profile |
| Data preparation | O2 | — | Unified corpus, documented split |
| Modelling (stage 1) | O3 | RQ1 | Five trained or configured classifiers |
| Modelling (stage 2) | O4 | RQ3 | Per-platform topics, merged taxonomy |
| Modelling (stage 3) | O5 | RQ2, RQ4 | RAG pipeline and standalone baseline |
| Evaluation | O6 | RQ1–RQ4 | Metrics, baselines, diagnostic experiments |
| Conclusion | O7 | RQ1–RQ4 | Critical discussion, limitations, future work |

With the process settled, the next question is what data it operates on — and since Chapter 2's first criticism was that single-domain corpora cannot support cross-domain claims, the composition of the corpus is itself a methodological decision rather than a given.

---

## 3.2 Data Collection

### 3.2.1 Datasets

Three publicly available datasets are used, chosen so that the corpus spans genuinely different registers rather than three variations of the same thing. This is the direct response to the dataset criticism in Section 2.1.3: a study that wants to say something about domain transfer must contain domains that actually differ.

**Amazon product reviews** cover physical products, where complaints concern quality, delivery and value. They vary widely in length, from one-line verdicts to detailed accounts of a product's performance over time. **Yelp reviews** describe service and hospitality experiences, where complaints concern staff, waiting and atmosphere; these are the longest reviews in the corpus, typically running to extended narrative. **Twitter US Airline Sentiment** provides short, abbreviated, often public-facing complaints directed at a company, where the issues are delays, cancellations and baggage.

The three therefore differ on register, length and communicative purpose simultaneously — a reviewer writing for other shoppers, a diner writing a narrative account, and a customer publicly demanding a response are doing different things with language. All three were obtained from their public distributions; no data was scraped and no new data was collected.

### 3.2.2 Dataset characteristics

After preprocessing, the unified corpus contains **642,692 reviews**. Table 3.2 gives the composition.

**Table 3.2** — Composition of the unified corpus

| Platform | Reviews | Share | Label origin | Median words |
|---|---:|---:|---|---:|
| Yelp | 449,858 | 70.0% | Star rating | 85 |
| Amazon | 178,883 | 27.8% | Star rating | 29 |
| Twitter Airline | 13,951 | 2.2% | Human annotation | 19 |
| **Total** | **642,692** | **100%** | | |

The median word counts confirm that the register variation is real and not merely assumed: Yelp reviews are roughly four and a half times longer than airline tweets. That spread is what makes the per-platform comparison in Chapter 4 informative, and it is also why a method that performs well on one platform should not be expected to transfer without evidence.

The sentiment distribution is close to balanced by design: negative 218,799, neutral 212,503, positive 211,390. Balancing was applied during sampling so that no class dominates, which matters because macro F1 is the headline metric and an imbalanced corpus would make the neutral class both rare and hard — conflating two separate difficulties and obscuring which one a model was failing on.

Each record carries: `review_id`, `platform`, `product_category`, `text`, `cleaned_text`, `rating`, `sentiment_label` and `date`.

### 3.2.3 Justification and limitations

The three sources were chosen because they differ on the dimension the study cares about. A corpus of three product-review sources would not test transfer across register; these three do.

Four limitations follow, and each is carried into the interpretation of results rather than parked here.

Labels are derived from star ratings for Amazon and Yelp, with ratings of two or below treated as negative, three as neutral, and four or above as positive. This is standard practice at this scale — it is what makes corpora of this size possible at all — but it is an approximation: a four-star review can contain a specific complaint, and it will be labelled positive. This places a ceiling on the accuracy any classifier in this study can reach, and it is the most likely source of difficulty on the neutral class, since a three-star review is the least semantically coherent of the three groups.

Label provenance is not homogeneous. Twitter labels come from human annotation, so the corpus mixes two labelling processes. This is a genuine threat to comparability across platforms and is reported as such wherever per-platform results are interpreted.

Platform representation is uneven. Twitter contributes only 2.2% of the corpus, so per-platform analysis on that subset rests on fewer examples and is read with more caution. The diagnostic experiments in Section 3.6.5 address this directly by using a platform-balanced sample rather than the natural distribution.

Review data carries selection bias. People with strong opinions are more likely to write reviews, so the sentiment distribution reflects who chooses to write rather than the customer base as a whole. No claim about population-level customer sentiment is made anywhere in this dissertation.

---

## 3.3 Research Process

Having established what the data is and what it will support, the workflow sets out how the stages connect. It runs in five stages, each evaluated before the next depends on it.

1. **Data preparation** — three raw sources are cleaned, mapped to a common schema, labelled and balanced, producing `reviews_unified.parquet`.
2. **Sentiment classification** — five models are trained or configured and evaluated on a shared held-out split.
3. **Complaint discovery** — BERTopic runs on the negative reviews of each platform separately; the resulting topics are merged into one cross-platform taxonomy and named.
4. **Retrieval and question answering** — the corpus is embedded and indexed in FAISS; questions are answered both with retrieval and without it, for comparison.
5. **Evaluation** — each stage is scored with metrics appropriate to it, with baseline comparisons and diagnostic experiments that test the cause of unexpected results.

*(Figure 3.1 — Research workflow diagram. A block diagram showing the five stages above as a left-to-right pipeline, with the three raw datasets feeding stage 1, and evaluation outputs branching from stages 2, 3 and 4. To be produced and inserted here.)*

Stage ordering is not arbitrary, and the dependencies are worth making explicit because they propagate. Sentiment classification precedes complaint discovery because discovery operates on negative reviews only, so the quality of the sentiment stage constrains what the discovery stage receives: a review misclassified as negative enters the topic model as noise. Discovery precedes retrieval because the taxonomy provides the vocabulary in which complaints are described. Each stage therefore inherits the limitations of the one before it, which is itself a finding worth reporting rather than a design flaw to conceal — and Chapter 5 returns to it when interpreting the taxonomy's coverage.

---

## 3.4 Tools and Technologies

The selections below follow the justifications given in Section 2.3; what is added here is the configuration detail needed to repeat the work.

**Language and core libraries.** Python 3, with pandas for data handling and scikit-learn for classical models, splitting and classification metrics. Both are standard and well documented, which supports reproducibility. `[AUTHOR TO ADD: exact Python version used, e.g. 3.10.x]`

**Transformers.** Hugging Face Transformers provides RoBERTa and DistilBERT along with the `Trainer` API used for fine-tuning. It was chosen for its model hub, which makes the exact checkpoints used identifiable by name rather than by description.

**Sentiment baseline.** `vaderSentiment` provides the lexicon baseline without training.

**Topic modelling.** BERTopic, with UMAP for dimensionality reduction and HDBSCAN for clustering. These are BERTopic's intended components and are configured explicitly rather than left at defaults, so that the settings are reportable and the results attributable to choices rather than to library conventions.

**Embeddings and retrieval.** `sentence-transformers` supplies `all-MiniLM-L6-v2` (384 dimensions), and FAISS provides the vector index. FAISS runs locally with no external service, which removes both cost and a reproducibility dependency — an index that depends on a hosted service cannot be reconstructed by a reader once that service changes.

**Generation.** Groq's hosted Llama 3.1 (`llama-3.1-8b-instant`) at temperature 0.2, orchestrated with LangChain. Low temperature is chosen deliberately: the task is evidence-grounded answering, where output variability is a defect rather than a feature, and where a reproducible evaluation requires responses that do not swing between runs.

**Evaluation and tracking.** RAGAS for generation metrics, scikit-learn for classification metrics, MLflow for experiment tracking. MLflow preserves parameters, metrics and artefacts per run, so an individual result can be traced back to the configuration that produced it.

Every dependency is version-pinned in `requirements.txt`, random seeds are fixed at 42 throughout, and unit tests cover the topic-merging and category-shortlisting logic. `[AUTHOR TO ADD: hardware used for DistilBERT fine-tuning — GPU model and memory, or Colab tier; and the public repository URL if the code is to be released]`

---

## 3.5 Data Processing

### 3.5.1 Cleaning and unification

Each source arrives with its own schema and is mapped onto the common structure given in Section 3.2.2. Cleaning removes records with missing text or labels, normalises text case and whitespace, and produces `cleaned_text` alongside the original `text`. Both are retained deliberately: models consume the cleaned version, while the retrieval stage serves the original, since an analyst reading a retrieved review as evidence needs it in its readable form.

For Amazon and Yelp, `sentiment_label` is derived from the star rating using the rule in Section 3.2.3. For Twitter, the existing `airline_sentiment` annotation is used and mapped to a pseudo-rating so that the schema stays uniform across sources.

Class balancing is applied during sampling, producing the near-equal distribution reported in Section 3.2.2.

### 3.5.2 Language filtering

The complaint discovery stage additionally filters to English text using language detection, since topic clustering degrades when multiple languages are mixed — non-English documents cluster by language rather than by subject, producing topics that describe the corpus's composition instead of its complaints. This removes a small proportion of each platform's negative reviews.

### 3.5.3 Data splitting

Splitting is where a comparative study is either controlled or not, so it is stated precisely rather than summarised.

For the sentiment comparison, the corpus is split 80/20 with stratification on the sentiment label and `random_state=42`, giving a held-out test set of **128,539 reviews**. Stratification keeps the class distribution identical across both partitions, so that a model cannot gain or lose on the aggregate simply because one partition happened to contain more of the class it handles best.

The fine-tuned DistilBERT model was trained using a 70/15/15 stratified split, also seeded at 42, giving train, validation and test partitions. The validation partition is used for early stopping and best-checkpoint selection; the test partition is held out entirely.

**A limitation must be recorded here, and it is the most consequential methodological caveat in this dissertation.** The fine-tuned model and the other four models were developed against separately generated corpus versions. The two partitions are therefore not identical, and a proportion of the sentiment comparison's test set overlaps with data the fine-tuned model saw during training. The comparison is consequently controlled for four of the five models and not for the fifth.

Three things follow. The clean, uncontaminated figure for the fine-tuned model is the one obtained on its own held-out test partition, and it is that figure which is interpreted in Chapter 5. Any statement comparing the fine-tuned model directly against the others on the shared test set is qualified at the point it is made. And establishing a single canonical split, regenerated once and reused by every model, is identified in Chapter 6 as the first correction future work should make.

Stating this openly is preferable to presenting a comparison whose controls do not hold. It also has an analytical consequence worth noting: the fine-tuned model's advantage over logistic regression on the shared split should be read as an upper bound rather than a point estimate.

---

## 3.6 Model Development

### 3.6.1 Sentiment classification

Five models span four method families, chosen so that each represents a distinct answer to the same problem rather than a variation on one answer.

**VADER** requires no training. Its compound score is mapped to three classes using the conventional thresholds: at or above 0.05 positive, at or below −0.05 negative, otherwise neutral. Retaining the conventional thresholds matters for comparability, and the threshold sweep described in Section 3.6.5 tests whether that convention is what limits its performance.

**Naive Bayes** and **logistic regression** operate over TF-IDF features with unigrams and bigrams. Logistic regression uses the lbfgs solver with `max_iter=3000`. Both are trained on an 80/20 stratified split with five-fold stratified cross-validation used during development. Tuning these properly is a methodological commitment rather than routine practice: Section 2.1.3 criticised published comparisons for pairing carefully fine-tuned transformers against under-tuned classical baselines, and a study repeating that asymmetry would reproduce the bias it set out to test.

**RoBERTa** is used pretrained and unmodified, in the `twitter-roberta-base-sentiment-latest` configuration, with inputs truncated at 512 tokens. It is deliberately not fine-tuned. Its role is to represent what transfer learning achieves *without* task-specific training, which is the condition a practitioner faces when adopting an off-the-shelf model — and, given that this checkpoint was pretrained on tweets (Barbieri et al., 2020), it also makes the domain-transfer question in Section 2.1.2 directly testable on three platforms of differing register.

**DistilBERT** is fine-tuned from `distilbert-base-uncased` with a three-class head. Table 3.3 gives the configuration.

**Table 3.3** — Fine-tuning configuration for DistilBERT

| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Split | 70 / 15 / 15 stratified, seed 42 |
| Epochs | 4, early stopping patience 2 |
| Batch size | 32 train / 64 evaluation |
| Learning rate | 2e-5 |
| Warmup steps | 1,000 |
| Weight decay | 0.01 |
| Precision | fp16 mixed precision |
| Class imbalance | Balanced class weights in a weighted cross-entropy loss |
| Model selection | Best checkpoint by validation macro F1 |
| Seed | 42 |

Two choices deserve justification because both were non-obvious during implementation. Class weighting was applied through a custom trainer that overrides the loss function, because the standard `Trainer` ignores class weights passed by configuration; without this override the neutral class would be under-served despite the corpus being balanced overall — a defect that would have been invisible in the aggregate metrics and visible only in the per-class breakdown. And early stopping on validation macro F1 rather than validation loss was chosen because macro F1 is the metric the study reports, and selecting checkpoints on one quantity while reporting another risks presenting a model optimised for something other than the reported result.

### 3.6.2 Complaint discovery

BERTopic runs separately on the negative reviews of each platform. Running per platform, rather than over the pooled corpus, is a deliberate control: Yelp contributes 70% of the corpus, and pooling would let it dominate the topic space, producing a taxonomy that describes restaurant service and calls it customer feedback. The cost of this choice is that topics must afterwards be reconciled across platforms, which is the merging step below — the study accepts a harder consolidation problem in exchange for a taxonomy that is not simply a description of its largest source.

Minimum topic size is derived from corpus size rather than fixed, set at approximately 0.15% of the platform's document count and bounded between 30 and 300. This keeps topic granularity comparable across platforms of very different sizes: a fixed minimum would produce either an unusable proliferation of topics on the largest platform or almost none on the smallest. In practice this yielded minimum topic sizes of 86 for Amazon (57,931 documents), 224 for Yelp (149,838 documents) and 30 for Twitter Airline (8,890 documents, where the lower bound binds).

The per-platform topics are then merged. Each topic's centroid embedding is taken, and agglomerative clustering groups near-duplicate topics across platforms down to a target of 24 categories. Each merged cluster is named by a language model, which is given the cluster's constituent topic labels and keywords and asked to produce a category name, a one-sentence description and exemplar phrases. The description is not presentational: it becomes the hypothesis used by the zero-shot classifier in the next stage, so the naming step directly determines assignment quality — a dependency whose consequences Chapter 4 reports.

### 3.6.3 Category assignment

Assigning an individual review to a category uses two stages. An embedding comparison ranks all categories against the review and retains a shortlist of six. A zero-shot entailment model then scores only that shortlist, using each category's description as the hypothesis sentence — following the verbalisation finding in Section 2.1.2, where label phrasing was reported to matter as much as model choice.

The shortlist keeps entailment cost bounded as the taxonomy grows, which a flat zero-shot pass over all categories would not. A confidence threshold of 0.35 is applied: below it, the review is recorded as unclassified rather than forced into the nearest category. This is a deliberate trade of coverage for honesty. Leaving a review unclassified is more informative than assigning it wrongly, and the unclassified rate becomes a reportable measure of taxonomy coverage instead of a hidden error. As Chapter 4 shows, that decision is what made the taxonomy's principal weakness measurable rather than invisible.

### 3.6.4 Retrieval and generation

The corpus is embedded with `all-MiniLM-L6-v2` and indexed in FAISS with normalised vectors. Retrieval returns five reviews per query from a candidate pool of twenty, selected by maximal marginal relevance. MMR is preferred over plain similarity search for a domain-specific reason: complaint questions tend to retrieve near-duplicate reviews expressing the same grievance, and five restatements of one complaint provide weaker grounding than five distinct ones, while scoring identically on similarity.

Each retrieved review is tagged with its platform, rating and sentiment before entering the prompt. This is not cosmetic. The prompt instructs the model to cite only platforms present in the evidence, and without the tags that instruction can neither be followed nor checked. The prompt further requires that no statistic, category or quotation appear unless present in the retrieved text, and specifies a fixed refusal string for cases where the evidence is insufficient — making refusal an available output rather than something the model must improvise.

The same questions are then answered by the model with no retrieval at all, providing the baseline for RQ2. Holding the generator, temperature and question set constant across the two conditions means the difference between them is attributable to retrieval rather than to model or prompt variation.

### 3.6.5 Evaluation metrics

Metrics are matched to what each stage permits, which is why they differ so sharply between stages.

**Sentiment.** Accuracy, macro F1, weighted F1, per-class precision, recall and F1, and confusion matrices. Macro F1 is the headline figure because it weights all three classes equally and therefore exposes the single-class collapse that Section 2.1.3 identified as the literature's blind spot. Per-platform metrics are reported alongside aggregates for the same reason. Pairwise model comparison uses McNemar's exact test, appropriate for paired predictions on a common test set.

**Taxonomy.** No ground truth exists, so no accuracy is computed and none is implied. Quality is argued from mean pairwise similarity between merged category centroids, within-category coherence measured as cosine similarity of each review to its category centroid, category frequency distribution, platform distribution, and unclassified rate. A sample of reviews is also set aside for manual inspection. Reporting five weak indicators together is the direct methodological response to the coherence-validity critique in Section 2.1.3: if no single automated measure is trustworthy, a claim should not rest on one.

**Generation.** RAGAS faithfulness, answer relevancy, context precision and context recall, computed for both the retrieval-augmented and standalone conditions. Four metrics are used because they can disagree informatively — an unconstrained model can score highly on relevancy while having no evidential basis whatever, so relevancy alone would reward precisely the behaviour the system exists to prevent.

**Diagnostic experiments.** Aggregate metrics establish a ranking but not a cause, and Chapter 2 criticised the literature for stopping at the ranking. Where a model behaves unexpectedly, three further experiments test the proposed explanation rather than leaving it as speculation. A per-platform breakdown evaluates every model separately on each source, using a sample balanced across platform and class so that no platform dominates — 600 reviews per platform, 1,800 in total. A binary re-evaluation removes the neutral class and requires each model to choose between negative and positive using its own class scores, isolating that class's contribution to the aggregate. A threshold sweep re-labels VADER's cached compound scores across progressively wider neutral bands, testing whether its weakness is a decision-boundary artefact or a property of the lexicon itself. All three reuse a single cached set of predictions, so each model is run over the sample only once — a practical necessity given transformer inference cost, and one that also guarantees the three diagnostics describe identical model behaviour.

One caution about scope applies to all three. They are computed on the 1,800-review diagnostic sample, not the full held-out test set, so their figures are not directly interchangeable with the headline metrics and are not treated as such in Chapter 4.

A constraint on the generation evaluation belongs here rather than in the results. The model that judges the answers is the same model that produces them, which introduces the self-enhancement bias documented in the LLM-as-judge literature reviewed in Section 2.2.3. This followed from free-tier API limits, which also constrained the number of evaluation questions. Both facts qualify the strength of the RQ2 and RQ4 conclusions, and both are restated wherever those conclusions appear.

### 3.6.6 Scope of the generation evaluation: implemented versus executed

Two planned components were not completed, and the boundary is recorded here so that Chapter 4 reports only work actually carried out. The distinction between code that exists and experiments that were run matters for the integrity of the results chapter, and conflating the two would misrepresent the evidence base.

The ablation suite covering retrieval depth, retriever type, prompt version and metadata tagging was **implemented in `scripts/evaluate_rag.py` but not executed**. Each configuration requires a full pass over the question set with several judged sub-calls per answer, and the free-tier rate limit made the total call volume unattainable within the project timeframe. The consequence is stated plainly: the contribution of individual retrieval design decisions is argued from design reasoning in this chapter, not measured. Chapter 5 treats those arguments as reasoning rather than evidence, and Chapter 6 lists the ablation as the highest-value piece of future work, since it is the one experiment that would convert several design justifications into measured contributions.

The generation comparison was therefore carried out with `evaluate/evaluate_llm_vs_rag.py`, which scores the retrieval-augmented pipeline and the standalone baseline on a common question set using the four RAGAS metrics. The same rate limit constrained that evaluation to **five questions**, which is small enough that per-question variance dominates the reported means. This is stated as a bound on the evidence rather than presented as a complete evaluation, and Chapter 5's answer to RQ2 is framed as directional accordingly.

---

## 3.7 Ethical Considerations

**Data access.** All three datasets are publicly available and were collected and released by their original publishers. No data was scraped, and no new data was gathered from individuals.

**Ethical approval.** No human participants were involved and no primary data collection took place, so institutional ethical approval for human-subjects research was not required. This position was reviewed against the University's research ethics guidance for secondary analysis of public datasets.

**Privacy and personal data.** The corpus contains no special-category personal data. Reviewer names and account identifiers are not used at any stage; reviews are referenced only by `review_id`. No attempt is made to link reviews to individuals, or to profile any reviewer across records.

Under GDPR, review text is user-generated content that may incidentally contain personal information written voluntarily by the reviewer. Three measures apply. Processing is limited to research purposes under legitimate interest, with no commercial use. No identifying fields are extracted, retained or displayed. And outputs are aggregated: the system reports complaint patterns rather than individuals, and the sample reviews it surfaces are presented as evidence for a pattern, not as profiles.

**Responsible use of generated output.** Two risks are worth recording. Generated summaries describe named businesses, and acting on an unsupported summary could affect a business unfairly; returning source reviews with every answer is a partial safeguard, since any claim can be checked before it is used. Separately, sentiment and complaint categories assigned automatically carry error rates that this study measures and reports, and using such output for consequential decisions without regard to those error rates would be inappropriate. Reporting per-class and per-platform performance instead of a single accuracy figure is intended to make those limits visible to anyone relying on the results — which is an ethical argument for the reporting granularity, not only a methodological one.

**API and data handling.** Review text is transmitted to a third-party inference provider during the generation stage. Since the data is already public and contains no identifying fields, this does not create a new disclosure risk, but it is recorded for transparency. API credentials are held in environment configuration and excluded from version control.

---

## 3.8 Chapter Summary

This chapter has set out an adapted CRISP-DM methodology, chosen because its data preparation and evaluation phases carry the weight a comparative study requires, and modified so that modelling and evaluation execute once per pipeline stage. Each phase maps to a research objective and through it to a research question.

The corpus of 642,692 reviews spans three registers, which is what makes the cross-domain comparison meaningful, and its limitations — rating-derived labels, mixed label provenance, uneven platform representation and selection bias — have been stated at the point each becomes relevant rather than deferred. Preprocessing, splitting, model configuration and evaluation metrics have been documented in sufficient detail to be repeated, including the split inconsistency affecting the fine-tuned model, which is disclosed here with its analytical consequence rather than left for a reader to infer.

Every substantive design decision in this chapter answers a limitation identified in Chapter 2. Controlled splits and a shared protocol address the confounding in single-dataset comparisons. Tuned classical baselines address the asymmetric-effort criticism. Per-class and per-platform reporting addresses the aggregate-metric problem. Five indicators for the taxonomy address the coherence-validity critique. A standalone baseline and four generation metrics address the missing comparison in the RAG literature. And the explicit separation of implemented from executed work in Section 3.6.6 addresses the risk that a reader takes design reasoning for measured evidence.

Chapter 4 applies this methodology and reports what it produced: the exploratory analysis of the three sources, the training and evaluation of the five sentiment classifiers together with the three diagnostic experiments, the topic discovery and taxonomy construction, and the retrieval and generation evaluations. Interpretation is deferred to Chapter 5; Chapter 4's task is to establish the evidence.
