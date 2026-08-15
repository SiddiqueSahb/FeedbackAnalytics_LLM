# Chapter 5 — Discussion

Chapter 4 reported what happened. This chapter argues about what it means. Its purpose is to explain why the results took the form they did, to compare them with the literature reviewed in Chapter 2, to judge the methods and the study itself critically, and to establish how far each research objective was met.

The chapter is organised around the results rather than the pipeline. Section 5.1 interprets the sentiment findings, including the two that contradict expectation. Section 5.2 interprets the taxonomy results. Section 5.3 interprets the retrieval and generation results. Section 5.4 compares the findings with existing research, benchmarks them against published figures, and states what is novel. Section 5.5 examines computational cost and deployment trade-offs. Section 5.6 evaluates the study critically, including its limitations. Section 5.7 assesses each objective and research question in turn. Section 5.8 draws out the practical implications, and Section 5.9 summarises.

---

## 5.1 Sentiment Classification

### 5.1.1 Why the pretrained transformer lost

The headline anomaly is that RoBERTa scored 0.5694 macro F1 against logistic regression's 0.8033. Taken at face value this appears to contradict a well-established finding. It does not, and the diagnostic evidence explains why.

One qualification belongs at the outset, because it narrows the claim. McNemar testing shows the gap to logistic regression is statistically robust at p = 2.6 × 10⁻⁴, but the gap to Naive Bayes is not: on the 675 reviews where those two disagree, each is correct on almost exactly half, giving p = 0.82. Despite a difference of 0.19 in macro F1, RoBERTa and Naive Bayes cannot be separated on the available evidence. The defensible statement is therefore that the pretrained transformer underperformed the *stronger* classical model, not that it underperformed classical machine learning as a family. That distinction matters, and it only became visible because the comparison was tested rather than read off a table.

The failure is confined to one class. RoBERTa's per-class F1 is 0.743 for negative and 0.740 for positive, both respectable, against 0.225 for neutral. VADER shows the same shape more extremely, at 0.086 on neutral. The confusion matrices locate the error precisely: 85.7% of genuinely neutral reviews are pushed to one of the two poles, 36.2% negative and 49.5% positive. RoBERTa is not weak; it is behaving as a binary classifier confronted with a three-class task.

Separating precision from recall sharpens the diagnosis further. RoBERTa's neutral precision is 0.518 while its neutral recall is 0.143; VADER's are 0.377 and 0.048. Precision is moderate in both cases, recall is near zero. Neither model is systematically mistaken about what neutral text looks like — when either does assign the label it is right around half the time. What they fail to do is assign it at all. A recall failure with intact precision is the signature of a decision boundary that never reaches the middle class, which is precisely what would be expected if the model's internal notion of neutral does not correspond to the one being tested.

Two experiments confirm this rather than merely asserting it. Removing the neutral class and forcing a binary decision raises RoBERTa's macro F1 from 0.6669 to 0.9341, above logistic regression's 0.9216. The expected ordering is restored the moment the offending class is removed, which would not happen if the model were generally inferior. And the per-platform breakdown shows RoBERTa's neutral F1 at 0.713 on Twitter Airline against 0.198 on Yelp. The same model handles neutral competently on one platform and fails on another.

The explanation follows from what the model is. `twitter-roberta-base-sentiment-latest` was pretrained on tweets and fine-tuned for a three-class scheme in which neutral means a human judged a post to express no clear sentiment (Barbieri et al., 2020). The label it learned and the label it is being tested against are different constructs. On Amazon and Yelp, neutral means a three-star rating, which is a compound judgement often containing both praise and complaint. A model trained to recognise affectively flat text will not recognise a mixed three-star review as neutral, because it is not affectively flat at all.

This makes the result a measurement of label mismatch and domain shift, not of model quality. Its practical implication is direct: an off-the-shelf sentiment model is only as transferable as its label definition, and no accuracy figure quoted from a benchmark can tell a practitioner whether that definition matches their data.

### 5.1.2 The ranking reverses by platform

The per-platform evidence is stronger than a diagnosis of one anomaly. On Twitter Airline, RoBERTa achieves 0.7853 macro F1 against logistic regression's 0.6434 and Naive Bayes's 0.4838 — the transformer wins decisively. On Yelp the ordering inverts, with RoBERTa at 0.5544 against logistic regression's 0.7948.

No single ranking of these methods is therefore correct. Which method is best depends on the relationship between the pretraining domain and the target domain, and a study conducted on one platform would have concluded confidently and incorrectly in either direction. This is the most substantive finding of the chapter for RQ1, and it is only visible because the comparison was run across three registers with per-platform reporting.

### 5.1.3 Why VADER failed, and why the obvious explanation is wrong

VADER's macro F1 of 0.4381 is barely above chance. The intuitive explanation is that its decision thresholds are badly calibrated: the default neutral band of ±0.05 is very narrow, so few texts are ever assigned to neutral.

The threshold sweep tests this and largely rejects it. Widening the band raises macro F1 from 0.5094 to a maximum of 0.5353 at ±0.25 on the balanced sample. The improvement is genuine but small, and macro F1 declines again as the band widens further. Threshold choice accounts for perhaps a fifth of the gap to the classical models.

The larger problem is the lexicon itself. VADER labels 40.4% of genuinely negative reviews as positive, which is not a threshold artefact but a systematic failure of word-level scoring on text containing mixed vocabulary. A review reading "great value and freshness but the taste is awful" carries positive lexicon terms that a context-free scorer cannot discount. This is the limitation Section 2.1.2 identified in principle, and the confusion matrix demonstrates it in practice.

The finding still supports VADER's role in the study. As a baseline it establishes what rules alone achieve, and the 36-point macro F1 gap to the fine-tuned transformer quantifies the value of learning from data on this corpus.

### 5.1.4 Fine-tuning

The fine-tuned model achieved 0.8375 macro F1 on clean held-out data, and its performance is notably stable across platforms: 0.8448 on Amazon, 0.8301 on Yelp and 0.8505 on Twitter. Where the pretrained model varied by 23 points across platforms, the fine-tuned model varies by two.

That stability is the real return on fine-tuning, and it is a different claim from the usual one. The gain over logistic regression on aggregate macro F1 is modest — roughly three points on the shared test set — though McNemar testing confirms it is a real difference rather than sampling noise, at p = 1.7 × 10⁻²¹. The gain in consistency across domains is larger and less often reported. For a practitioner processing feedback from several sources, the second matters more than the first, and a single-domain comparison would not have surfaced it.

The neutral class remains hardest even after fine-tuning, at 0.747 F1 against 0.837 and 0.889 for the other classes. This is consistent with the labelling limitation set out in Section 4.2.4: three-star reviews are genuinely ambiguous, and some ceiling is imposed by the labels themselves, not by any model.

### 5.1.5 Why the transformer beat the classical models

Stating that DistilBERT scored higher than Naive Bayes is not an explanation, and the size of the gap deserves accounting for. On the shared test set DistilBERT reached 0.8244 macro F1 against Naive Bayes's 0.7580 and logistic regression's 0.8033. Three mechanisms explain the difference, and the per-class evidence indicates which matters most here.

The first is representation of word order and negation. TF-IDF discards sequence entirely, so "not worth the money" and "worth the money, not..." produce identical feature vectors. Self-attention conditions each token on its context, which allows negation scope to be represented (Devlin et al., 2019). Amazon and Yelp reviews frequently contain concessive constructions — "the food was good but the service was appalling" — where the sentiment-bearing clause is not identifiable from unigram counts.

The second is handling of unseen and rare vocabulary. Naive Bayes over TF-IDF treats every term independently and has no representation for words absent from training. DistilBERT's WordPiece tokeniser decomposes unfamiliar words into known subwords, and its pretrained embeddings place semantically similar terms near one another, so "malfunctioning" and "faulty" contribute similarly even if only one appeared in training.

The third is Naive Bayes's conditional independence assumption specifically, which is why it trails logistic regression by 4.5 points despite using identical features. Naive Bayes assumes features are independent given the class, which is plainly false for text: "customer" and "service" co-occur constantly. Logistic regression makes no such assumption and can learn compensating weights, so most of the gap between Naive Bayes and DistilBERT is not attributable to the transformer architecture at all — roughly two thirds of it is recovered by simply replacing Naive Bayes with a better-specified linear model.

The confusion matrices in Figure 4.2 indicate where the remaining advantage lies. DistilBERT's gain over logistic regression is concentrated in the neutral class, 0.747 against 0.718, and in reduced polar confusion: it misclassifies negative reviews as positive only 1.0% of the time against logistic regression's 1.5%. Both are cases requiring the model to weigh mixed evidence within a single review, which is precisely what contextual representation supports and bag-of-words does not.

This reading has a practical consequence worth stating. If the gain concentrates on the ambiguous middle class, then for a business interested only in separating satisfied from dissatisfied customers, the fine-tuning cost buys very little: on the binary re-evaluation the two models sit at 0.9533 and 0.9216, a gap of three points for roughly three hours of GPU training.

---

## 5.2 Complaint Taxonomy

### 5.2.1 What the taxonomy achieved

Consolidation worked at the level it was designed for. The 121 raw topics reduced to 24 categories with a centroid separation of 0.2831, meaning the resulting categories describe distinguishable things. Mean within-category coherence on held-out data was 0.6334, and 21 of the 24 categories were used.

The per-platform discovery decision was also vindicated. Yelp produced 18 topics from 149,838 documents while Amazon produced 69 from 57,931. Pooling would have allowed the larger corpus to dominate the topic space, and the resulting taxonomy would have described Yelp's concerns while claiming to describe customer feedback generally.

### 5.2.2 Why Amazon produced 69 topics and the merge kept 24

The asymmetry in raw topic counts is the most informative structural result of this stage, and it is not an artefact of parameter choice. Amazon yielded 69 topics from a third of Yelp's documents; Yelp yielded 18 from three times as many.

The cause is the semantic breadth of what is being complained about. Amazon spans electronics, groceries, clothing, tools and cosmetics, and complaints in each are lexically distinct: a failing battery, a stale flavour, a poor fit and a distorted speaker share no vocabulary. HDBSCAN, clustering over embeddings, finds dense regions wherever such vocabulary clusters, and Amazon's product diversity creates many small dense regions. Yelp is semantically narrow by comparison — restaurants, bars and salons, where complaints concentrate on service, waiting, staff conduct and food quality. Many documents describing few underlying concerns produce few large clusters.

The merge to 24 therefore performs different work on each platform. For Amazon it is genuine consolidation, collapsing product-specific complaint vocabularies into transferable categories: separate topics about batteries, screens and connectivity become one category concerning product performance. For Yelp it is closer to relabelling, since 18 topics were already close to the target. The compression ratio differs by almost a factor of four, and the taxonomy is consequently a better fit for Yelp's structure than for Amazon's.

Three implications follow, and they are visible in the results. First, the catch-all behaviour identified in Section 5.2.3 is a predictable consequence: "Product Performance Failures" absorbs 28.2% of held-out reviews because it is the merged descendant of many distinct Amazon topics that had little in common beyond being product faults. Second, imposing a single target count across platforms with different intrinsic granularity is questionable — a count derived per platform, or a hierarchical taxonomy with Amazon categories nested more deeply, would fit the data better. Third, the reduction from 121 to 24 discarded genuine distinctions that a business would care about: knowing that complaints concern battery life specifically is more actionable than knowing they concern product performance generally.

The practical reading is that the level of aggregation should follow the decision the taxonomy supports. For executive reporting, 24 categories is appropriate. For a product team investigating a specific failure, it is too coarse, and the 69 Amazon topics discarded in the merge were closer to what such a team needs.

### 5.2.3 Where it failed, and why

Three defects are more informative than the headline figures.

First, 27.6% of held-out reviews fell below the confidence threshold and were left unclassified. This is a coverage failure, though a visible one: the threshold was set precisely so that weak matches would be recorded rather than forced into the nearest category. Had the threshold been removed, those 124 reviews would have been distributed among existing categories and the taxonomy would have appeared to work considerably better than it does.

Second, the largest category absorbed 28.2% of the sample while showing among the lowest within-category coherence at 0.3997. The inverse relationship between size and coherence across categories indicates that "Product Performance Failures" is functioning as a catch-all, accumulating loosely related complaints because no more specific category fits. Combined with the unclassified rate, this points to a taxonomy that is simultaneously too coarse in one region and too sparse elsewhere. The target of 24 categories was chosen in advance rather than derived from the data, and this result suggests that choice was not optimal.

Third, and most seriously, the taxonomy has no mechanism for rejecting non-complaints. It was constructed from negative reviews only, but the assignment stage was applied to reviews of all sentiments, and the results show positive reviews being assigned to complaint categories: one category received 19 positive and zero negative reviews, and the unclassified group holds 60 positive against 34 negative. The zero-shot classifier scores relative fit among candidate categories; it has no notion of whether a text is a complaint at all. A sentiment filter before assignment would address this directly, and its absence is a design error, not a limitation of the method.

A fourth, smaller defect indicates a pipeline reliability issue: one category retained an un-renamed raw BERTopic label ("Spray Bottle, Continuous Spray, Dryer Diffuser, Fit Dryer"), meaning the language-model naming step failed silently on at least one cluster and no validation caught it.

### 5.2.4 The evaluation problem itself

These conclusions rest on unsupervised measures, and their weakness deserves acknowledgement. Section 2.1.3 reviewed evidence that automated coherence metrics correspond poorly to human judgement, with false discovery rates between 16% and 56% reported against human assessment (Doogan-Poet Sirakoff & Buntine, 2024), and that they can favour topics humans find obscure.

The approach taken here — several imperfect indicators read together, plus a manual validation sample — follows the direction that critique points in, but it does not escape the underlying problem. The coherence figures reported are embedding-based and inherit whatever biases the embedding model carries. Where the argument in Section 5.2.2 is strongest is not in the coherence values themselves but in the structural evidence: the unclassified rate, the size distribution, the sentiment breakdown and the failed naming are all directly observable and do not depend on a contested metric.

---

## 5.3 Retrieval and Generation

### 5.3.1 Retrieval

Dense retrieval was 5.9 times faster than TF-IDF at identical Precision@3 of 0.90. The honest reading is narrower than it might appear. On these 30 queries the embedding retriever does not find relevant reviews more often; it finds them faster. The similarity scores cannot support a quality claim either, since cosine similarity over dense and sparse vectors measure different things.

The result is nonetheless useful, because latency determines whether interactive question answering is viable at all. At 127 ms per retrieval, TF-IDF would consume a fifth of a second before generation begins. The case for dense retrieval here is architectural, not a claim about accuracy, and stating it that way is more defensible than claiming a quality advantage the data does not show.

### 5.3.2 The relevancy inversion

The standalone model scored 0.920 on answer relevancy against the retrieval-augmented pipeline's 0.634, and did so in less than half the time. On a single metric, retrieval made the system worse.

This is explicable, and the explanation matters more than the number. Answer relevancy measures how well a response addresses the question asked. An unconstrained model can answer any question about customer complaints fluently and comprehensively, because it is drawing on parametric knowledge and is not restricted to what any particular corpus contains. The retrieval-augmented pipeline is constrained to five retrieved reviews and instructed to refuse when evidence is insufficient. When the retrieved context is thin, a well-behaved grounded system produces a narrow answer or declines — behaviour that answer relevancy penalises and faithfulness rewards.

The inversion therefore reflects a real property of the metric, not a defect in the system, and it shows concretely why Section 2.2.3 argued for four metrics instead of one. A study reporting only answer relevancy would have concluded that retrieval augmentation degrades performance. The faithfulness score of 0.755, and the fact that faithfulness is undefined for the baseline, tell the opposite story: the baseline's fluent answers have no verifiable basis, and its high relevancy is precisely the behaviour the framework exists to prevent.

Self-enhancement bias cannot explain this away either. Because the same model generates and judges, bias would inflate the pipeline's scores, not the baseline's, so the true gap is likely wider, not narrower.

### 5.3.3 What the generation results cannot establish

The generation evaluation is the weakest part of this study, and its limits should be stated plainly.

Five questions is too few. Per-question faithfulness ranged from 0.444 to 1.000; context precision included scores of 0.000 and 1.000; context recall was 0.000 on two of five questions. With that variance and that sample, the means carry uncertainty wide enough that the specific values should not be quoted as measurements of system quality. What the evidence supports is the qualitative finding — that relevancy and faithfulness diverge, and that the baseline cannot be assessed for grounding — not any precise figure.

The judge model is the second constraint. Using the generating model as evaluator is a known threat to validity, catalogued in the survey literature reviewed in Section 2.1.2 (Gu et al., 2024; Li et al., 2024). It was accepted because free-tier rate limits made an independent judge impractical within the project's resources, and it should be read as a bound on confidence in the RQ2 and RQ4 conclusions.

---

## 5.4 Comparison with Existing Research

### 5.4.1 Where the findings agree

The overall ordering of methods matches the literature. Fine-tuned transformers performed best, classical models next, lexicon methods worst — consistent with the comparative studies on Amazon review data reviewed in Section 2.1.2. BERTopic produced usable topics from short informal text where LDA is documented to struggle, consistent with Turan et al. (2024) and Mishra (2026).

### 5.4.2 Benchmarking against published figures

Comparing absolute numbers with published work requires care, because task definition differs more than model choice does. Table 5.1 sets this study's results beside reported figures, with the differences in task noted.

**Table 5.1** — This study against published benchmarks

| Source | Task | Dataset | Best reported | This study |
|---|---|---|---|---|
| Areshey & Mathkour (2024) | Binary polarity | Yelp | RoBERTa 98.30%, BERT 97.40%, DistilBERT 96.00% | — |
| This study, binary re-evaluation | Binary, neutral removed | Three platforms | — | DistilBERT macro F1 0.9533, RoBERTa 0.9341 |
| Financial sentiment study (2023) | Three-class | Financial text | BERT 87% accuracy, macro F1 0.85 | DistilBERT 84.37% accuracy, macro F1 0.8375 |
| Turan et al. (2024) | Topic quality | Amazon | BERTopic most interpretable of three | Centroid separation 0.2831 |
| Indonesian complaint study (2024) | Topic coherence | Customer complaints | LDA 0.68, BERTopic 0.51 | Within-category coherence 0.6334 |

Three comparisons are worth drawing out.

The apparent gap against the Yelp binary benchmarks — 96–98% there against 84% here — is almost entirely a task difference, not a modelling shortfall. Those studies classify polarity on two classes with the ambiguous middle removed. When this study removes the same class, DistilBERT reaches 0.9533 macro F1 and RoBERTa 0.9341, which sits in the same region. The comparison is a useful corrective to reading benchmark leaderboards as achievable targets: adding a third class costs roughly ten points, and much of the published literature's headline performance depends on having excluded the hardest cases.

Against a genuine three-class benchmark the correspondence is close. A fine-tuned BERT reported 87% accuracy and 0.85 macro F1 on three-class financial sentiment; DistilBERT here achieved 84.37% and 0.8375 on three-class review sentiment across three platforms. Given that this corpus is multi-domain and rating-labelled rather than expert-annotated, performance within three points of a single-domain benchmark is a reasonable outcome, and it suggests the pipeline is not underperforming relative to the field.

The taxonomy comparison is less favourable and should be read as such. The Indonesian complaint study reported BERTopic coherence of 0.51 against LDA's 0.68, and this study's within-category coherence of 0.6334 is not directly comparable since it uses embedding similarity rather than C_v. What both indicate is that BERTopic's coherence advantage over LDA is not established — its benefit lies in handling short informal text and in producing separable clusters, and the coherence literature reviewed in Section 2.1.3 casts doubt on whether these metrics measure interpretability at all.

### 5.4.3 Where they diverge, and why

The divergence is instructive, not contradictory. Published comparisons report transformers outperforming classical models; this study finds a pretrained transformer 23 points below a linear model on aggregate macro F1. The difference is explained by experimental design, not by disagreement about the underlying methods. Those studies evaluate within a single domain, usually Amazon reviews, where a domain-matched or domain-neutral model is being tested on data resembling its pretraining distribution. This study tests a Twitter-pretrained model on Amazon and Yelp prose, and measures what happens.

The cross-domain sentiment literature reviewed in Section 2.1.2 treats exactly this problem as serious enough to warrant purpose-built architecture — feature projection, adversarial domain adaptation, multi-source attention. That literature and the applied comparison literature rarely meet. This study sits between them: it takes the applied position of using off-the-shelf models, but tests the assumption of transferability that applied comparisons usually leave implicit, and finds it does not hold.

Similarly, the relevancy inversion is not a contradiction of the RAG literature but an instance of the evaluation problem that literature has begun to identify. Yu et al. (2024) argue that aggregate RAG scores cannot diagnose whether failure originates in retrieval or generation; the divergence between relevancy and faithfulness observed here is a concrete example of why single-metric evaluation misleads.

### 5.4.4 What is novel

Three contributions can be claimed without overstatement.

The stage-wise cross-domain comparison is the principal one. The literature contains many single-domain comparisons and a separate body of work on domain adaptation, but comparisons of the same off-the-shelf methods across three registers with per-platform and per-class reporting are uncommon. The finding that the ranking reverses by platform is a direct product of that design.

The diagnostic method is the second. Instead of reporting an anomalous result and speculating, this study tested the explanation through two independent experiments — binary re-evaluation and per-platform breakdown — that make the mechanism demonstrable. This is a modest methodological contribution, but a real one, since the reporting conventions criticised in Section 2.1.3 would have concealed the finding entirely.

The third is negative and correspondingly honest: the demonstration that a complaint taxonomy built from negative reviews will assign positive reviews to complaint categories unless a filter intervenes. This is not a headline result, but it is the kind of failure that a study reporting only aggregate coherence would never have surfaced.

---

## 5.5 Computational Cost and Deployment Trade-offs

Accuracy is one axis of method selection and rarely the binding one in practice. This section sets out what each stage costs and what that implies for deployment, since a method that wins on macro F1 but cannot be served within budget is not the method a practitioner should choose.

### 5.5.1 Cost of the sentiment stage

Table 5.2 collects the costs measured or recorded in this project alongside published figures for the model families used.

**Table 5.2** — Cost profile of the five sentiment classifiers

| Model | Parameters | Training cost | Artefact size | Hardware to serve | Interpretability |
|---|---|---|---|---|---|
| VADER | None (lexicon) | None | ~1 MB lexicon | CPU, trivial | High — rules inspectable |
| Naive Bayes | ~vocabulary size | Seconds, CPU | 76 MB (with vectoriser) | CPU | Moderate — feature log-probabilities |
| Logistic Regression | ~vocabulary size | Minutes, CPU | shared with above | CPU | Moderate — signed coefficients |
| RoBERTa (pretrained) | ~125M | None (used as-is) | ~500 MB | GPU preferred | Low |
| DistilBERT (fine-tuned) | ~66M | 10,545 s (2 h 56 m), GPU | 257 MB | CPU viable, GPU preferred | Low |

The distillation choice is vindicated on these figures. DistilBERT carries roughly 40% fewer parameters than BERT-base while retaining about 97% of its performance on GLUE and running around 60% faster (Sanh et al., 2019), which is what made fine-tuning feasible on freely available hardware within this project. Selecting BERT or RoBERTa for fine-tuning would have roughly doubled both training time and served model size for a gain the evidence does not suggest would have been large.

The more interesting comparison is between logistic regression and DistilBERT. The fine-tuned model gains three points of macro F1 on the shared test set for a 257 MB artefact, three hours of GPU training, and materially higher inference cost. Logistic regression trains in minutes on a CPU and serves from a small sparse model. For a system processing a few thousand reviews daily on modest infrastructure, the classical model is the defensible engineering choice, and Section 5.1.5 shows the transformer's advantage concentrates on the ambiguous neutral class that many applications do not need resolved.

Two caveats bound this analysis. Per-review inference latency was not measured for the five models, so the comparison rests on parameter counts and published throughput figures rather than on timings from this system. That is a gap, and adding a timing pass to the diagnostics script would close it cheaply. And interpretability is assessed qualitatively here: no formal explanation method such as SHAP or LIME was applied, so the claim that classical models are more interpretable rests on the transparency of their parameters rather than on a measured comparison of explanation quality.

### 5.5.2 Cost of retrieval and generation

The retrieval and generation stages carry costs of a different kind, dominated by storage and network latency rather than training.

Indexing 642,692 reviews at 384 dimensions produces a raw embedding matrix of 0.99 GB, and the artefacts on disk are substantial: 942 MB for the FAISS index and a further 1.4 GB for the LangChain index with its metadata. Storage therefore scales linearly with corpus size, and a corpus ten times larger would need roughly 10 GB before any redundancy measures. This is the practical argument for the small embedding model. Moving to a 768-dimensional model such as `all-mpnet-base-v2` would double the index footprint and roughly double query time for a retrieval quality gain that Section 5.3.1 suggests would be difficult to demonstrate, given that Precision@3 was already identical between dense and sparse retrieval.

At query time the dominant cost is generation, not retrieval. Retrieval takes 21.71 ms against TF-IDF's 127.61 ms, but the end-to-end retrieval-augmented answer takes 1,516 ms against the standalone model's 650 ms. Retrieval accounts for under two per cent of that total; the additional 866 ms comes from the larger prompt the model must process once five reviews are inserted. The cost of grounding, in other words, is paid in generation tokens rather than in search.

That has a direct implication for deployment. Grounding roughly doubles response time and increases token consumption per query, which on a metered API translates into a proportional cost increase. Whether that is worth paying depends on the use case: for exploratory analysis where a wrong answer is discovered quickly, it may not be; for reporting that informs a decision, the verifiability it buys is difficult to obtain any other way.

### 5.5.3 Deployment implications

Taken together the cost profile suggests a tiered architecture rather than a single model choice. High-volume routine classification is well served by logistic regression on CPU. The fine-tuned transformer earns its cost where the neutral class matters or where feedback arrives from several registers, since Section 5.1.4 showed its cross-platform stability to be its real advantage. The retrieval-augmented component is inherently interactive and low-volume, so its latency is acceptable, but its per-query cost means it belongs in an analyst-facing tool rather than in an automated pipeline processing every incoming review.

None of this was tested in deployment, and that limits how firmly it can be asserted. The figures above are measured costs and published throughput claims, not observations of a served system under load.

---

## 5.6 Critical Evaluation

### 5.6.1 Strengths

The comparison rests on a large, genuinely multi-domain corpus with a stratified, seeded split, and the sentiment stage is evaluated on 128,539 held-out reviews. Reporting is at class and platform level throughout, which is what made the central finding visible. Baselines are present at every stage where one was possible: a lexicon floor for sentiment, TF-IDF for retrieval, and an unconstrained model for generation. Differences between models are tested for statistical significance rather than asserted from a ranking, which identified one pair that cannot be separated and prevented an overclaim. Where results were unfavourable or contradicted expectation, they are reported and explained, not omitted.

### 5.6.2 Limitations

Several limitations bound what this study can claim, and they are of unequal severity.

The most serious concerns experimental control. The fine-tuned model was trained on a separately generated split, so approximately a quarter of the shared sentiment test set overlaps data it had seen. The clean figure from its own held-out partition is reported as definitive, but the consequence is that the five-model comparison is not as controlled as the design intended. A single canonical split reused by every model would remove the objection entirely.

Label quality bounds everything downstream. Deriving sentiment from star ratings means a four-star review containing a complaint is labelled positive, and the neutral class carries different meanings across platforms. Some part of every model's neutral-class difficulty is attributable to the labels, not to the models.

The generation evaluation rests on five questions with a self-grading judge, as discussed in Section 5.3.3. This is the weakest evidence in the study.

Complaint categorisation has no ground truth, so no accuracy can be reported for it, and the unsupervised measures used instead are contested in the literature. A manually labelled evaluation set would convert RQ3 from an argument into a measurement.

Finally, the ablation experiments implemented in `scripts/evaluate_rag.py` — varying retrieval depth, retriever type, prompt design and metadata tagging — were not run within the project timeframe because of API rate limiting. Their absence means the contribution of individual design decisions in the retrieval stage is argued, not measured.

### 5.6.3 Critical evaluation of the methods

Judged on outcomes, some choices held up better than others.

BERTopic was the right choice for the discovery stage, and the per-platform decision was correct. The fixed target of 24 categories was not: the size distribution and unclassified rate both suggest the number should have been derived from the data.

The zero-shot assignment approach is sound in principle and avoids needing labelled examples for newly discovered categories, but applying it without a preceding sentiment filter was an error, and the sentiment-by-category results expose it.

The prompt-level grounding constraints and metadata tagging appear to work, in that faithfulness reached 0.755, but without the ablation results this cannot be attributed to those specific mechanisms with confidence. The faithfulness figure shows the pipeline as a whole is reasonably grounded; it does not isolate which design decision produced that.

Using a free-tier hosted model was a resource-driven decision with consequences that propagated further than expected. It constrained sample sizes, prevented an independent judge, and ultimately prevented the ablations from running. In retrospect, budgeting a small amount for API access would have strengthened the evaluation more than any modelling change.

---

## 5.7 Assessment Against Objectives and Research Questions

**RQ1 — How does sentiment classification performance differ across method families on multi-domain feedback?** Answered. Fine-tuned DistilBERT performed best at 0.8375 macro F1 on clean held-out data, followed by logistic regression at 0.8033, Naive Bayes at 0.7580, pretrained RoBERTa at 0.5694 and VADER at 0.4381. Significance testing supports most of this ordering but not all of it: nine of ten pairwise differences hold at p < 0.05, while Naive Bayes and RoBERTa are indistinguishable at p = 0.82. The substantive answer is that the ordering is not stable in a second sense either — on Twitter Airline the pretrained transformer outperforms both classical models, so the ranking depends on the match between pretraining and target domain. Fine-tuning's clearest benefit is cross-domain consistency, not peak aggregate accuracy.

**RQ2 — How effectively does RAG improve complaint exploration compared with a standalone LLM?** Partially answered. RAG achieved 0.755 faithfulness with source attribution, while the baseline scored higher on answer relevancy at 0.920 against 0.634. The improvement RAG provides is verifiability, not responsiveness, and the two metrics measure different things. The evidence is weakened by a five-question sample and a self-grading judge, so this conclusion is directional, not precise.

**RQ3 — Can BERTopic produce a coherent cross-domain complaint taxonomy?** Answered with qualification. It produced 24 well-separated categories from 121 raw topics at 0.2831 centroid separation, with 0.6334 mean coherence on held-out data. But 27.6% of reviews were unclassified, one category absorbed 28.2% with low internal coherence, and the taxonomy cannot distinguish complaints from non-complaints. The answer is that BERTopic can produce a usable cross-domain taxonomy, but not without a consolidation stage, a data-driven category count, and a sentiment filter at assignment.

**RQ4 — To what extent are insights traceable to and verifiable against source reviews?** Answered. Every generated answer returns the identifiers of the reviews retrieved to produce it, and faithfulness of 0.755 quantifies how far answers follow from that evidence. Traceability is therefore demonstrated as an architectural property. The figure also shows the limit: at 0.755, roughly a quarter of the claims assessed were not fully supported by the retrieved context, so traceability being available is not the same as every statement being grounded. What is not established is whether that traceability is useful in practice, which would require the user study identified as future work.

Objectives 1 to 6 were met. Objective 6 was met only partially: stage-appropriate metrics and baseline comparisons were delivered, but the ablation component was not executed. Objective 7 is delivered by this chapter and the next.

---

## 5.8 Practical Implications

The preceding sections judged the methods. This one asks what an organisation could actually do with the results, since a comparative study that never reaches that question leaves its motivation unanswered.

**On method selection (Objective 3, RQ1).** The clearest practical finding is that benchmark scores do not transfer. A practitioner choosing a sentiment model on published accuracy would likely select a pretrained transformer and, on Yelp or Amazon prose, obtain worse performance than a linear model trained in minutes. The selection rule this study supports is narrower and more useful: check whether the candidate model's label definition matches yours before considering its reported accuracy, and if feedback arrives from several sources, prefer the model with the smallest variance across them rather than the highest mean.

**On the taxonomy (Objective 4, RQ3).** The taxonomy is usable for executive-level reporting, where 24 categories at 0.2831 separation give a defensible summary of what customers complain about across platforms. It is not usable as an automated routing or triage mechanism in its current state, and the reason is specific: with 27.6% of reviews unclassified and one category absorbing 28.2%, routing on it would leave a quarter of complaints unhandled and overload a single queue. Two changes would address this — a sentiment filter before assignment to stop positive reviews entering complaint categories, and a category count derived per platform rather than fixed at 24. Section 5.2.2 also implies a hierarchical structure would serve better than a flat one, with broad categories for reporting and the discarded finer topics retained beneath them for investigation.

**On the retrieval-augmented component (Objectives 5 and 6, RQ2 and RQ4).** Its practical value is not speed or fluency, both of which the standalone model wins. It is that every answer arrives with the reviews that produced it, so a claim can be checked in seconds before it informs a decision. That matters most where the cost of acting on a wrong conclusion is high and where the analyst is accountable for the recommendation. Where an analyst is exploring informally and will notice a wrong answer immediately, the standalone model is faster, cheaper and adequate. The faithfulness figure of 0.755 also sets a boundary on the claim: roughly a quarter of assessed statements were not fully supported, so the system supports verification but does not remove the need for it.

**On the evaluation approach itself (Objective 6).** The most transferable contribution may be methodological. Reporting per-class and per-platform results rather than aggregates, and testing the mechanism behind an anomalous result rather than reporting it and moving on, is what turned an apparently disappointing finding into the study's main result. That practice costs little and is applicable to any comparative evaluation.

---

## 5.9 Chapter Summary

The central finding is that method ranking in customer feedback analysis is conditional on domain. A pretrained transformer that appears clearly inferior on aggregate metrics outperforms both classical models on the platform matching its pretraining distribution, and its apparent weakness resolves into a single-class failure caused by a mismatch between label definitions. Fine-tuning's principal value is stability across domains, not peak accuracy.

The taxonomy results are more mixed and more critical of the approach taken. Consolidation produced well-separated categories, but coverage was incomplete, one category behaved as a catch-all, and the absence of a sentiment filter allowed positive reviews into complaint categories.

The generation results establish the qualitative case for retrieval augmentation — verifiable grounding where the baseline offers none — while being too small and too self-referentially judged to support precise claims.

Chapter 6 draws these together, assesses the aim as a whole, and sets out the future work each limitation implies.
