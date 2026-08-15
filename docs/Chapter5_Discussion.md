# Chapter 5 — Discussion

Chapter 4 reported what happened. This chapter argues about what it means. Its purpose is to explain why the results took the form they did, to compare them with the literature reviewed in Chapter 2, to judge both the methods and the study itself critically, and to establish how far each research objective was met.

The chapter is organised around the results rather than the pipeline. Section 5.1 interprets the sentiment findings, including the two that contradict expectation. Section 5.2 interprets the taxonomy results. Section 5.3 interprets the retrieval and generation results. Section 5.4 compares the findings with existing research and states what is novel. Section 5.5 examines computational cost and deployment trade-offs. Section 5.6 evaluates the study critically, including its limitations. Section 5.7 assesses each objective and research question in turn. Section 5.8 draws out the practical implications, and Section 5.9 summarises.

---

## 5.1 Sentiment Classification

### 5.1.1 Why the pretrained transformer lost

The headline anomaly is that RoBERTa scored 0.5694 macro F1 against logistic regression's 0.8033. Taken at face value this appears to contradict a well-established finding. It does not, and the diagnostic evidence explains why.

One qualification belongs at the outset, because it narrows the claim. McNemar testing shows the gap to logistic regression is statistically robust at p = 2.6 × 10⁻⁴, but the gap to Naive Bayes is not: on the 675 reviews where those two disagree, each is correct on almost exactly half, giving p = 0.82. This does not establish that the two models are equivalent — a non-significant result cannot demonstrate equality — but the available paired evidence does not support ranking either above the other. The defensible statement is therefore that the pretrained transformer underperformed the *stronger* classical model, not that it underperformed classical machine learning as a family. That distinction only became visible because the comparison was tested rather than read off a table.

The failure is confined to one class. RoBERTa's per-class F1 is 0.743 for negative and 0.740 for positive, both respectable, against 0.225 for neutral. VADER shows the same shape more extremely, at 0.086 on neutral. The confusion matrices locate the error precisely: 85.7% of genuinely neutral reviews are pushed to one of the two poles, 36.2% negative and 49.5% positive. RoBERTa is not weak; it is behaving as a binary classifier confronted with a three-class task.

Separating precision from recall sharpens the diagnosis further. RoBERTa's neutral precision is 0.522 while its neutral recall is 0.143; VADER's are 0.377 and 0.048. Precision is moderate in both cases, recall is near zero. Neither model is systematically mistaken about what neutral text looks like — when either does assign the label it is right around half the time. What they fail to do is assign it at all. A recall failure with intact precision is the signature of a decision boundary that never reaches the middle class, which is exactly what would be expected if the model's internal notion of neutral does not correspond to the one being tested.

Two experiments confirm this rather than merely asserting it. Removing the neutral class and forcing a binary decision raises RoBERTa's macro F1 from 0.6669 to 0.9341, above logistic regression's 0.9216. The expected ordering is restored the moment the offending class is removed, which would not happen if the model were generally inferior. And the per-platform breakdown shows RoBERTa's neutral F1 at 0.713 on Twitter Airline against 0.198 on Yelp. The same model handles the same class competently on one platform and fails on another.

The explanation follows from what the model is. `twitter-roberta-base-sentiment-latest` was pretrained on tweets and fine-tuned for a three-class scheme in which neutral means a human judged a post to express no clear sentiment (Barbieri et al., 2020). The label it learned and the label it is tested against here are different constructs. On Amazon and Yelp, neutral means a three-star rating, which is a compound judgement often containing both praise and complaint. A model trained to recognise affectively flat text will not recognise a mixed three-star review as neutral, because such a review is not affectively flat — it is affectively contradictory.

The evidence supports this interpretation rather than proving it. A design that directly established causation would require re-testing the same checkpoint against human-annotated neutral labels on Amazon and Yelp prose, which was not done. What can be said is that three independent observations — the recall-specific failure, the platform-conditional reversal, and the recovery under binary evaluation — are jointly consistent with label mismatch and difficult to explain by general model inferiority.

Its practical implication is direct: an off-the-shelf sentiment model is only as transferable as its label definition, and no accuracy figure quoted from a benchmark can tell a practitioner whether that definition matches their data.

### 5.1.2 The ranking reverses by platform

The per-platform evidence is stronger than a diagnosis of one anomaly. On Twitter Airline, RoBERTa achieves 0.7853 macro F1 against logistic regression's 0.6434 and Naive Bayes's 0.4838 — the transformer wins decisively. On Yelp the ordering inverts, with RoBERTa at 0.5544 against logistic regression's 0.7948.

Naive Bayes shows the mirror image, scoring 0.7952 on Amazon and 0.4838 on Twitter. Two models therefore swap places entirely depending on which platform is used to judge them.

No single ranking of these methods is correct. Which method is best depends on the relationship between the pretraining or training domain and the target domain, and a study conducted on one platform would have concluded confidently and incorrectly in either direction. This is the most substantive finding of the chapter for RQ1, and it exists only because the comparison was run across three registers with per-platform reporting — precisely the design the aggregate-metric convention criticised in Section 2.1.3 would have prevented.

### 5.1.3 Why VADER failed, and why the obvious explanation is wrong

VADER's macro F1 of 0.4381 against an accuracy of 0.5142 is the widest gap between those two measures of any model in the study, and the gap is the finding. Uniform random guessing on this balanced three-class task would score approximately 0.333 on both, so VADER is well above chance; what it is not is evenly competent, and accuracy alone would have concealed that.

The intuitive explanation is that its decision thresholds are badly calibrated: the default neutral band of ±0.05 is very narrow, so few texts are ever assigned to neutral. That explanation is testable, and the threshold sweep largely rejects it. Widening the band raises macro F1 from 0.5094 to a maximum of 0.5353 at ±0.25 on the balanced sample, a gain of 0.026, after which macro F1 declines again as the band swallows genuinely polar reviews. Threshold choice accounts for a modest fraction of the gap to the classical models, not most of it.

The larger problem is the lexicon itself. VADER labels 40.4% of genuinely negative reviews as positive, which is not a threshold artefact but a systematic failure of word-level scoring on text containing mixed vocabulary. A review reading "great value and freshness but the taste is awful" carries positive lexicon terms that a context-free scorer cannot discount. This is the limitation Section 2.1.2 identified in principle, and the confusion matrix demonstrates it in practice.

The finding still supports VADER's role in the study. As a baseline it establishes what rules alone achieve, and the roughly 40-point macro F1 gap to the fine-tuned transformer quantifies the value of learning from data on this corpus — a floor that would be invisible had no untrained method been included.

### 5.1.4 Fine-tuning

The fine-tuned model achieved 0.8375 macro F1 on clean held-out data, and its performance is notably stable across platforms: 0.8448 on Amazon, 0.8301 on Yelp and 0.8505 on Twitter. Where the pretrained model varied by 23 points across platforms, the fine-tuned model varies by two.

That stability is the real return on fine-tuning, and it is a different claim from the usual one. The gain over logistic regression on aggregate macro F1 is modest — roughly two points on the shared test set — though McNemar testing on the diagnostic sample confirms a real difference rather than sampling noise at p = 1.7 × 10⁻²¹. The gain in consistency across domains is larger and less often reported. For a practitioner processing feedback from several sources, the second matters more than the first, and a single-domain comparison could not have surfaced it.

The neutral class remains hardest even after fine-tuning, at 0.747 F1 against 0.837 and 0.889 for the other classes. This is consistent with the labelling limitation set out in Section 4.2.4: three-star reviews are genuinely ambiguous, and part of the remaining error is imposed by the labels rather than by any model. No classifier trained on this corpus can exceed the ceiling those labels create.

### 5.1.5 Why the transformer beat the classical models

Stating that DistilBERT scored higher than Naive Bayes is not an explanation, and the size of the gap deserves accounting for. On the shared test set DistilBERT reached 0.8244 macro F1 against Naive Bayes's 0.7580 and logistic regression's 0.8033. Three mechanisms explain the difference, and the per-class evidence indicates which matters most here.

The first is representation of word order and negation. TF-IDF discards sequence entirely, so "not worth the money" and "worth the money, not..." produce identical feature vectors. Self-attention conditions each token on its context, which allows negation scope to be represented (Devlin et al., 2019). Amazon and Yelp reviews frequently contain concessive constructions — "the food was good but the service was appalling" — where the sentiment-bearing clause is not identifiable from unigram counts.

The second is handling of unseen and rare vocabulary. Naive Bayes over TF-IDF treats every term independently and has no representation for words absent from training. DistilBERT's WordPiece tokeniser decomposes unfamiliar words into known subwords, and its pretrained embeddings place semantically similar terms near one another, so "malfunctioning" and "faulty" contribute similarly even if only one appeared in training.

The third mechanism explains the gap *within* the classical family and is the most instructive. Naive Bayes trails logistic regression by 4.5 points of macro F1 despite using identical features, because its conditional independence assumption is plainly false for text — "customer" and "service" co-occur constantly. Logistic regression makes no such assumption and learns compensating weights. Most of the distance between Naive Bayes and DistilBERT is therefore not attributable to the transformer architecture at all: roughly two thirds of it is recovered by replacing one linear model with a better-specified linear model.

The per-class evidence indicates where the remaining advantage lies. DistilBERT's gain over logistic regression is concentrated in the neutral class, 0.747 against 0.718, precisely the case requiring a model to weigh contradictory evidence within a single review — which is what contextual representation supports and bag-of-words does not.

This reading has a practical consequence worth stating. If the gain concentrates on the ambiguous middle class, then for a business interested only in separating satisfied from dissatisfied customers, fine-tuning buys very little: on the binary re-evaluation the two models sit at 0.9533 and 0.9216, a gap of three points for roughly three hours of GPU training and a 257 MB artefact.

---

## 5.2 Complaint Taxonomy

### 5.2.1 What the taxonomy achieved

Consolidation worked at the level it was designed for. The 121 raw topics reduced to 24 categories with a centroid separation of 0.2831, meaning the resulting categories describe distinguishable things. Mean within-category coherence on held-out data was 0.6334, and 21 of the 24 categories were used.

The per-platform discovery decision was also vindicated. Yelp produced 18 topics from 149,838 documents while Amazon produced 69 from 57,931. Pooling would have allowed the larger corpus to dominate the topic space, and the resulting taxonomy would have described Yelp's concerns while claiming to describe customer feedback generally.

### 5.2.2 Why Amazon produced 69 topics and the merge kept 24

The asymmetry in raw topic counts is the most informative structural result of this stage, and it is not an artefact of parameter choice — minimum topic size was scaled to each platform's document count precisely to prevent that.

The cause is the semantic breadth of what is being complained about. Amazon spans electronics, groceries, clothing, tools and cosmetics, and complaints in each are lexically distinct: a failing battery, a stale flavour, a poor fit and a distorted speaker share almost no vocabulary. HDBSCAN, clustering over embeddings, finds dense regions wherever such vocabulary clusters, and Amazon's product diversity creates many small dense regions. Yelp is semantically narrow by comparison — restaurants, bars and salons, where complaints concentrate on service, waiting, staff conduct and food quality. Many documents describing few underlying concerns produce few large clusters.

The merge to 24 therefore performs different work on each platform. For Amazon it is genuine consolidation, collapsing product-specific complaint vocabularies into transferable categories: separate topics about batteries, screens and connectivity become one category concerning product performance. For Yelp it is closer to relabelling, since 18 topics were already near the target. The compression ratio differs by almost a factor of four, and the taxonomy consequently fits Yelp's structure better than Amazon's.

Three implications follow, and all three are visible in the results. First, the catch-all behaviour identified in Section 5.2.3 is a predictable consequence rather than a surprise: "Product Performance Failures" absorbs 28.2% of held-out reviews because it is the merged descendant of many distinct Amazon topics that had little in common beyond being product faults. Second, imposing a single target count across platforms of different intrinsic granularity is questionable — a count derived per platform, or a hierarchical taxonomy with Amazon categories nested more deeply, would fit the data better. Third, the reduction from 121 to 24 discarded distinctions a business would care about: knowing that complaints concern battery life specifically is more actionable than knowing they concern product performance generally.

The practical reading is that the level of aggregation should follow the decision the taxonomy supports. For executive reporting, 24 categories is appropriate. For a product team investigating a specific failure it is too coarse, and the 69 Amazon topics discarded in the merge were closer to what such a team needs.

### 5.2.3 Where it failed, and why

Three defects are more informative than the headline figures, and none is softened here.

First, 27.6% of held-out reviews fell below the confidence threshold and were left unclassified. This is a coverage failure, though a visible one: the threshold was set precisely so that weak matches would be recorded rather than forced into the nearest category. Had it been removed, those 124 reviews would have been distributed among existing categories and the taxonomy would have appeared to work considerably better than it does. The design decision that produced the unflattering number is the same decision that makes the number trustworthy.

Second, the largest category absorbed 28.2% of the sample while showing among the lowest within-category coherence at 0.3997. The inverse relationship between size and coherence across categories — 0.6763 for an eleven-review category against 0.3997 for a 127-review one — indicates that "Product Performance Failures" functions as a catch-all, accumulating loosely related complaints because no more specific category fits. Taken with the unclassified rate, this describes a taxonomy that is simultaneously too coarse in one region and too sparse elsewhere: two buckets hold 55.8% of the sample while three categories hold one review each. The target of 24 categories was chosen in advance rather than derived from the data, and this result suggests that choice was not optimal.

Third, and most seriously, the taxonomy has no mechanism for rejecting non-complaints. It was constructed from negative reviews only, but the assignment stage was applied to reviews of all sentiments, and the results show positive reviews entering complaint categories: one category received 19 positive reviews and zero negative, and the unclassified group holds 60 positive against 34 negative. The zero-shot classifier scores relative fit among candidate categories; it has no notion of whether a text is a complaint at all. A sentiment filter before assignment would address this directly, and its absence is a design error rather than a limitation of the method — a distinction worth preserving, since it means the finding is correctable rather than fundamental.

A fourth, smaller defect indicates a pipeline reliability issue: one category retained an un-renamed raw BERTopic label ("Spray Bottle, Continuous Spray, Dryer Diffuser, Fit Dryer"), meaning the language-model naming step failed silently on at least one cluster and no validation caught it before the taxonomy was used. That the same category is also the one receiving 19 positive and zero negative reviews suggests the two defects may share a cause: a cluster that resisted naming may have been incoherent to begin with.

### 5.2.4 The evaluation problem itself

These conclusions rest on unsupervised measures, and their weakness deserves acknowledgement rather than a footnote. Section 2.1.3 reviewed evidence that automated coherence metrics correspond to human judgement more weakly and more variably than their routine use implies (Hoyle et al., 2021; Doogan & Buntine, 2021; Lim & Lauw, 2024), and that they can favour topics humans find obscure.

The approach taken here — several imperfect indicators read together, plus a manual validation sample — follows the direction that critique points in, but does not escape the underlying problem. The coherence figures reported are embedding-based and inherit whatever biases the embedding model carries; the same MiniLM model produced both the clusters and the measure of their quality, which is a circularity worth naming.

This is why the argument in Sections 5.2.2 and 5.2.3 rests where it does. The strongest evidence against the taxonomy is not the coherence values but the structural observations: the unclassified rate, the size distribution, the sentiment breakdown and the failed naming step are all directly countable and depend on no contested metric. Had this study reported only mean coherence of 0.6334, it would have described a working taxonomy. The multi-indicator design is what prevented that conclusion, and this is the clearest instance in the dissertation of an evaluation choice changing a finding.

---

## 5.3 Retrieval and Generation

### 5.3.1 Retrieval

Dense retrieval was 5.9 times faster than TF-IDF at identical Precision@3 of 0.90. The honest reading is narrower than it might appear. On these 30 queries the embedding retriever does not find relevant reviews more often; it finds them faster. The similarity scores cannot support a quality claim either, since cosine similarity over dense and sparse vectors measure different quantities.

The result is nonetheless useful, because latency determines whether interactive question answering is viable. The case for dense retrieval here is architectural rather than a claim about accuracy, and stating it that way is more defensible than asserting a quality advantage the data does not show. With 30 queries, a modest genuine difference in precision could also have passed undetected, so the equality of Precision@3 should be read as "no difference demonstrated" rather than "no difference exists".

### 5.3.2 The relevancy comparison, and what the per-question data reveals

On mean answer relevancy the standalone model scored 0.751 against the retrieval-augmented pipeline's 0.671, and did so in roughly a third of the time — 669.6 ms against 2,022.6 ms. On that single metric, retrieval made the system worse.

The per-question breakdown complicates this in both directions, and the honest account requires reporting both.

The standalone model's per-question relevancy is 0.988, 0.000, 0.961, 0.952 and 0.853. It therefore wins on four of the five questions, several by a wide margin — 0.988 against 0.743, 0.961 against 0.579, 0.952 against 0.570. Its mean is dragged down to 0.751 by a single complete failure on the delivery question, where it scored zero. Excluding that one score would place it near 0.938. The unconstrained model is genuinely more responsive on most questions, and reporting only the means would understate that.

But the same data shows the two systems failing differently. The pipeline's relevancy spans 0.570 to 0.743, a range of 0.17; the baseline's spans 0.000 to 0.988, a range of 0.99. The baseline has the higher ceiling and the lower floor. On the one question where it collapsed entirely, the grounded pipeline scored 0.728 — its second-highest score of the set.

Two conclusions follow, and neither is the simple one. First, the mean comparison favouring the baseline is real but rests on a distribution the mean does not describe: a practitioner choosing between these systems is choosing between higher average responsiveness with occasional total failure, and lower average responsiveness with a floor. Second, five questions is far too few to establish which pattern generalises. One anomalous score in a set of five moves the headline by nearly two tenths, which is itself an argument about sample size rather than about the systems.

What retrieval demonstrably provides is measurable grounding. Faithfulness of 0.849 indicates that most claims in the pipeline's answers follow from the retrieved evidence, and the metric is simply undefined for the baseline, since there is no context for an answer to be faithful to. That absence is the finding, not a gap in the evaluation: the baseline's fluent answers rest on nothing checkable, and its relevancy advantage is earned precisely by not being constrained to evidence. A study reporting only answer relevancy would have concluded that retrieval augmentation degrades performance, which is the concrete demonstration of why Section 2.2.3 argued for four metrics rather than one.

Self-enhancement bias cannot explain the relevancy result away either. Because the same model generates and judges, bias would inflate the pipeline's scores rather than the baseline's, so the direction of that threat runs against the observed result rather than producing it.

### 5.3.3 Retrieval, not generation, is the pipeline's weak component

The component-level metrics isolate where the pipeline underperforms, and the answer is not where prompt-engineering attention usually goes.

Context precision averages 0.503 across the five questions, with a range from 0.000 to 0.867. On the airline question it scored zero: by this measure none of the retrieved reviews was relevant, and yet the answer generated from that context still scored 0.917 on faithfulness and 0.579 on relevancy. Faithfulness at 0.849 against context precision at 0.503 describes a system that is being faithful to mediocre evidence — generating honestly from what it was given, while what it was given was often not the best available.

Context recall shows a different pattern: 1.000 on four questions and 0.000 on the fifth. That fifth is "Summarise the major customer complaints", the only open-ended question in the set, with no specific topic to retrieve against. A similarity-based retriever has nothing to match on when the query is a request for breadth rather than a question about a subject, which is a structural limitation of dense retrieval rather than a tuning failure.

This is exactly the diagnostic that Yu et al. (2024) argue aggregate RAG scores cannot provide. Had this evaluation reported a single composite score, the natural response would have been to revise the prompt. The component metrics point elsewhere: generation is working reasonably and retrieval is the constraint, so effort would be better spent on retrieval depth, reranking, or query reformulation for open-ended questions. The ablation suite described in Section 3.6.6 was designed to test exactly these variables and was not executed, so this remains a direction indicated by the evidence rather than a measured conclusion.

### 5.3.4 What the generation results cannot establish

The generation evaluation is the weakest part of this study, and its limits should be stated plainly rather than hedged.

Five questions is too few. Per-question faithfulness ranged from 0.625 to 1.000; context precision included both 0.000 and 0.867; context recall was 0.000 on one of five and 1.000 on the other four. With that variance and that sample, the specific values should not be quoted as measurements of system quality, and no confidence interval is reported because none would be meaningful. What the evidence supports is the qualitative finding — that relevancy and faithfulness diverge, that grounding is measurable for one system and undefined for the other, and that retrieval rather than generation is the weaker component — not any precise figure.

The judge model is the second constraint. Using the generating model as evaluator is a known threat to validity, catalogued in the survey literature reviewed in Section 2.1.2 (Gu et al., 2024; Li et al., 2024). It was accepted because free-tier rate limits made an independent judge impractical within the project's resources, and it bounds confidence in the RQ2 and RQ4 conclusions.

---

## 5.4 Comparison with Existing Research

### 5.4.1 Where the findings agree

The overall ordering of methods matches the literature. Fine-tuned transformers performed best, classical models next, lexicon methods worst — consistent with the comparative studies on Amazon review data reviewed in Section 2.1.2. BERTopic produced usable topics from short informal text where LDA is documented to struggle, consistent with Turan et al. (2024) and Mishra (2026).

### 5.4.2 Benchmarking against published figures

Comparing absolute numbers with published work requires care, because task definition differs more than model choice does. Table 5.1 sets this study's results beside reported figures, with the differences in task noted.

**Table 5.1** — This study against published benchmarks

| Source | Task | Dataset | Best reported | This study |
|---|---|---|---|---|
| Areshey & Mathkour (2023) | Binary polarity | Yelp | RoBERTa 98.30%, BERT 97.40%, DistilBERT 96.00% | — |
| This study, binary re-evaluation | Binary, neutral removed | Three platforms | — | DistilBERT macro F1 0.9533, RoBERTa 0.9341 |
| Financial sentiment study (2023) **[V-fin: citation to complete]** | Three-class | Financial text | BERT 87% accuracy, macro F1 0.85 | DistilBERT 84.37% accuracy, macro F1 0.8375 |
| Turan et al. (2024) **[V9]** | Topic quality | Amazon | BERTopic most interpretable of three | Centroid separation 0.2831 |
| Indonesian complaint study (2024) **[V12]** | Topic coherence | Customer complaints | LDA 0.68, BERTopic 0.51 | Within-category coherence 0.6334 |

Three comparisons are worth drawing out.

The apparent gap against the Yelp binary benchmarks — 96–98% there against 84% here — is almost entirely a task difference, not a modelling shortfall. Those studies classify polarity on two classes with the ambiguous middle removed. When this study removes the same class, DistilBERT reaches 0.9533 macro F1 and RoBERTa 0.9341, which sits in the same region. The comparison is a useful corrective to reading benchmark leaderboards as achievable targets: adding a third class costs roughly ten points, and much of the published literature's headline performance depends on having excluded the hardest cases.

Against a genuine three-class benchmark the correspondence is close. A fine-tuned BERT reported 87% accuracy and 0.85 macro F1 on three-class financial sentiment; DistilBERT here achieved 84.37% and 0.8375 on three-class review sentiment across three platforms. Given that this corpus is multi-domain and rating-labelled rather than expert-annotated, performance within three points of a single-domain benchmark suggests the pipeline is not underperforming relative to the field.

The taxonomy comparison is less favourable and is read as such. The Indonesian complaint study reported BERTopic coherence of 0.51 against LDA's 0.68, and this study's within-category coherence of 0.6334 is not directly comparable, since it uses embedding similarity rather than C_v. What both indicate is that BERTopic's coherence advantage over LDA is not established. Its benefit lies in handling short informal text and producing separable clusters, and the coherence literature reviewed in Section 2.1.3 casts doubt on whether these metrics measure interpretability at all — which is a reason to avoid resting any claim on the comparison in either direction.

### 5.4.3 Where they diverge, and why

The divergence is instructive rather than contradictory. Published comparisons report transformers outperforming classical models; this study finds a pretrained transformer 23 points of macro F1 below a linear model in aggregate. The difference is explained by experimental design, not by disagreement about the underlying methods. Those studies evaluate within a single domain, usually Amazon reviews, where a domain-matched or domain-neutral model is tested on data resembling its pretraining distribution. This study tests a Twitter-pretrained model on Amazon and Yelp prose and measures what happens.

The cross-domain sentiment literature reviewed in Section 2.1.2 treats exactly this problem as serious enough to warrant purpose-built architecture — feature projection, adversarial domain adaptation, multi-source attention. That literature and the applied comparison literature rarely meet, which is the gap this study occupies. It takes the applied position of using off-the-shelf models, but tests the transferability assumption that applied comparisons leave implicit, and finds it does not hold. The result is therefore evidence for the cross-domain literature's premise, arrived at from the applied literature's method.

Similarly, the relevancy comparison is not a contradiction of the RAG literature but an instance of the evaluation problem that literature has begun to identify. Yu et al. (2024) argue that aggregate RAG scores cannot diagnose whether failure originates in retrieval or generation; Section 5.3.3 is a worked example of component-level metrics doing exactly the diagnostic work they call for.

### 5.4.4 What is claimed as a contribution

Three contributions can be claimed without overstatement, and are deliberately stated at MSc level rather than inflated.

The stage-wise cross-domain comparison is the principal one. The literature contains many single-domain comparisons and a separate body of work on domain adaptation, but comparisons of the same off-the-shelf methods across three registers with per-platform and per-class reporting did not appear in the literature reviewed here. The finding that the ranking reverses by platform is a direct product of that design.

The diagnostic method is the second. Rather than reporting an anomalous result and speculating, this study tested the proposed explanation through three independent experiments — binary re-evaluation, per-platform breakdown and threshold sweep — that make the mechanism demonstrable and, in VADER's case, rule out the intuitive explanation. This is a modest methodological contribution, but a real one, since the reporting conventions criticised in Section 2.1.3 would have concealed the finding entirely.

The third is negative and correspondingly honest: the demonstration that a complaint taxonomy built from negative reviews will assign positive reviews to complaint categories unless a filter intervenes, and that a multi-indicator evaluation catches this where a single coherence score does not. No novel algorithm is proposed, and none is claimed.

---

## 5.5 Computational Cost and Deployment Trade-offs

Accuracy is one axis of method selection and rarely the binding one in practice. This section sets out what each stage costs and what that implies for deployment, since a method that wins on macro F1 but cannot be served within budget is not the method a practitioner should choose.

### 5.5.1 Cost of the sentiment stage

**Table 5.2** — Cost profile of the five sentiment classifiers

| Model | Parameters | Training cost | Artefact size | Hardware to serve | Interpretability |
|---|---|---|---|---|---|
| VADER | None (lexicon) | None | ~1 MB lexicon | CPU, trivial | High — rules inspectable |
| Naive Bayes | ~vocabulary size | Seconds, CPU | 76 MB (with vectoriser) | CPU | Moderate — feature log-probabilities |
| Logistic Regression | ~vocabulary size | Minutes, CPU | shared with above | CPU | Moderate — signed coefficients |
| RoBERTa (pretrained) | ~125M | None (used as-is) | ~500 MB | GPU preferred | Low |
| DistilBERT (fine-tuned) | ~66M | 10,545 s (2 h 56 m), GPU | 257 MB | CPU viable, GPU preferred | Low |

The distillation choice is vindicated on these figures. DistilBERT carries roughly 40% fewer parameters than BERT-base while retaining about 97% of its performance on GLUE and running around 60% faster (Sanh et al., 2019), which is what made fine-tuning feasible on freely available hardware within this project. Selecting BERT or RoBERTa for fine-tuning would have roughly doubled both training time and served model size for a gain the evidence does not suggest would have been large.

The more interesting comparison is between logistic regression and DistilBERT. The fine-tuned model gains around two points of macro F1 on the shared test set for a 257 MB artefact, nearly three hours of GPU training and materially higher inference cost. Logistic regression trains in minutes on a CPU and serves from a small sparse model. For a system processing a few thousand reviews daily on modest infrastructure the classical model is the defensible engineering choice — and Section 5.1.5 showed the transformer's advantage concentrates on the ambiguous neutral class that many applications do not need resolved.

The counter-argument is Section 5.1.4's stability finding. Logistic regression's macro F1 spans 0.6434 to 0.7948 across the three platforms; DistilBERT's spans 0.8301 to 0.8505. An organisation processing a single feedback stream can reasonably prefer the linear model; one processing several cannot, and the cost calculation changes accordingly.

Two caveats bound this analysis. Per-review inference latency was not measured for the five models, so the comparison rests on parameter counts and published throughput figures rather than timings from this system. That is a gap, and adding a timing pass to the diagnostics script would close it cheaply. And interpretability is assessed qualitatively: no formal explanation method such as SHAP or LIME was applied, so the claim that classical models are more interpretable rests on the transparency of their parameters rather than on a measured comparison of explanation quality.

### 5.5.2 Cost of retrieval and generation

The retrieval and generation stages carry costs of a different kind, dominated by storage and network latency rather than training.

Indexing 642,692 reviews at 384 dimensions produces a raw embedding matrix of approximately 0.99 GB, and the artefacts on disk are substantial. Storage scales linearly with corpus size, so a corpus ten times larger would need roughly 10 GB before any compression. This is the practical argument for the small embedding model: moving to a 768-dimensional model such as `all-mpnet-base-v2` would double the index footprint and roughly double query time for a retrieval quality gain that Section 5.3.1 suggests would be difficult to demonstrate, given that Precision@3 was already identical between dense and sparse retrieval.

At query time the dominant cost is generation, not retrieval. Retrieval takes 21.71 ms, while the end-to-end retrieval-augmented answer takes 2,022.6 ms against the standalone model's 669.6 ms. Retrieval accounts for roughly one per cent of the total; the additional 1,353 ms comes from the larger prompt the model must process once five reviews are inserted, plus the extra call made by the history-aware retriever. The cost of grounding is paid in generation tokens rather than in search.

That has a direct implication for deployment. Grounding roughly triples response time and increases token consumption per query, which on a metered API translates into a proportional cost increase. Whether that is worth paying depends on the use case: for exploratory analysis where a wrong answer is noticed quickly it may not be; for reporting that informs a decision, the verifiability it buys is difficult to obtain any other way.

### 5.5.3 Deployment implications

Taken together the cost profile suggests a tiered architecture rather than a single model choice. High-volume routine classification is well served by logistic regression on CPU where feedback arrives from one register, and by the fine-tuned transformer where it arrives from several, since cross-platform stability is what that model actually buys. The retrieval-augmented component is inherently interactive and low-volume, so its latency is tolerable, but its per-query cost places it in an analyst-facing tool rather than in an automated pipeline processing every incoming review.

None of this was tested in deployment, which limits how firmly it can be asserted. The figures above are measured costs and published throughput claims, not observations of a served system under load.

---

## 5.6 Critical Evaluation

### 5.6.1 Strengths

The comparison rests on a large, genuinely multi-domain corpus with a stratified, seeded split, and the sentiment stage is evaluated on 128,539 held-out reviews. Reporting is at class and platform level throughout, which is what made the central finding visible. Baselines are present at every stage where one was possible: a lexicon floor for sentiment, TF-IDF for retrieval, and an unconstrained model for generation. Differences between models are tested for statistical significance rather than asserted from a ranking, which identified one pair that cannot be separated and prevented an overclaim. Where results were unfavourable or contradicted expectation — the taxonomy's coverage, the relevancy comparison, the mediocre context precision — they are reported and explained rather than omitted.

### 5.6.2 Limitations

Several limitations bound what this study can claim, and they are of unequal severity.

The most serious concerns experimental control. The fine-tuned model was trained against a separately generated corpus version, so a portion of the shared sentiment test set overlaps data it had seen, and the exact extent cannot now be recovered because the earlier corpus version was not retained. The clean figure from the model's own held-out partition is reported as definitive, but the consequence is that the five-model comparison is not as controlled as the design intended. A single canonical split, generated once and reused by every model, would remove the objection entirely and is the first correction Chapter 6 recommends.

Label quality bounds everything downstream. Deriving sentiment from star ratings means a four-star review containing a complaint is labelled positive, and the neutral class carries different meanings across platforms — a three-star rating on two, a human judgement of affective flatness on the third. Some part of every model's neutral-class difficulty is attributable to the labels rather than to the models, and this study cannot separate the two.

The generation evaluation rests on five questions with a self-grading judge, as discussed in Section 5.3.4. This is the weakest evidence in the study and the conclusions drawn from it are correspondingly qualified.

Complaint categorisation has no ground truth, so no accuracy can be reported for it, and the unsupervised measures used instead are contested in the literature and partly circular, since the embedding model that formed the clusters also measures their quality.

Finally, the ablation experiments implemented in `scripts/evaluate_rag.py` — varying retrieval depth, retriever type, prompt design and metadata tagging — were not executed within the project timeframe because of API rate limiting. Their absence means the contribution of individual design decisions in the retrieval stage is argued rather than measured. Given that Section 5.3.3 identifies retrieval as the pipeline's weak component, this is the most consequential of the missing experiments.

### 5.6.3 Critical evaluation of the methods

Judged on outcomes, some choices held up better than others.

BERTopic was the right choice for the discovery stage, and the per-platform decision was correct. The fixed target of 24 categories was not: the size distribution and unclassified rate both indicate the number should have been derived from the data, and the platform asymmetry in Section 5.2.2 suggests a hierarchical structure would have served better than a flat one.

The zero-shot assignment approach is sound in principle and avoids needing labelled examples for newly discovered categories, but applying it without a preceding sentiment filter was an error that the sentiment-by-category results expose plainly.

The prompt-level grounding constraints and metadata tagging appear to work, in that faithfulness reached 0.849, but without the ablation results this cannot be attributed to those specific mechanisms with confidence. The faithfulness figure shows the pipeline as a whole is reasonably grounded; it does not isolate which design decision produced that, and the honest position is that the mechanism remains untested.

Using a free-tier hosted model was a resource-driven decision whose consequences propagated further than anticipated. It constrained the evaluation to five questions, prevented an independent judge, and ultimately prevented the ablations from running. In retrospect, budgeting a small amount for API access would have strengthened the evaluation more than any modelling change available within the same effort.

---

## 5.7 Assessment Against Objectives and Research Questions

**RQ1 — How does sentiment classification performance differ across method families on multi-domain feedback?** Answered. Fine-tuned DistilBERT performed best at 0.8375 macro F1 on clean held-out data, followed by logistic regression at 0.8033, Naive Bayes at 0.7580, pretrained RoBERTa at 0.5694 and VADER at 0.4381. Significance testing on the diagnostic sample supports most of this ordering but not all of it: nine of ten pairwise differences hold at p < 0.05, while Naive Bayes and RoBERTa cannot be separated at p = 0.82. The substantive answer is that the ordering is not stable in a second and more important sense — on Twitter Airline the pretrained transformer outperforms both classical models, so the ranking depends on the match between pretraining and target domain. Fine-tuning's clearest benefit is cross-domain consistency, not peak aggregate accuracy.

**RQ2 — How effectively does RAG improve complaint exploration compared with a standalone LLM?** Partially answered, and the qualification matters. RAG achieved 0.849 faithfulness with source attribution, while the baseline scored higher on mean answer relevancy at 0.751 against 0.671 and answered roughly three times faster. The improvement RAG provides is verifiability, not responsiveness. The per-question data adds a second finding: the baseline's advantage rests on four strong scores and one complete failure, giving it both the higher ceiling and the lower floor, while the grounded pipeline varies far less. With five questions, a self-grading judge and a mediocre context precision of 0.503, this conclusion is directional rather than precise.

**RQ3 — Can BERTopic produce a coherent cross-domain complaint taxonomy?** Answered with substantial qualification. It produced 24 well-separated categories from 121 raw topics at 0.2831 centroid separation, with 0.6334 mean within-category coherence on held-out data. But 27.6% of reviews were left unclassified, one category absorbed 28.2% of the sample with among the lowest internal coherence recorded, and the taxonomy cannot distinguish complaints from non-complaints. The answer is that BERTopic can produce a usable cross-domain taxonomy, but not without a consolidation stage, a data-driven rather than fixed category count, and a sentiment filter at assignment.

**RQ4 — To what extent are insights traceable to and verifiable against source reviews?** Answered. Every generated answer returns the identifiers of the reviews retrieved to produce it, and faithfulness of 0.849 quantifies how far answers follow from that evidence. Traceability is demonstrated as an architectural property rather than asserted. The figures also mark the limit: at 0.849 faithfulness, roughly one claim in six was not fully supported by the retrieved context, and at 0.503 context precision the evidence retrieved was frequently not the best available. Traceability being available is therefore not the same as every statement being grounded. Whether that traceability is useful in practice is not established and would require the user study identified as future work.

Objectives 1 to 5 were met. Objective 6 was met partially: stage-appropriate metrics and baseline comparisons were delivered at every stage, but the ablation component was implemented and not executed. Objective 7 is delivered by this chapter and the next.

---

## 5.8 Practical Implications

The preceding sections judged the methods. This one asks what an organisation could do with the results, since a comparative study that never reaches that question leaves its own motivation unanswered.

**On method selection (Objective 3, RQ1).** The clearest practical finding is that benchmark scores do not transfer. A practitioner choosing a sentiment model on published accuracy would likely select a pretrained transformer and, on Yelp or Amazon prose, obtain worse performance than a linear model trained in minutes. The selection rule this study supports is narrower and more useful: check whether the candidate model's label definition matches your own before considering its reported accuracy, and if feedback arrives from several sources, prefer the model with the smallest variance across them rather than the highest mean.

**On the taxonomy (Objective 4, RQ3).** The taxonomy is usable for executive-level reporting, where 24 categories at 0.2831 separation give a defensible summary of what customers complain about across platforms. It is not usable as an automated routing or triage mechanism in its current state, and the reason is specific: with 27.6% of reviews unclassified and one category absorbing 28.2%, routing on it would leave a quarter of complaints unhandled and overload a single queue. Two changes would address this — a sentiment filter before assignment, and a category count derived per platform rather than fixed at 24. Section 5.2.2 also implies a hierarchical structure would serve better than a flat one, with broad categories for reporting and the discarded finer topics retained beneath them for investigation.

**On the retrieval-augmented component (Objectives 5 and 6, RQ2 and RQ4).** Its practical value is not speed or fluency, both of which the standalone model wins on average. It is that every answer arrives with the reviews that produced it, so a claim can be checked in seconds before it informs a decision, and that its worst case is meaningfully better than the baseline's. That matters most where the cost of acting on a wrong conclusion is high and where the analyst is accountable for the recommendation. Where an analyst is exploring informally and will notice a wrong answer immediately, the standalone model is faster, cheaper and adequate. The faithfulness and context-precision figures also bound the claim: the system supports verification but does not remove the need for it.

**On the evaluation approach itself (Objective 6).** The most transferable contribution may be methodological. Reporting per-class and per-platform results rather than aggregates, testing the mechanism behind an anomalous result rather than reporting it and moving on, and reading several weak indicators together rather than one headline metric are what turned three apparently disappointing findings into the study's main results. That practice costs little and applies to any comparative evaluation.

---

## 5.9 Chapter Summary

The central finding is that method ranking in customer feedback analysis is conditional on domain. A pretrained transformer that appears clearly inferior on aggregate metrics outperforms both classical models on the platform matching its pretraining distribution, and its apparent weakness resolves into a single-class failure consistent with a mismatch between label definitions. Fine-tuning's principal value is stability across domains rather than peak accuracy, and the gap between it and a well-specified linear model is smaller than the literature's framing suggests.

The taxonomy results are more mixed and more critical of the approach taken. Consolidation produced well-separated categories, but coverage was incomplete at 27.6% unclassified, one category behaved as a catch-all, the naming step failed silently on one cluster, and the absence of a sentiment filter allowed positive reviews into complaint categories. The multi-indicator evaluation is what exposed all four; a single coherence score would have described a working system.

The generation results establish the qualitative case for retrieval augmentation — measurable grounding where the baseline offers none, and a higher floor when the baseline fails — while being too small and too self-referentially judged to support precise claims. The component metrics also locate the pipeline's weakness in retrieval rather than generation, which is where the unexecuted ablation would have been most valuable.

Chapter 6 draws these together, assesses the aim as a whole, and sets out the future work each limitation implies.
