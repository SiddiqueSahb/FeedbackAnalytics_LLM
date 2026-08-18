# Chapter 5 — Discussion

Chapter 4 reported what happened. This chapter argues about what it means: why the results took the form they did, how they sit against the literature reviewed in Chapter 2, and how far each objective was met. Section 5.1 interprets the sentiment findings, 5.2 the taxonomy, 5.3 the retrieval and generation results. Section 5.4 compares with existing research, 5.5 examines cost, 5.6 evaluates the study critically, and 5.7 assesses each research question.

---

## 5.1 Sentiment Classification

### 5.1.1 Why the pretrained transformer lost

RoBERTa scored 0.5694 macro F1 against logistic regression's 0.8033. Taken at face value this contradicts a well-established finding. It does not, and the diagnostic evidence explains why.

One qualification narrows the claim first. McNemar testing shows the gap to logistic regression is robust at p = 2.6 × 10⁻⁴, but the gap to Naive Bayes is not: on the 675 reviews where those two disagree, each is correct on almost exactly half, giving p = 0.82. The defensible statement is that the pretrained transformer underperformed the *stronger* classical model, not classical machine learning as a family — a distinction visible only because the comparison was tested rather than read off a table.

The failure is confined to one class. RoBERTa's per-class F1 is 0.743 negative and 0.740 positive, both respectable, against 0.225 neutral. The confusion matrices locate the error precisely: 85.7% of genuinely neutral reviews are pushed to a pole. Separating precision from recall sharpens this further — neutral precision is 0.522 while recall is 0.143. When the model does assign the label it is right about half the time; what it fails to do is assign it at all. A recall failure with intact precision is the signature of a decision boundary that never reaches the middle class.

Two experiments confirm rather than assert this. Removing the neutral class raises RoBERTa's macro F1 from 0.6669 to 0.9341, above logistic regression's 0.9216 — the expected ordering returns the moment the offending class is removed, which would not happen if the model were generally inferior. And the per-platform breakdown shows neutral F1 of 0.713 on Twitter Airline against 0.198 on Yelp: the same model handles the same class competently on one platform and fails on another.

The explanation follows from what the model is. `twitter-roberta-base-sentiment-latest` was pretrained on tweets for a scheme in which neutral means a human judged a post to express no clear sentiment (Barbieri et al., 2020). On Amazon and Yelp, neutral means a three-star rating — a compound judgement often containing both praise and complaint. A model trained to recognise affectively flat text will not recognise a mixed three-star review as neutral, because such a review is not flat but contradictory. The error analysis in Section 4.5.4 shows this concretely: the misclassified true-neutral cases carry explicit praise and explicit criticism together — "the tea is very good however the bags are fragile". A three-star rating encodes a net judgement across both clauses; a model whose neutral class means *no clear sentiment* has no representation for that and must choose a pole.

The evidence supports this interpretation rather than proving it. Establishing causation would require re-testing the same checkpoint against human-annotated neutral labels on review prose, which was not done. What can be said is that four independent observations — the recall-specific failure, the platform-conditional reversal, the recovery under binary evaluation, and the character of individual errors — are jointly consistent with label mismatch and hard to explain by general inferiority.

The practical implication is direct: an off-the-shelf sentiment model is only as transferable as its label definition, and no benchmark accuracy tells a practitioner whether that definition matches their data.

### 5.1.2 The ranking reverses by platform

The per-platform evidence is stronger than a diagnosis of one anomaly. On Twitter Airline, RoBERTa achieves 0.7853 macro F1 against logistic regression's 0.6434 and Naive Bayes's 0.4838 — the transformer wins decisively. On Yelp the ordering inverts, RoBERTa at 0.5544 against logistic regression's 0.7948. Naive Bayes shows the mirror image, 0.7952 on Amazon and 0.4838 on Twitter. Two models swap places entirely depending on which platform judges them.

No single ranking is correct. Which method is best depends on the relationship between training and target domain, and a study on one platform would have concluded confidently and incorrectly in either direction. This is the chapter's most substantive finding for RQ1, and it exists only because the comparison ran across three registers with per-platform reporting — precisely the design that the aggregate-metric convention criticised in Section 2.1.3 would have prevented.

### 5.1.3 Why VADER failed, and why the obvious explanation is wrong

VADER's macro F1 of 0.4381 against accuracy of 0.5142 is the widest gap between those measures of any model here, and the gap is the finding. Random guessing on this balanced three-class task scores about 0.333 on both, so VADER is well above chance; what it is not is evenly competent, and accuracy alone would have concealed that.

The intuitive explanation is miscalibrated thresholds: the default neutral band of ±0.05 is narrow, so few texts are assigned neutral. That is testable, and the threshold sweep largely rejects it. Widening the band raises macro F1 from 0.5094 to a maximum of 0.5353 at ±0.25 — a gain of 0.026 — after which it declines as the band swallows genuinely polar reviews. Threshold choice accounts for a modest fraction of the gap, not most of it.

The larger problem is the lexicon. VADER labels 40.4% of genuinely negative reviews positive, which is systematic failure of word-level scoring on mixed vocabulary. A one-star review reading "Worked well and looked good until keys stopped working after 3 months" scores +0.477, because the complaint is carried by temporal contrast rather than negative vocabulary. These errors sit confidently on the wrong side of the boundary rather than near it, which is why widening the band recovers so little.

The finding still supports VADER's role. As a baseline it establishes what rules alone achieve, and the roughly 40-point gap to the fine-tuned transformer quantifies the value of learning from data — a floor invisible had no untrained method been included.

### 5.1.4 What fine-tuning actually buys

The fine-tuned model reached 0.8375 macro F1 on clean held-out data, and is notably stable: 0.8448 on Amazon, 0.8301 on Yelp, 0.8505 on Twitter. Where the pretrained model varied by 23 points across platforms, this varies by two. That stability is the real return, and it is a different claim from the usual one. The gain over logistic regression on aggregate macro F1 is modest — roughly two points, though McNemar confirms it at p = 1.7 × 10⁻²¹. The gain in cross-domain consistency is larger and less often reported, and for a practitioner processing several feedback sources it matters more.

Explaining the gap to the classical models requires accounting for three mechanisms. TF-IDF discards sequence, so "not worth the money" and "worth the money, not..." produce identical vectors, while self-attention conditions each token on context and can represent negation scope (Devlin et al., 2019). Subword tokenisation handles unseen vocabulary that Naive Bayes cannot represent at all. But the third mechanism is the most instructive and concerns the classical family itself: Naive Bayes trails logistic regression by 4.5 points despite identical features, because its conditional independence assumption is plainly false for text. Roughly two thirds of the distance between Naive Bayes and DistilBERT is recovered simply by replacing one linear model with a better-specified linear model — the transformer architecture accounts for less of the gap than it appears to.

The per-class evidence locates the remainder: DistilBERT's gain over logistic regression concentrates in the neutral class, 0.747 against 0.718, precisely the case requiring contradictory evidence to be weighed within one review. The practical consequence follows directly. For a business separating satisfied from dissatisfied customers only, fine-tuning buys little — on binary re-evaluation the two models sit at 0.9533 and 0.9216, three points for three hours of GPU training and a 257 MB artefact.

---

## 5.2 Complaint Taxonomy

### 5.2.1 What worked, and the platform asymmetry that explains the rest

Consolidation worked at the level it was designed for: 121 raw topics reduced to 24 categories at 0.2831 centroid separation, with 0.6334 mean within-category coherence and 21 categories used. The per-platform discovery decision was vindicated — Yelp produced 18 topics from 149,838 documents while Amazon produced 69 from 57,931, so pooling would have let the larger corpus dominate and produced a taxonomy describing Yelp's concerns while claiming to describe feedback generally.

That asymmetry is the most informative structural result, and it is not a parameter artefact — minimum topic size was scaled to each platform's document count precisely to prevent that. The cause is semantic breadth. Amazon spans electronics, groceries, clothing and cosmetics, where a failing battery, a stale flavour and a distorted speaker share almost no vocabulary, so HDBSCAN finds many small dense regions. Yelp is narrow by comparison — restaurants, bars, salons, where complaints concentrate on service, waiting and food quality — so many documents describe few underlying concerns and produce few large clusters.

The merge therefore does different work on each platform: genuine consolidation for Amazon, closer to relabelling for Yelp, with compression ratios differing by almost a factor of four. The taxonomy consequently fits Yelp's structure better than Amazon's, and three consequences follow. The catch-all behaviour below is predictable rather than surprising. Imposing one target count across platforms of different intrinsic granularity is questionable. And the reduction discarded distinctions a business would care about — knowing complaints concern battery life is more actionable than knowing they concern product performance. The level of aggregation should follow the decision it supports: 24 categories suits executive reporting, but a product team investigating a specific failure needs the 69 Amazon topics the merge discarded.

### 5.2.2 Where it failed

Three defects are more informative than the headline figures, and none is softened.

First, 27.6% of held-out reviews fell below the confidence threshold and were left unclassified. This is a coverage failure, but a visible one: the threshold was set so weak matches would be recorded rather than forced. Had it been removed, those 124 reviews would have been distributed among existing categories and the taxonomy would have appeared to work considerably better than it does. The decision producing the unflattering number is the same decision making the number trustworthy.

Second, the largest category absorbed 28.2% of the sample while showing among the lowest coherence at 0.3997. The inverse relationship between size and coherence — 0.6763 for an eleven-review category against 0.3997 for a 127-review one — indicates a catch-all accumulating loosely related complaints because nothing more specific fits. Two buckets hold 55.8% of the sample while three categories hold one review each: a taxonomy simultaneously too coarse in one region and too sparse elsewhere. The target of 24 was chosen in advance rather than derived from the data, and this suggests that choice was not optimal.

Third, and most seriously, the taxonomy cannot reject non-complaints. It was built from negative reviews but applied to reviews of all sentiments, and positive reviews duly entered complaint categories: one category received 19 positive and zero negative, and the unclassified group holds 60 positive against 34 negative. The zero-shot classifier scores relative fit among candidates; it has no notion of whether a text is a complaint at all. A sentiment filter before assignment would fix this, which makes it a design error rather than a limitation of the method — a distinction worth keeping, since it means the finding is correctable.

A fourth, smaller defect signals a reliability issue: one category retained an un-renamed raw BERTopic label, meaning the naming step failed silently and no validation caught it. That the same category received 19 positive and zero negative reviews suggests both defects may share a cause — a cluster that resisted naming may have been incoherent to begin with.

### 5.2.3 The evaluation problem itself

These conclusions rest on unsupervised measures whose weakness deserves acknowledgement. Section 2.1.3 reviewed evidence that automated coherence corresponds to human judgement more weakly than routine use implies (Hoyle et al., 2021; Doogan and Buntine, 2021; Lim and Lauw, 2024). The approach taken here — several imperfect indicators plus a manual sample — follows that critique but does not escape it: the coherence figures are embedding-based, and the same MiniLM model produced both the clusters and the measure of their quality, which is a circularity worth naming.

This is why the argument above rests where it does. The strongest evidence against the taxonomy is not the coherence values but the structural observations — unclassified rate, size distribution, sentiment breakdown, failed naming step — all directly countable and dependent on no contested metric. Reporting only mean coherence of 0.6334 would have described a working taxonomy. The multi-indicator design prevented that conclusion, and this is the clearest instance in the dissertation of an evaluation choice changing a finding.

---

## 5.3 Retrieval and Generation

### 5.3.1 Retrieval

Dense retrieval was 5.9 times faster than TF-IDF at identical Precision@3 of 0.90. The honest reading is narrow: on these 30 queries the embedding retriever does not find relevant reviews more often, it finds them faster. The similarity scores cannot support a quality claim either, since cosine similarity over dense and sparse vectors measures different quantities. The result is still useful, because latency determines whether interactive question answering is viable — but the case for dense retrieval here is architectural rather than a claim about accuracy. With 30 queries a modest genuine difference could also pass undetected, so equal Precision@3 should be read as "no difference demonstrated" rather than "no difference exists".

### 5.3.2 The relevancy comparison

On mean answer relevancy the standalone model scored 0.753 against the pipeline's 0.610, in under half the time — 734.0 ms against 1,582.7 ms. On that metric, retrieval made the system worse.

The per-question breakdown complicates this in both directions. The baseline's scores are 0.988, 0.000, 0.961, 0.964 and 0.853: it wins four of five, several by wide margins, and its mean is dragged down by one complete failure on the delivery question. Excluding that would place it near 0.941. The unconstrained model is genuinely more responsive on most questions, and reporting only means would understate that.

But the same data shows the two failing differently. The pipeline's four scored questions span 0.437 to 0.819, a range of 0.38; the baseline's spans 0.99. The baseline has the higher ceiling and the lower floor — on the question where it collapsed entirely, the grounded pipeline scored 0.740, its second-highest. A practitioner choosing between them is choosing between higher average responsiveness with occasional total failure, and lower average responsiveness that varies with retrieval quality. Five questions is far too few to establish which pattern generalises; one anomalous score moves the baseline's headline by nearly two tenths, which is an argument about sample size rather than about the systems.

What retrieval demonstrably provides is measurable grounding. Faithfulness of 0.540 indicates a majority of claims follow from retrieved evidence, and the metric is simply undefined for the baseline, since there is no context for an answer to be faithful to. That absence is the finding, not a gap: the baseline's fluent answers rest on nothing checkable, and its relevancy advantage is earned precisely by not being constrained to evidence. A study reporting only answer relevancy would have concluded that retrieval augmentation degrades performance — the concrete demonstration of why Section 2.2.3 argued for four metrics rather than one. Self-enhancement bias cannot explain the result away either: because the same model generates and judges, bias would inflate the pipeline's scores rather than the baseline's, so the threat runs against the observed result.

### 5.3.3 Three components, three different verdicts

The component metrics isolate where the pipeline underperforms, and the answer is not where prompt-engineering attention usually goes. It is also not one story: one part of this system needs a genuinely different fix.

Context precision averages 0.416, falling to zero on two questions. These two zeros look identical in the table but come from different places, and treating them as one finding would be a mistake. On the airline question retrieval genuinely happened and genuinely struggled — two thin, largely uninformative reviews cleared the threshold, and faithfulness fell with them to 0.400 rather than staying artificially high as it did in an earlier, un-gated version of this pipeline, where a comparable zero-precision question still scored 0.917. That earlier result described a system being faithful to evidence its own retrieval had judged irrelevant. The grounding threshold closes exactly that gap here.

The customer-service question is a different matter, and pinning it on the threshold would credit the wrong component. Retrieval was never attempted. The scope guard — the two-stage check meant to catch questions about weather or sport — rejected "What customer service issues appear most often?" as out of scope, which it plainly is not. The returned text matches the scope guard's fixed message, not the threshold's, and the same false rejection recurred identically across both conditions of the metadata-tagging ablation: three runs, one consistent misfire. That is a finding about the scope classifier's precision, not a demonstration of the grounding threshold succeeding.

Context recall is 0.000 on three of five questions for three different reasons. On the summarisation question it repeats the earlier finding — the only open-ended question, with no specific topic to retrieve against, a structural limitation of similarity-based retrieval rather than a tuning failure. On the airline question it reflects genuinely thin retrieval. On the customer-service question it is the mechanical consequence of the false rejection.

This is exactly the diagnostic Yu et al. (2024) argue aggregate RAG scores cannot provide, and it matters twice over here: once for separating grounded-but-mediocre answers from confident hallucination, and again for separating a retrieval weakness from a scope-classification bug that looks similar in the numbers. A single composite score would have prompted prompt revision; the actual diagnosis is narrower — retrieval quality needs work on thinly-covered questions, and the scope guard needs a precision fix that has nothing to do with retrieval.

### 5.3.4 What the generation results cannot establish

Five questions is too few. Per-question faithfulness ranged from 0.000 to 1.000, context precision from 0.000 to 1.000, context recall was 0.000 on three of five. The specific values should not be quoted as measurements of system quality, and no confidence interval is reported because none would be meaningful. What the evidence supports is qualitative: that relevancy and faithfulness diverge, that grounding is measurable for one system and undefined for the other, that retrieval is the weaker component among questions it attempts, and that a false rejection can look indistinguishable from a deliberate refusal in aggregate numbers while meaning something very different about how far to trust the system.

The judge model is the second constraint, and this run produced a direct instance rather than a theoretical one: the RAGAS judge raised an `OutputParserException` scoring the airline question's relevancy, so that score is missing rather than measured and the mean is computed over four questions. Using the generating model as evaluator is a known threat catalogued in Section 2.2.3 (Gu et al., 2024; Li et al., 2024). It was accepted because free-tier limits made an independent judge impractical, and it bounds confidence in the RQ2 and RQ4 conclusions.

---

## 5.4 Comparison with Existing Research

The overall ordering matches the literature — fine-tuned transformers best, classical next, lexicon worst — consistent with the comparative studies reviewed in Section 2.1.2, and BERTopic produced usable topics from short informal text where LDA is documented to struggle (Turan et al., 2024; Egger and Yu, 2022).

**Table 5.1** — This study against published work

| Source | Task | Dataset | This study |
|---|---|---|---|
| Areshey and Mathkour (2024) | Binary polarity, five transformers | Yelp | Binary re-evaluation: DistilBERT 0.9533, RoBERTa 0.9341 |
| Ali et al. (2024) | Three-way ML/DL/transformer | Amazon | Three-class, three platforms: DistilBERT 0.8375 |
| Sunar (2025) | Embedding comparison | Amazon Magazine Subscriptions | DistilBERT best in both |

The apparent gap against published Yelp benchmarks is a task difference, not a modelling shortfall, and this study shows that with its own data rather than by quoting someone else's. Binary studies remove the ambiguous middle class; removing it here — same corpus, same models, only the task definition changed — raises DistilBERT from 0.8375 to 0.9533 and RoBERTa from 0.6669 to 0.9341. Adding a third class costs this pipeline roughly twelve points of macro F1, so a substantial part of the headline performance in the binary literature depends on having excluded the hardest cases.

The divergence on cross-domain transfer is instructive rather than contradictory. Published comparisons report transformers outperforming classical models; this study finds a pretrained transformer 23 points below a linear model in aggregate. The difference is experimental design: those studies evaluate within one domain where a domain-matched model meets data resembling its pretraining distribution, whereas this study tests a Twitter-pretrained model on review prose and measures what happens. The cross-domain literature treats that as serious enough to warrant purpose-built architecture (Kong et al., 2023; Ganin et al., 2016), but rarely meets the applied comparison literature — which is the gap this study occupies, taking the applied position while testing the transferability assumption applied comparisons leave implicit.

The metadata-tagging ablation diverges more specifically. Section 2.2.3 noted that applied RAG work adopts grounding aids together and justifies them by reasoning, with the implicit expectation that they are close to free. The measured result is that tagging improves faithfulness by 0.118 while costing 0.202 on answer relevancy — a trade-off. The likely mechanism is mundane: tagged answers spend more length on attribution, and a relevancy judge scores directness. But it suggests studies reporting only faithfulness for a metadata-augmented system may be reporting the favourable half of a trade. This is one ablation on five questions, offered as an observation worth testing at scale rather than an established effect.

**What is claimed as a contribution.** Four things, stated at MSc level rather than inflated. The stage-wise cross-domain comparison is the principal one: comparisons of the same off-the-shelf methods across three registers with per-platform and per-class reporting did not appear in the literature reviewed here, and the finding that the ranking reverses is a direct product of that design. The diagnostic method is second — testing proposed explanations through independent experiments, one of which ruled out the intuitive explanation rather than confirming it. The same habit paid off again at the generation stage, where a zero that looked like the grounding threshold working turned out, on inspection of the returned text, to be a scope classifier misfiring; both cases make the point that a number consistent with your design working is not evidence that it worked. Third is the executed ablation, which converted a design justification into a measured effect with a cost attached. Fourth is negative and correspondingly honest: a complaint taxonomy built from negative reviews will assign positive reviews to complaint categories unless a filter intervenes, and a multi-indicator evaluation catches this where a single coherence score does not.

---

## 5.5 Cost and Deployment Trade-offs

Accuracy is one axis of method selection and rarely the binding one. Table 5.2 gives the cost profile.

**Table 5.2** — Cost profile of the five sentiment classifiers

| Model | Parameters | Training cost | Artefact | Serving | Interpretability |
|---|---|---|---|---|---|
| VADER | None | None | ~1 MB | CPU, trivial | High |
| Naive Bayes | ~vocabulary | Seconds, CPU | 76 MB | CPU | Moderate |
| Logistic Regression | ~vocabulary | Minutes, CPU | shared | CPU | Moderate |
| RoBERTa (pretrained) | ~125M | None | ~500 MB | GPU preferred | Low |
| DistilBERT (fine-tuned) | ~66M | 2 h 56 m, GPU | 257 MB | CPU viable | Low |

The distillation choice is vindicated: DistilBERT carries 40% fewer parameters than BERT-base while retaining about 97% of its performance (Sanh et al., 2019), which made fine-tuning feasible on freely available hardware. The more interesting comparison is against logistic regression — two points of macro F1 for a 257 MB artefact, three hours of GPU training and higher inference cost, where the linear model trains in minutes on CPU. For a few thousand reviews daily the classical model is the defensible engineering choice. The counter-argument is stability: logistic regression spans 0.6434 to 0.7948 across platforms where DistilBERT spans 0.8301 to 0.8505, so an organisation processing one feedback stream can prefer the linear model while one processing several cannot.

At query time the dominant cost is generation, not retrieval: 21.71 ms retrieval against 1,582.7 ms end-to-end, roughly one per cent of the total. The additional 848.7 ms over the baseline comes from the larger prompt and the history-aware retriever's extra call, and is avoided entirely when either guard rejects a question — the one rejection in this sample completed in 112 ms. Grounding therefore roughly doubles response time and increases token consumption proportionally, which on a metered API is a direct cost. Whether that is worth paying depends on the use case: for exploratory analysis where a wrong answer is noticed quickly it may not be; for reporting that informs a decision, the verifiability is difficult to obtain otherwise.

Two caveats bound this. Per-review inference latency was not measured, so the model comparison rests on parameter counts and published throughput rather than timings from this system. And interpretability is assessed qualitatively — no SHAP or LIME analysis was applied. Nothing here was tested in deployment.

---

## 5.6 Critical Evaluation

**Strengths.** The comparison rests on a large, genuinely multi-domain corpus with a stratified seeded split, evaluated on 128,539 held-out reviews. Reporting is at class and platform level throughout, which is what made the central finding visible. Baselines are present at every stage: a lexicon floor for sentiment, TF-IDF for retrieval, an unconstrained model for generation. Differences are tested for significance rather than asserted, which identified one pair that cannot be separated and prevented an overclaim. Unfavourable results are reported and explained rather than omitted.

**Limitations**, of unequal severity. The most serious concerns experimental control: the fine-tuned model was developed against a separately generated corpus version, so its shared-test-set figure is less tightly controlled than the other four, and the exact overlap cannot be recovered because the earlier version was not retained. The clean figure from the model's own held-out partition is reported as definitive throughout. Label quality bounds everything downstream — rating-derived labels mean a four-star review containing a complaint is labelled positive, and "neutral" means different things on different platforms, so some part of every model's neutral difficulty is attributable to labels rather than models. The generation evaluation rests on five questions with a self-grading judge and is the weakest evidence in the study. Categorisation has no ground truth, and the unsupervised measures used instead are contested and partly circular. Finally, three of four planned ablation conditions were not carried out, so the contribution of retrieval depth, retriever type and prompt design is argued rather than measured; given that Section 5.3.3 identifies retrieval quality as a genuine weak point, those remain the most consequential missing experiments.

**On the methods themselves.** BERTopic was the right choice and the per-platform decision was correct; the fixed target of 24 categories was not, and the platform asymmetry suggests a hierarchical structure would have served better than a flat one. Zero-shot assignment is sound in principle but applying it without a preceding sentiment filter was an error the results expose plainly. The prompt-level grounding constraints and metadata tagging were long defended only by reasoning; since one has now been tested, the honest position is that one of three design decisions has been measured and returned a trade-off, while the prompt rules and the threshold remain untested against each other. Using a free-tier hosted model was a resource decision whose consequences propagated further than anticipated — five questions, no independent judge, three ablations unrun. Budgeting a small amount for API access would have strengthened the evaluation more than any modelling change available within the same effort.

---

## 5.7 Assessment Against the Research Questions

**RQ1 — How does sentiment classification performance differ between lexicon-based, classical machine learning, pretrained transformer and fine-tuned transformer approaches on multi-domain customer feedback?** Answered. Fine-tuned DistilBERT performed best at 0.8375 macro F1 on clean held-out data, then logistic regression 0.8033, Naive Bayes 0.7580, pretrained RoBERTa 0.5694, VADER 0.4381. Significance testing supports most of this ordering: nine of ten pairwise differences hold at p < 0.05, while Naive Bayes and RoBERTa cannot be separated at p = 0.82. The substantive answer is that the ordering is not stable in a more important sense — on Twitter Airline the pretrained transformer outperforms both classical models, so the ranking depends on the match between pretraining and target domain. Fine-tuning's clearest benefit is cross-domain consistency, not peak aggregate accuracy.

**RQ2 — How effectively does retrieval-augmented generation improve complaint exploration compared with a standalone large language model?** Partially answered. RAG achieved 0.540 faithfulness with source attribution while the baseline scored higher on relevancy, 0.753 against 0.610, in under half the time. The improvement RAG provides is verifiability, not responsiveness. The per-question data adds that the baseline's advantage rests on four strong scores and one complete failure, giving it the higher ceiling and the lower floor. The metadata-tagging ablation adds a measured but mixed result: tagging helps faithfulness and costs relevancy rather than simply validating the design. With five questions, a judge that failed to parse on one, and context precision of 0.416, this conclusion is directional rather than precise.

**RQ3 — Can BERTopic produce a coherent cross-domain complaint taxonomy from heterogeneous customer reviews?** Answered with substantial qualification. It produced 24 well-separated categories from 121 raw topics at 0.2831 separation with 0.6334 coherence — but left 27.6% unclassified, allowed one category to absorb 28.2% of the sample at among the lowest coherence recorded, and cannot distinguish complaints from non-complaints. BERTopic can produce a usable cross-domain taxonomy, but not without a consolidation stage, a data-driven rather than fixed category count, and a sentiment filter at assignment.

**RQ4 — To what extent can retrieval-augmented answers be traced to, and verified against, the source reviews used to produce them?** Answered. Every answer returns the identifiers of the reviews producing it, and faithfulness of 0.540 quantifies how far answers follow from that evidence when the pipeline answers. Traceability is demonstrated as an architectural property rather than asserted. The one unanswered question demonstrates that the pipeline can decline rather than fabricate — a real safety property — but the specific decline came from the scope guard misjudging an in-scope question rather than the grounding threshold detecting insufficient evidence, so it shows the general principle rather than the specific mechanism. The figures bound the claim on questions that were answered: at 0.540 faithfulness roughly half of claims were not fully supported, and at 0.416 context precision the retrieved evidence was frequently not the best available. Traceability being available is not the same as every statement being grounded, and whether it is useful in practice would require the user study identified as future work.

Objectives 1 to 5 were met. Objective 6 was met partially: stage-appropriate metrics and baselines were delivered at every stage, and one of four planned ablation conditions was executed. Objective 7 is delivered by this chapter and the next.

---

## 5.8 Practical Implications and Summary

The clearest practical finding is that benchmark scores do not transfer. A practitioner selecting on published accuracy would likely choose a pretrained transformer and, on review prose, get worse performance than a linear model trained in minutes. The rule this study supports is narrower: check whether the candidate's label definition matches your own before considering its accuracy, and where feedback arrives from several sources prefer the smallest variance rather than the highest mean.

The taxonomy is usable for executive reporting but not for automated routing: with 27.6% unclassified and one category absorbing 28.2%, routing on it would leave a quarter of complaints unhandled and overload a single queue. The retrieval-augmented component's value is not speed or fluency, both of which the baseline wins on average — it is that every answer arrives with the reviews that produced it, so a claim can be checked before it informs a decision, and that its worst case is better.

The most transferable contribution may be methodological. Per-class and per-platform reporting, testing the mechanism behind an anomaly rather than reporting it and moving on, and reading several weak indicators together are what turned three apparently disappointing findings into this study's main results.

Chapter 6 draws these together, assesses the aim as a whole, and sets out the future work each limitation implies.
