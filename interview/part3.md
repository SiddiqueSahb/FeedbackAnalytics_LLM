# ROUND 3 — Machine Learning

### Q23. Your pretrained RoBERTa scored below logistic regression. Explain.

**Why asked.** Counter-intuitive result. They want to see whether you investigated or hand-waved.

**Model answer.** "RoBERTa got 0.5694 macro F1 against logistic regression's 0.8033. It looks like it contradicts a well-established finding. It doesn't, and four independent observations explain why.

First, the failure is confined to one class: per-class F1 is 0.743 negative and 0.740 positive — both fine — against 0.225 neutral. Second, separating precision from recall localises it further: neutral precision 0.522, recall 0.143. When the model assigns the label it's right about half the time; what it fails to do is assign it at all. A recall failure with intact precision is the signature of a decision boundary that never reaches the middle class.

Third, removing the neutral class raises its binary macro F1 to 0.9341 — above logistic regression's 0.9216. The expected ordering returns the moment that class is removed, which wouldn't happen if the model were generally inferior. Fourth, per-platform: neutral F1 is 0.713 on Twitter Airline against 0.198 on Yelp. Same model, same class, competent on one platform and failing on another.

The explanation is label mismatch. `twitter-roberta-base-sentiment-latest` was pretrained on tweets for a scheme where neutral means a human judged the post to express no clear sentiment. On Amazon and Yelp neutral means a three-star rating — a compound judgement often containing both praise and complaint. A model trained to recognise affectively *flat* text has no representation for a review that's affectively *contradictory*, so it must pick a pole."

**Follow-ups.** How would you prove causation rather than consistency?

**Answer, and give it unprompted.** "I can't from this evidence, and I say so in the dissertation. Proving it needs re-testing the same checkpoint against human-annotated neutral labels on review prose — same model, same text, different label definition. I didn't run it. Four consistent observations is strong circumstantial evidence, not a causal claim."

---

### Q24. Show me the error analysis. What do the mistakes actually look like?

**Model answer.** "I joined the cached per-review predictions back to source text and found three patterns.

Mixed evaluative content in three-star reviews: of 600 true-neutral reviews in the balanced sample, 300 got a polar label from RoBERTa while the fine-tuned model handled them correctly. These aren't uninformative texts — they carry clear praise and clear criticism together. 'The tea is very good however the bags are fragile' is a three-star review encoding a net judgement across both clauses.

Positive vocabulary in negative reviews, which is VADER's failure: a one-star review reading 'Worked well and looked good until keys stopped working after 3 months' scores +0.477 compound, because the complaint is carried by temporal contrast rather than negative words. Those errors sit *confidently* on the wrong side of the boundary, not near it — which is why widening the neutral band recovers so little.

Third, a residual group all five models get wrong, where rating and text genuinely diverge. That's the ceiling imposed by rating-derived labelling, not a modelling failure."

**Follow-ups.** How would you fix the label problem? Would you relabel?

**Answer.** "Relabelling a sample by hand is the honest fix — a few thousand reviews annotated for sentiment directly rather than derived from stars. That gives you a clean evaluation set even if you still train on rating-derived labels, and it lets you measure how much of the neutral difficulty is labels versus models. Right now I can't separate those two."

---

### Q25. Why did you test VADER's threshold, and what did it tell you?

**Model answer.** "Because the intuitive explanation for VADER's weakness is miscalibrated thresholds — the default neutral band is ±0.05, which is narrow, so few texts get assigned neutral. That's testable, so I tested it rather than asserting it.

The sweep largely rejects it. Widening the band raises macro F1 from 0.5094 to a maximum of 0.5353 at ±0.25 — a gain of 0.026 — then it declines as the band swallows genuinely polar reviews. So threshold choice explains a modest fraction of the gap, not most of it.

The larger problem is the lexicon itself: VADER labels 40.4% of genuinely negative reviews positive. That's a systematic failure of word-level scoring on mixed vocabulary, not a boundary-placement issue.

I'd highlight this one because it's the experiment that *disconfirmed* my own hypothesis. Ruling out the intuitive explanation is more informative than confirming it would have been."

**Follow-ups.** What's the lesson for hyperparameter tuning generally?

---

### Q26. Walk me through your fine-tuning setup and justify two non-obvious choices.

**Model answer.** "DistilBERT base uncased, three-class head, 70/15/15 stratified split seeded at 42, 4 epochs with early-stopping patience 2, batch 32 train / 64 eval, learning rate 2e-5, 1,000 warmup steps, weight decay 0.01, fp16.

Two choices that weren't obvious during implementation. Class weighting had to go through a custom trainer overriding the loss function, because the standard `Trainer` ignores class weights passed by configuration. Without that override the neutral class would be under-served despite a balanced corpus — and that's a defect invisible in aggregate metrics, visible only per class. Which connects to my whole reporting argument.

Second, early stopping on validation macro F1 rather than validation loss. Macro F1 is the metric I report, and selecting checkpoints on one quantity while reporting another risks presenting a model optimised for something other than the reported result. Validation loss fell to epoch 2 then rose while training loss kept declining — textbook overfitting — so best-checkpoint restoration selected epoch 2."

**Follow-ups.** Why 2e-5? Why fp16? Why DistilBERT over BERT?

**Answers.** 2e-5 is the standard fine-tuning range for BERT-family models — higher destroys pretrained representations, lower doesn't converge in few epochs. fp16 halves memory and speeds training on tensor cores; risk is gradient underflow, handled by loss scaling. DistilBERT because distillation retains ~97% of BERT-base performance at ~40% fewer parameters, which made fine-tuning feasible on freely available hardware — a real constraint I'd rather state than dress up as a technical preference.

---

### Q27. Your fine-tuned model only beat logistic regression by two points. Was fine-tuning worth it?

**Model answer.** "On aggregate macro F1, barely — roughly two points, though McNemar confirms it's a real difference at p = 1.7 × 10⁻²¹ rather than sampling noise. If that were the whole story, the answer would be no.

The real return is stability. DistilBERT scores 0.8448 on Amazon, 0.8301 on Yelp, 0.8505 on Twitter — a two-point spread. The pretrained model varies by twenty-three points across the same platforms. Logistic regression spans 0.6434 to 0.7948.

So the selection rule I'd give a practitioner is conditional: for a single-register feedback stream, logistic regression is the defensible engineering choice — it trains in minutes on CPU against three hours of GPU and a 257MB artefact. For several registers it isn't, because you're buying variance reduction, not peak accuracy.

There's a further wrinkle. DistilBERT's gain over logistic regression concentrates in the neutral class, 0.747 against 0.718. So for a business only separating satisfied from dissatisfied customers, fine-tuning buys very little — on binary re-evaluation the two sit at 0.9533 and 0.9216."

**Follow-ups.** Cost of serving each? When does the transformer become the wrong choice?

---

### Q28. What is McNemar's test and why use it over comparing F1 scores?

**Model answer.** "Because comparing two macro F1 numbers treats them as independent samples when they're not — both models were evaluated on the same reviews, so their errors are paired and correlated.

McNemar's uses only the discordant pairs: reviews where the two models disagree. Under the null hypothesis that both are equally accurate, each disagreement is equally likely to favour either model, so an exact binomial test on the discordant counts gives the probability of the observed imbalance.

It changed a claim I'd already written. Comparing macro F1 suggested a clean ordering of five models. Testing showed nine of ten pairwise differences hold at p < 0.05, but Naive Bayes against RoBERTa gives p = 0.82 — on the 675 reviews where they disagree, each is right on almost exactly half. So the defensible statement is that the pretrained transformer underperformed the *stronger* classical model, not classical ML as a family.

Important caveat: a non-significant result doesn't demonstrate equivalence. It means the available paired evidence doesn't support ranking either above the other."

**Follow-ups.** Why exact rather than chi-square? What sample did you use and why does it differ from the headline?

**Answer.** Exact binomial rather than the chi-square approximation because discordant counts can be small. And the test ran on a balanced 1,800-review diagnostic sample rather than the full test set, because per-review predictions weren't retained for the full set — so those accuracies aren't interchangeable with the headline numbers, and I flag that in the chapter.

---

### Q29. Explain macro F1 versus weighted F1, and why macro is your headline.

**Model answer.** "Macro averages the per-class F1 scores equally; weighted averages them by class support. On a balanced corpus they're close — mine are near-identical, which itself confirms the balancing worked.

Macro is the headline because it exposes single-class collapse, which is the literature's blind spot and the thing my central finding depends on. A three-class model performing well on two classes and collapsing on the third can still report a respectable weighted F1. VADER is the concrete case: accuracy 0.5142, macro F1 0.4381. That gap *is* the finding — performance unevenly distributed in a way accuracy conceals.

Balanced accuracy tracking accuracy closely for every model is my check that no model is benefiting from an uneven class prior."

**Follow-ups.** When would weighted F1 be the right headline? What about per-class F1 only?

---

### Q30. Explain self-attention. Why does it help here specifically?

**Model answer.** "Each token is projected into query, key and value vectors. Attention weights come from scaled dot products between queries and keys, softmaxed, then used to take a weighted sum of values — so every token's representation is conditioned on every other token in the sequence. Scaling by √d_k keeps the dot products out of the softmax saturation region as dimension grows. Multi-head runs several of these in parallel subspaces.

Why it matters for my data specifically: TF-IDF discards sequence entirely, so 'not worth the money' and 'worth the money, not…' produce identical feature vectors. Amazon and Yelp reviews are full of concessive constructions — 'the food was good but the service was appalling' — where the sentiment-bearing clause isn't identifiable from unigram counts. Attention can represent negation scope and clause relationships.

But I'd add the caveat my own results force: the transformer architecture accounts for less of the performance gap than it appears to. Naive Bayes trails logistic regression by 4.5 points on *identical* features, because its conditional-independence assumption is plainly false for text. Roughly two thirds of the distance from Naive Bayes to DistilBERT is recovered just by swapping one linear model for a better-specified one."

**Follow-ups.** Complexity of attention? What's the bottleneck for long sequences? Why did you truncate at 512?

---

### Q31. How did you evaluate the topic model with no ground truth?

**Model answer.** "Five weak indicators read together rather than one score, because the coherence-validity literature says no single automated measure is trustworthy. Hoyle et al. showed coherence metrics don't reliably reproduce human judgements; Doogan and Buntine argued the target construct — interpretability — is itself ambiguous; Lim and Lauw tested the correspondence at scale and found it weaker and more task-dependent than routine use implies.

So I used: centroid separation between merged categories (0.2831), within-category coherence as cosine similarity to centroid (0.6334 mean), category frequency distribution, platform distribution, and unclassified rate (27.6%) — plus a manual inspection sample.

The multi-indicator design is what changed my conclusion. Mean coherence of 0.6334 in isolation describes a working taxonomy. The other indicators say otherwise: 27.6% unclassified, one category absorbing 28.2% of the sample at among the *lowest* coherence recorded (0.3997), and positive reviews entering complaint categories. That's a taxonomy simultaneously too coarse in one region and too sparse elsewhere.

I'd also name the circularity: the same MiniLM model produced the clusters and measures their quality. So the structural observations — unclassified rate, size distribution, sentiment breakdown — are more reliable than the coherence figure, because they're directly countable and depend on no contested metric."

**Follow-ups.** How would you get ground truth? What would you annotate and how many?

**Answer.** "A gold set of 150–300 reviews annotated with complaint categories. That converts the question from an argument into a measurement — precision and recall for the categorisation stage — and lets the 0.35 confidence threshold be calibrated against human judgement rather than chosen."

---

### Q32. Your taxonomy assigns positive reviews to complaint categories. How did that happen?

**Model answer.** "Design error, and I'd call it that rather than a limitation of the method. The taxonomy was built from negative reviews only, but the assignment stage was applied to reviews of all sentiments. The zero-shot classifier scores *relative fit* among candidate categories — it has no notion of whether a text is a complaint at all. So a positive review gets assigned to whichever complaint category it least badly resembles.

The evidence: one category received 19 positive reviews and zero negative, and the unclassified group holds 60 positive against 34 negative.

The fix is a sentiment filter before assignment — reject non-complaints rather than categorising them. That's a one-line change in the pipeline order, which is exactly why I call it a design error: it's correctable, not fundamental.

There's a related defect worth mentioning because they may share a cause. One category retained an un-renamed raw BERTopic label, meaning the LLM naming step failed silently with no validation catching it. That's the *same* category receiving 19 positive and zero negative — so a cluster that resisted naming may have been incoherent to begin with."

**Follow-ups.** How would you validate the naming step? What's the assertion?

---

### Q33. Explain RAGAS faithfulness and answer relevancy. What are their weaknesses?

**Model answer.** "Faithfulness decomposes the answer into claims and checks each against the retrieved context, scoring the proportion supported. Answer relevancy works backwards — it generates candidate questions the answer would address and measures their similarity to the actual question.

Both are computed by an LLM judge, which is where the weaknesses live. Three in my case. Self-enhancement bias: the model judging is the model generating, documented as a validity threat in the LLM-as-judge surveys. Judge reliability failed visibly — RAGAS raised an `OutputParserException` on one question, returning prose instead of structured output, so my relevancy mean is over four questions not five. And relevancy actively rewards the wrong behaviour for my use case: an unconstrained model scores highly on relevancy while having no evidential basis at all, which is precisely what the system exists to prevent.

That last point is why I used four metrics rather than one. Reporting only relevancy would have concluded that retrieval augmentation degrades performance."

**Follow-ups.** Would an independent judge fix the bias entirely? How would you validate the judge itself?

**Answer.** "It removes self-enhancement but not judge error generally. Validating the judge means human agreement on a sample — score maybe 50 answers by hand and measure correlation with the judge. Without that you're trusting an unvalidated instrument, which is a version of the same problem I criticised the coherence literature for."

---

### Q34. Explain your ablation. Would you ship metadata tagging?

**Model answer.** "The design argument was that tagging each retrieved review with platform, rating and sentiment is what makes the prompt's citation rules enforceable — the prompt says 'cite only platforms present in the evidence,' and without tags that instruction can't be followed or checked. That was reasoning, not evidence, so I tested it: same five questions, same pipeline, run twice, retrieval held identical so any difference comes from what the generator was shown.

Result is a trade-off, not the confirmation I expected. Faithfulness 0.667 tagged against 0.549 untagged, so +0.118. But answer relevancy 0.452 against 0.654 — a *loss* of 0.202, the largest single delta in the table. Context precision and recall are identical to three decimals, which confirms the ablation isolated what it claimed to.

Likely mechanism is mundane: tagged answers spend more of their length on citation-style attribution, and a relevancy judge scores directness.

On shipping — yes, and one piece of qualitative evidence drives that more than the metrics. For the battery-life question, the untagged answer opened 'Platform(s) covered: Twitter Airline, Amazon' when none of the five retrieved reviews mentions an airline. The tagged answer correctly said Yelp and Amazon. That's a platform hallucination caught in the act, prevented by tagging. For an analyst-facing tool where citations must be checkable, a fabricated source is a worse failure than a less punchy answer."

**Follow-ups.** n=5 — is that enough to ship on? What would change your mind?

**Answer.** "No, and I'd frame it as directional. What would change my mind is the relevancy loss reproducing at n=100 *and* the hallucination benefit not — if tagging costs directness without reliably preventing fabrication, the trade stops being worth it."

---

### Q35. What's the difference between bi-encoder and cross-encoder, and where would each go?

**Model answer.** "A bi-encoder embeds query and document independently, so document vectors precompute offline and query time is a vector search — that's what makes retrieval over 642,692 documents feasible at 21.71ms. The cost is that it can't model term interaction between query and document.

A cross-encoder takes the pair jointly through the transformer and outputs a relevance score. Much more accurate, but nothing precomputes, so scoring all 642,692 pairs per query is impossible.

Hence two-stage retrieval: bi-encoder recalls a wide candidate set cheaply, cross-encoder reranks a small set precisely. My pipeline only has stage one, and I can point at the cost — a Yelp review about waiting to get a battery *replaced in-store* ranked highly for a question about battery *life*, because the bi-encoder matched on the token without modelling what was asked about it. A cross-encoder scoring the pair jointly would likely have demoted it."

**Follow-ups.** What about late-interaction models like ColBERT? Where do they sit?

---

### Q36. How would you detect model drift in production?

**Model answer.** "Three layers, cheapest first.

Input drift: monitor the embedding distribution of incoming reviews against the indexed corpus — population stability index or a distance between distributions. If new reviews start landing in embedding regions the index doesn't cover, retrieval quality degrades silently before any accuracy metric moves.

Prediction drift: track the sentiment class distribution over rolling windows. My corpus is roughly balanced at 34/33/33; a sustained shift signals either genuine change in customer sentiment or a broken preprocessing step, and you need to distinguish those.

Behavioural drift, specific to my system: refusal rate broken down by *reason*. The scope-guard bug taught me that lesson — an unexplained rise in out-of-scope rejections means the classifier is degrading, and it's invisible in any accuracy metric because those questions never reach the model.

For ground-truth-dependent metrics I'd sample and label continuously rather than waiting for a labelled batch, because by the time you have one the drift is weeks old."

**Follow-ups.** What threshold triggers retraining? How do you avoid alert fatigue?
