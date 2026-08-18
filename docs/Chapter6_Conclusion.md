# Chapter 6 — Conclusion and Future Work

Chapter 5 interpreted the results one stage at a time. This chapter draws them together: what was done against what was promised, what the work contributes, what it cannot claim, and what follows from each limitation.

---

## 6.1 Summary of the Dissertation

This dissertation compared, stage by stage, the classical machine learning, transformer-based and retrieval-augmented approaches available at each point in a customer complaint analysis pipeline, on one corpus of 642,692 reviews from Amazon, Yelp and Twitter Airline, using a single evaluation protocol so that differences between methods could not be attributed to differences in data.

The motivation was practical. Method selection in applied feedback analysis is usually made on reputation, supported by benchmark figures from single-domain studies reporting aggregate metrics. This study tested whether that basis survives contact with data resembling what an organisation actually holds. It does not, and the way it fails proved more interesting than the fact of failure.

**Objectives 1 to 5 were met.** Chapter 2 delivered the literature review and located two gaps: the shortage of controlled comparison at each pipeline stage on multi-domain data, and the shortage of verification of what these systems produce. CRISP-DM was selected and adapted so that modelling and evaluation execute once per stage, and the corpus was assembled with a documented labelling rule, balanced classes and fixed seeds — with one qualification recorded in Section 3.5.1, that the fine-tuned model was developed against a separately generated corpus version. Five classifiers were evaluated on 128,539 held-out reviews with per-class, per-platform and confusion-matrix reporting, pairwise significance testing, and three diagnostic experiments. BERTopic produced 121 raw topics merged into 24 categories at 0.2831 centroid separation. The retrieval-augmented component and its standalone baseline were both built, with source identifiers returned on every answer.

**Objective 6 was met in part.** Supervised metrics, unsupervised coherence and coverage measures, and reference-free generation metrics were all applied, and differences between models were tested for statistical significance rather than asserted from a ranking. Of four planned ablation conditions, one — metadata tagging — was built and executed, and its result is reported in Section 4.5.8; the other three were specified in the design but never built, for the reasons given in Section 3.6.5.

**Objective 7** is delivered by Chapter 5 and this chapter.

Taken together the objectives fulfil the aim unevenly. The aim was to establish which approach is justified at which stage and how far the output of each can be trusted. The first half is answered well by the comparative results, the per-platform breakdown and the significance testing. The second is answered at the classification and taxonomy stages, where class-level reporting and coverage figures quantify what can be trusted, and answered only directionally at the generation stage, where the evidence rests on five questions and a self-assessing judge.

---

## 6.2 Research Contributions

### 6.2.1 Key findings

**The ranking of sentiment methods is conditional on domain, not fixed.** Pretrained RoBERTa scored 0.7853 macro F1 on Twitter Airline against 0.5544 on Yelp, outperforming both classical models on the former. Naive Bayes showed the mirror pattern, 0.7952 on Amazon against 0.4838 on Twitter. No single ordering is correct, and a study conducted on one platform would have concluded confidently in either direction.

**An apparently poor result had a specific and testable cause.** RoBERTa's aggregate weakness resolved into a single-class failure: neutral F1 of 0.225 against 0.743 and 0.740 on the other classes, with precision 0.522 but recall 0.143. Removing the neutral class raised its binary macro F1 to 0.9341, above logistic regression's 0.9216. The evidence is consistent with a mismatch between the model's learned label definition and the rating-derived scheme it was tested against, though establishing that causally would require a further experiment not carried out here.

**Fine-tuning's principal benefit is stability, not peak accuracy.** DistilBERT gained roughly two points of macro F1 over logistic regression, confirmed significant at p = 1.7 × 10⁻²¹, but varied by only two points across platforms where the pretrained model varied by twenty-three. For a single-register feedback stream the linear model is defensible; for several registers it is not.

**Retrieval augmentation buys verifiability, and the trade is visible in the distribution rather than the mean.** The standalone model reached 0.753 mean answer relevancy against the pipeline's 0.610 in under half the time, but its per-question scores span 0.000 to 0.988 against the pipeline's 0.437 to 0.819: higher ceiling, lower floor. Faithfulness of 0.540 is measurable for the pipeline and undefined for the baseline, and on the question where retrieval was genuinely weak it fell rather than staying artificially high.

**Component-level metrics located a real weakness in retrieval, and a separate one in scope classification.** Context precision averaged 0.416 against faithfulness of 0.540. On the airline question, where precision fell to zero, faithfulness fell with it — direct evidence that the grounding threshold suppresses confident answers built on weak evidence. But not every zero had that cause: one question was never retrieved for at all, rejected beforehand by the scope guard on a false positive. Tracing that back mattered, because crediting it to the grounding threshold would have been a comfortable but wrong story.

**Metadata tagging is a trade-off, not a free improvement.** The one executed ablation found tagging lifting faithfulness by 0.118 while costing 0.202 on answer relevancy, alongside one directly observed platform hallucination that tagging prevented.

### 6.2.2 Academic and practical contributions

The stage-wise cross-domain comparison is the primary contribution. Controlled comparison of the same off-the-shelf methods across three registers, reported at class and platform level, did not appear in the literature reviewed here, and the finding that the ranking reverses by platform is a direct product of that design — applied evidence for a premise the cross-domain adaptation literature assumes.

The diagnostic method is second and more modest: testing a proposed explanation through three independent experiments, one of which ruled out the intuitive explanation rather than confirming it. With significance testing that identified one model pair which cannot be separated, this demonstrates a standard of evidence that costs little and applies to any comparative evaluation.

Third are negative results of a kind rarely published. A taxonomy built from negative reviews will assign positive reviews to complaint categories unless a filter intervenes. A fixed category count across platforms of differing semantic breadth produces a catch-all alongside a high unclassified rate. Answer relevancy alone rewards exactly the ungrounded behaviour retrieval augmentation exists to prevent. Each is a finding aggregate reporting conceals, surfaced by a reporting choice rather than a modelling one.

For practitioners the study offers a selection rule with evidence behind it: check whether a candidate's label definition matches your own before considering its reported accuracy, and where feedback arrives from several sources prefer the model with the smallest variance rather than the highest mean. It also leaves reusable artefacts — a documented preprocessing pipeline, a taxonomy with measured coverage limits, and an evaluation suite in which every figure is written to file by the script that produced it.

No novel algorithm is proposed, and none is claimed. The contribution is comparative, diagnostic and methodological.

---

## 6.3 Limitations and Future Work

### 6.3.1 Limitations

Each constrains a specific claim rather than the study as a whole.

**Split control.** The fine-tuned model was developed against a separately generated corpus version, so its shared-test-set figure is less tightly controlled than the other four models', and the extent cannot now be quantified because the earlier version was not retained. The uncontaminated figure from the model's own held-out partition, 0.8375, is reported as definitive throughout. This weakens the precision of the DistilBERT comparison specifically; it does not affect the other four models, the per-platform findings, or the diagnostics.

**Rating-derived labels.** A four-star review containing a complaint is labelled positive, and "neutral" means a three-star rating on two platforms and a human judgement of affective flatness on the third. Some part of every model's neutral-class difficulty is attributable to the labels rather than the models, and this study cannot separate the two.

**Generation evidence.** Five questions with a self-assessing judge is the weakest evidence here, and the judge's reliability is part of that weakness — it failed to return a parseable score for one question, so answer relevancy is averaged over four. The qualitative findings are supported; the specific values should not be quoted as system performance.

**No ground truth for categorisation.** Without labelled data no accuracy can be reported, and the unsupervised measures used instead are contested and partly circular, since the embedding model that formed the clusters also measures their quality. The structural evidence — unclassified rate, size distribution, sentiment breakdown, failed naming step — is more reliable and is what the conclusions rest on.

**Mostly unexecuted ablations.** One of four conditions was executed. The other three — retrieval depth, retriever type and prompt version — remain argued from design reasoning. Given that Section 5.3.3 identifies retrieval quality as a genuine limiting factor, the unexecuted retrieval-depth condition is the most consequential of what is left.

**A scope guard that rejects questions it should not.** Chasing down why one generation question went unanswered turned up something more specific than insufficient evidence: the two-stage scope guard rejected a legitimate complaint-exploration question. The same false rejection recurred identically across three runs, which rules out a fluke and points to a real precision problem in the guard's classifier. It was initially misread as the grounding threshold working correctly; tracing the exact returned message to its source corrected that. The practical consequence is mild — a false rejection is safer than a hallucinated answer — but the guard's accuracy is now an open question this study did not set out to test.

**No deployment or user study,** so claims about practical value rest on demonstrable traceability rather than observed benefit. **Significance testing** used a balanced sample of 1,800 reviews; the single non-significant pair should be read as "not distinguishable on the available paired evidence", not as established equivalence.

### 6.3.2 Future work

Ordered by how directly each addresses a limitation above.

**Establish a single canonical split.** Generating one stratified split, saving it, and having every script read it would remove the contamination issue entirely and make all five figures strictly comparable. It requires retraining the fine-tuned model but no methodological change.

**Calibrate and test the scope guard.** The false rejection is currently one observed case, reproduced three times but never systematically characterised. A set of roughly 100 questions — half clearly in scope, half outside, with a band of borderline phrasings between — would turn the guard's precision and recall into measured quantities and let the 0.35 similarity threshold be calibrated against evidence rather than chosen by inspection. This is the cheapest experiment on the list and addresses a defect the evaluation found by accident.

**Build and run the remaining three ablation conditions.** They exist as a specification rather than as code, and `evaluate/ablation_metadata_tagging.py` establishes the pattern — same question set, one variable changed, everything else held constant. Retrieval depth is the one to do first, since a *k*-sweep would say directly whether five retrieved reviews is the right amount of evidence or simply the number that was chosen.

**Use an independent judge and a larger question set.** Scoring answers with a different model would remove self-enhancement bias, and expanding beyond five questions would reduce the variance that currently dominates. Both were prevented by free-tier rate limits rather than any methodological obstacle.

**Build a manually labelled evaluation set.** A gold set of 150 to 300 reviews annotated with complaint categories would convert RQ3 from an argument into a measurement and allow the 0.35 confidence threshold to be calibrated against human judgement.

**Redesign the taxonomy stage.** Four changes follow from Section 5.2: a sentiment filter before assignment so non-complaints are rejected; a category count derived per platform rather than fixed at 24; a hierarchical structure, with broad categories for reporting and finer per-platform topics retained beneath them; and validation on the language-model naming step, which failed silently and passed an un-renamed label into the deployed taxonomy.

Beyond these, deployment as an analyst-facing service would enable the user study RQ4 was deliberately narrowed to avoid claiming, and would test whether measurable traceability translates into analyst trust.

---

## 6.4 Personal Reflections

The strength I am most confident about is a willingness to investigate results that looked wrong rather than reporting them and moving on. When the pretrained transformer scored below a linear model, the easy path was to note it as an anomaly; testing the explanation with a per-platform breakdown and a binary re-evaluation took about a day and turned the weakest-looking result into the most interesting finding.

The clearest weakness was planning, not technique. I generated data splits separately in different scripts as I needed them, and discovered late that this left the fine-tuned model's evaluation partially uncontrolled — and that the corpus version it trained on no longer existed, so the extent could not be measured afterwards. A single split file created at the outset would have prevented it. A related error was a mistyped tokeniser argument that was silently ignored, so the model trained on sequences padded to one length while inference truncated at another: no crash, no warning, and a reminder that code running successfully is not evidence it is doing what was intended.

Resource constraints shaped the work more than I expected. Free-tier limits reduced the generation evaluation to five questions, prevented an independent judge, and meant three of four designed ablations were never built — there was no point writing code I could not afford to run. Completing one of them late was enough to make me regret the other three.

Three lessons transfer. Reporting conventions determine what a study can see: had I reported only accuracy, the central finding would have been invisible. Statistical testing changes what can be claimed — testing the macro F1 ordering showed one difference does not hold, which meant narrowing a claim I had already written.

The third I found most uncomfortable. One question scored zero on three metrics and I wrote it up as the grounding threshold working exactly as intended. It fitted the design argument neatly. It was also wrong: the text the pipeline returned was the out-of-scope message, not the no-evidence message, meaning a scope classifier had rejected a valid question before retrieval ran. The numbers were identical either way; only the returned string distinguished a safety feature working from a component misfiring. I had confirmed my own design rather than tested it, and the correction surfaced only because later runs reproduced the rejection. The lesson is narrower than "be careful": when a result agrees with what you designed for, that is the moment to check the artefact, not to write the sentence you were hoping to write.

For a similar project I would fix experimental controls first, verify that configuration is taking effect rather than assuming a silent run means a correct one, decide the evaluation design before building anything, and budget for compute and API access at the planning stage.

---

## 6.5 Closing Remarks

The study set out to test whether method selection in customer feedback analysis can rest on published benchmark scores. It cannot. The ranking of sentiment methods reverses depending on which platform judges it; a topic model producing well-separated categories can still leave a quarter of reviews unclassified and assign positive reviews to complaint categories; and a retrieval-augmented system can score worse than an unconstrained one on the metric most often reported while being the only one of the two whose answers can be checked at all.

None of these conclusions required a new algorithm. They required running the comparison across more than one domain, reporting at a granularity that admits failure, testing explanations rather than asserting them, ablating a design decision instead of arguing for it, and looking at distributions rather than means.

It is worth ending on what that discipline cost, since it produced the caveats as well as the findings. Reporting per-class results exposed a failure the aggregate would have hidden. Running the ablation returned a trade-off rather than the confirmation it was set up to provide. Checking the returned text turned a tidy claim about a safety mechanism into an admission that a classifier had misfired. Each made the dissertation harder to write and, I think, more worth reading. A study that finds only what it expected to find has usually not looked hard enough.
