# Chapter 6 — Conclusion and Future Work

This chapter closes the study. It reviews what was done against what was promised, states what the work contributes and to whom, sets out the limitations honestly along with the future work each one implies, and reflects on the experience of carrying the project out. The intention is to leave a reader clear about both what can be claimed from this dissertation and what cannot.

---

## 6.1 Summary of the Dissertation

This dissertation compared, stage by stage, the classical machine learning, transformer-based and retrieval-augmented approaches available at each point in a customer complaint analysis pipeline. The comparison ran on one corpus of 642,692 reviews drawn from Amazon, Yelp and Twitter Airline feedback, using a single evaluation protocol so that differences between methods could not be attributed to differences in data.

The motivation was practical. Method selection in applied feedback analysis is usually made on reputation, supported by benchmark figures from single-domain studies that report aggregate metrics. This study set out to test whether that basis survives contact with data resembling what an organisation actually holds.

**Objective 1** required a literature review identifying where existing comparisons are absent or restricted to a single domain. Chapter 2 delivered this, and located two gaps: the shortage of controlled comparison at each pipeline stage on multi-domain data, and the shortage of verification of what these systems produce. The review also surfaced a body of work questioning whether automated topic coherence metrics measure interpretability at all, which directly shaped the evaluation design in Chapter 3.

**Objective 2** required a methodology and a unified dataset suitable for controlled comparison. CRISP-DM was selected and adapted so that modelling and evaluation execute once per pipeline stage. The corpus was assembled with a documented labelling rule, balanced classes and fixed seeds. This objective was met, with one qualification recorded in Section 3.5.3: the fine-tuned model was developed against a separately generated split, so the five-model comparison is less tightly controlled than intended.

**Objective 3** required five sentiment classifiers to be implemented, compared and diagnosed. All five were evaluated on a held-out set of 128,539 reviews with per-class, per-platform and confusion matrix reporting, and three diagnostic experiments tested the mechanism behind the results that contradicted expectation. This objective was fully met, and produced the study's principal finding.

**Objective 4** required complaint topics to be discovered per platform and consolidated into a cross-domain taxonomy. BERTopic produced 121 raw topics, merged into 24 categories at a centroid separation of 0.2831. The objective was met, though the evaluation exposed real defects in the result, discussed below.

**Objective 5** required a retrieval-augmented question-answering component and a standalone baseline to compare it against. Both were built, with source review identifiers returned on every answer and prompt-level grounding constraints made checkable through metadata tagging. This objective was met.

**Objective 6** required stage-appropriate evaluation with baseline comparison. Supervised metrics, unsupervised coherence and coverage measures, and reference-free generation metrics were all applied, and differences between models were tested for statistical significance rather than asserted from a ranking. This objective was met in part. The planned ablation of retrieval design decisions was implemented but not executed, for reasons given in Section 3.6.6.

**Objective 7** required critical discussion, evaluation against the objectives, and identification of future work. Chapter 5 and this chapter deliver it.

Taken together, the objectives fulfil the aim. The aim was to establish which approach is justified at which stage and how far the output of each can be trusted. The first half is answered by the comparative results and the significance testing; the second by the class-level and platform-level reporting, the taxonomy coverage figures, and the faithfulness scores with their stated limits. Where the aim is only partly met is in the strength of evidence at the generation stage, which rests on five questions and a self-assessing judge.

---

## 6.2 Research Contributions

### 6.2.1 Key findings

Four findings carry the study.

The ranking of sentiment methods is conditional on domain, not fixed. Pretrained RoBERTa scored 0.7853 macro F1 on Twitter Airline against 0.5544 on Yelp, and on the Twitter data it outperformed both classical models. No single ordering of these methods is correct, and a study conducted on one platform would have concluded confidently in either direction.

An apparently poor result had a specific and testable cause. RoBERTa's aggregate weakness resolved into a single-class failure: neutral F1 of 0.225 against 0.743 and 0.740 on the other two classes, with precision at 0.518 but recall at 0.143. Removing the neutral class raised its binary macro F1 to 0.9341, above logistic regression. The failure reflects a mismatch between the model's learned label definition and the rating-derived scheme it was tested against.

Fine-tuning's principal benefit is stability, not peak accuracy. DistilBERT gained roughly three points of macro F1 over logistic regression, a difference confirmed significant at p = 1.7 × 10⁻²¹, but varied by only two points across platforms where the pretrained model varied by twenty-three.

Retrieval augmentation improves verifiability while scoring worse on relevance. The standalone model reached 0.920 answer relevancy against the pipeline's 0.634, in less than half the time, while producing answers with no evidential basis at all. Faithfulness, which reached 0.755 for the pipeline, is undefined for the baseline.

### 6.2.2 Academic contributions

The stage-wise cross-domain comparison is the primary contribution. The literature contains many single-domain comparisons and a separate body of work on domain adaptation, but controlled comparison of the same off-the-shelf methods across three registers, reported at class and platform level, is uncommon. The finding that the ranking reverses by platform is a direct product of that design.

The diagnostic method is a second, more modest contribution. Instead of reporting an anomalous result and speculating about it, this study tested the explanation through independent experiments and confirmed it. Combined with pairwise significance testing, which identified one model pair that cannot be separated at p = 0.82 and so prevented an overclaim, the approach demonstrates a standard of evidence that costs little and is applicable to any comparative evaluation.

Third, the study contributes negative results that are rarely published. A complaint taxonomy built from negative reviews will assign positive reviews to complaint categories unless a sentiment filter intervenes; a fixed category count imposed across platforms of differing semantic breadth produces a catch-all category and a high unclassified rate; and answer relevancy alone will reward exactly the ungrounded behaviour that retrieval augmentation exists to prevent. Each is the kind of finding that aggregate reporting conceals.

### 6.2.3 Practical contributions

For practitioners, the study offers a method-selection rule with evidence behind it: check whether a candidate model's label definition matches your own before considering its reported accuracy, and where feedback arrives from several sources, prefer the model with the smallest variance across them over the highest mean.

It also produces reusable artefacts: a documented preprocessing pipeline over three public datasets, a 24-category cross-platform complaint taxonomy, and an evaluation suite covering supervised, unsupervised and generation metrics with significance testing.

The cost analysis in Section 5.5 supports a tiered deployment argument. High-volume routine classification is well served by logistic regression on CPU; the fine-tuned transformer earns its cost where the neutral class matters or feedback spans registers; and the retrieval-augmented component, at roughly 1.5 seconds and a metered API call per query, belongs in an analyst-facing tool, not an automated pipeline.

### 6.2.4 Beneficiaries

Customer experience and product teams who currently read samples of feedback by hand gain complaint categories derived from data instead of assumption, and answers that arrive with the reviews supporting them. Applied researchers building similar pipelines gain evidence on which components are fragile and which evaluation choices conceal failure. Practitioners choosing between these methods gain a comparison conducted on data that resembles theirs, rather than on a benchmark.

---

## 6.3 Limitations and Future Research

### 6.3.1 Limitations and their effect on the findings

The limitations are of unequal severity, and each constrains a specific claim.

**Split contamination.** The fine-tuned model was trained on a separately generated split, so approximately a quarter of the shared sentiment test set overlaps data it had seen. The uncontaminated figure from its own held-out partition is reported as definitive, but the five-model comparison is not as controlled as the design intended. This weakens the precision of the DistilBERT comparison specifically; it does not affect the other four models or the per-platform findings.

**Rating-derived labels.** Sentiment labels come from star ratings for two of the three platforms, so a four-star review containing a complaint is labelled positive, and "neutral" means a three-star rating on two platforms and a human judgement on the third. Some part of every model's neutral-class difficulty is attributable to the labels, not the models, which places a ceiling on absolute performance and makes cross-platform neutral comparisons less clean than they appear.

**Generation evidence.** Five questions with a self-assessing judge is the weakest evidence in the study. Per-question faithfulness ranged from 0.444 to 1.000 and context recall was 0.000 on two of five questions. The qualitative finding — that relevancy and faithfulness diverge, and that the baseline cannot be assessed for grounding at all — is supported. The specific values are not measurements that should be quoted as system performance.

**No ground truth for categorisation.** Without labelled data, no accuracy can be reported for the complaint taxonomy, and the unsupervised measures used instead are contested in the literature. The structural evidence, meaning the unclassified rate, the size distribution and the sentiment breakdown, is more reliable than the coherence figures.

**Unexecuted ablations.** The contribution of individual retrieval design decisions is argued from design reasoning, not measured. The claim that metadata tagging and prompt constraints produce the observed faithfulness therefore rests on inference.

**No deployment or user study.** The system has no interface and was never used by an analyst, so claims about practical value rest on the traceability it demonstrably provides rather than on observed benefit.

**Significance testing sample.** McNemar tests were computed on a balanced sample of 1,800 reviews because per-review predictions were not retained for the full test set. The strongly significant results are robust to this, but the single non-significant pair should be read as "not distinguishable on the available evidence", not as established equivalence.

### 6.3.2 Future work

Six directions follow, ordered by how directly they address the limitations above.

**Establish a single canonical split.** Generating one stratified split, saving it to disk and reusing it for every model would remove the contamination issue entirely and make all five figures strictly comparable. This is the first correction future work should make, and it requires no retraining beyond re-evaluation.

**Build a manually labelled evaluation set.** A gold set of 150 to 300 reviews annotated with complaint categories would convert RQ3 from an argument into a measurement, allowing accuracy, precision and recall to be reported for the categorisation stage and the confidence threshold to be calibrated against human judgement.

**Use an independent judge model.** Scoring generated answers with a different model from the one that produced them would remove the self-enhancement bias, and running a larger question set would reduce the variance that currently dominates the generation results. Both were prevented by free-tier rate limits rather than by any methodological obstacle.

**Execute the ablation suite.** The experiments varying retrieval depth, retriever type, prompt version and metadata tagging are already implemented. Running them would establish which design decisions produce the observed grounding, converting an argued claim into a measured one.

**Redesign the taxonomy stage.** Three specific changes follow from Section 5.2: apply a sentiment filter before category assignment so that non-complaints are rejected; derive the category count per platform instead of fixing it at 24; and adopt a hierarchical structure, with broad categories for reporting and the finer per-platform topics retained beneath them for investigation. Validation should also be added to the naming step, which failed silently on at least one cluster.

**Explore generative fine-tuning.** Parameter-efficient fine-tuning of a small instruction-tuned model on this corpus, compared against the fine-tuned encoder, would test whether generative fine-tuning improves the neutral-class boundary that Section 5.1 identifies as the limiting factor. This is the most substantial extension and the least certain to pay off, which is why it belongs in future work and not in the present study.

Beyond these, deployment as a web service with an analyst-facing interface would enable the user study that RQ4 was deliberately narrowed to avoid claiming.

---

## 6.4 Personal Reflections

### 6.4.1 Strengths demonstrated

The strength I am most confident about is a willingness to investigate results that looked wrong instead of reporting them and moving on. When the pretrained transformer scored below a linear model, the easy path was to note it as an anomaly. Testing the explanation with a per-platform breakdown and a binary re-evaluation took perhaps a day, and it turned the weakest-looking result in the project into its most interesting finding. I would not have predicted at the outset that this would matter more than any modelling decision I made.

I also became noticeably better at debugging problems that were not in my own code. A segmentation fault with no Python traceback turned out to be a conflict between two libraries each bundling their own OpenMP runtime, and an evaluation framework failed because it imported a class that a dependency had removed in a major version. Neither was solvable by reading my own logic. Both required reading other people's source.

### 6.4.2 Challenges and weaknesses

The clearest weakness was planning, not technique. I generated data splits separately in different scripts as I needed them, and only discovered late that this left the fine-tuned model's evaluation partially contaminated. A single split file created at the outset would have prevented it. The lesson is that experimental controls need designing before the experiments, not reconstructing afterwards, and I have had to disclose the consequence rather than fix it.

A related error was a mistyped tokeniser argument that was silently ignored, so the model trained on sequences padded to 512 tokens while inference truncated at 128. It caused no crash and produced no warning. It is a reminder that in machine learning, code that runs successfully is not evidence that it is doing what was intended.

Resource constraints shaped the work more than I expected. Free-tier API limits reduced the generation evaluation to five questions, prevented the use of an independent judge model, and ultimately stopped the ablation suite from running despite it being fully implemented. I treated the constraint as fixed and worked around it. In hindsight a small budget for API access would have strengthened the evaluation more than any modelling change I could have made, and I should have recognised that trade-off earlier.

I also underestimated how much of the difficulty in an unsupervised task lies in evaluation, not implementation. Running BERTopic was straightforward. Deciding what would count as evidence that its output was any good, given that no correct answer exists, took considerably longer and remains the part of the study I am least satisfied with.

### 6.4.3 What I learned

The most transferable lesson is that reporting conventions determine what a study can see. Had I reported only accuracy, as much of the literature I reviewed does, the central finding of this dissertation would have been invisible. Per-class and per-platform reporting cost nothing and revealed everything.

The second is about the value of honest negative results. The findings I initially wanted to hide — a transformer losing to a linear model, a quarter of reviews left unclassified, a baseline outscoring the system on one metric — turned out to be the ones that made the discussion worth reading. Explaining why an unexpected result occurred demonstrates more understanding than confirming an expected one.

The third is that statistical testing changes what can be claimed. Comparing macro F1 scores suggested a clean ordering of five models. Testing it showed that one of those differences does not hold, which meant narrowing a claim I had already written. I would now regard a comparison reported without significance testing as incomplete.

### 6.4.4 How future projects could be improved

Three changes I would make from the start of a similar project. Fix the experimental controls first: generate the split once, save it, and have every script read it. Verify that configuration is taking effect rather than assuming a silent run means a correct one, particularly for arguments passed to third-party libraries. And decide the evaluation design before building anything, because in this project the evaluation determined which findings were reachable, and adding measurements afterwards was consistently harder than it would have been to plan them at the outset.

I would also budget explicitly for compute and API access at the planning stage, treating it as a project constraint to be managed instead of a limit to be worked around after it starts biting.
