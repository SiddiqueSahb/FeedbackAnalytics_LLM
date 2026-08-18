# FeedbackIQ — MLE / Applied Scientist Interview Preparation

**Candidate:** Mohammad Asim Siddique
**Project:** FeedbackIQ — stage-wise comparison of classical ML, transformers and RAG over 642,692 customer reviews
**Target roles:** MLE · AI Engineer · Applied Scientist · Data Scientist
**Target level:** Google L4/L5 · Meta E4 · Amazon SDE II / Applied Scientist I · OpenAI MLE

---

## How to use this pack

Every answer here is built from your actual results. Do not memorise them — memorised answers collapse on the second follow-up, which is exactly where interviewers probe. Read the model answer, close the document, and say it in your own words with the numbers correct. If you cannot produce the number from memory, you do not yet know that part of your project well enough to claim it.

Three rules that matter more than any individual answer.

**Lead with the decision, not the tour.** Nobody wants your pipeline stage by stage. They want to know what you chose and why.

**Quantify or don't claim.** "It improved faithfulness" is worthless. "Faithfulness rose 0.118 while relevancy fell 0.202 on five questions" is a real answer, caveat included.

**Name the cost.** Every design decision has one. Candidates who state it read as engineers; candidates who hide it read as students.

### Your five strongest assets in an interview

| Asset | Why it lands |
|---|---|
| The RoBERTa label-mismatch diagnosis | A genuine debugging story with four independent lines of evidence. Better than most candidates' answer to "tell me about a hard bug." |
| The scope-guard misdiagnosis | You wrote up a result as your design working, then found it was a component misfiring. Self-correction is rare and senior-signalling. |
| The grounding threshold | An architectural constraint replacing a prompt instruction, with measured cost. Shows systems thinking about model failure. |
| The executed ablation | Returned a trade-off, not a confirmation. Shows you ran the experiment rather than assuming its outcome. |
| Reporting negative results | RAG scoring *worse* than the baseline on relevancy, reported and explained rather than buried. |

### Your five biggest liabilities — know these cold

| Liability | Expect to be pushed on |
|---|---|
| Split contamination | Test set overlapped training data; original corpus gone so the extent is unquantifiable. |
| `SIMILARITY_THRESHOLD = 0.35` | Unvalidated constant gating whether the system answers at all. |
| n = 5 generation evaluation | Any claim from it is directional. Do not oversell. |
| Same model generates and judges | Self-enhancement bias, acknowledged but unresolved. |
| Three of four ablations unbuilt | Design decisions argued, not measured. |

---

# ROUND 1 — Project and Architecture

### Q1. Walk me through your project in two minutes. What was the most technically interesting decision?

**Why asked.** Tests whether you can compress and prioritise. Rambling here loses the interview in the first four minutes.

**Model answer.** "I built a customer feedback analytics pipeline over 642,692 reviews from Amazon, Yelp and airline tweets. Three stages: sentiment classification comparing five models, complaint discovery with BERTopic consolidating 121 topics into 24 categories, and a RAG chatbot over a FAISS index that returns source reviews with every answer.

The decision I'd defend hardest is putting a hard similarity threshold in front of generation rather than trusting the prompt to refuse. Prompt-only refusal is the standard pattern and it's unenforceable — Ji et al. document hallucination as a systematic property, so an instruction not to hallucinate is enforced only by the model's compliance on that request. I have direct evidence it fails: in an earlier un-gated run, one question retrieved context scoring 0.000 on context precision and the model still produced an answer RAGAS scored 0.917 on faithfulness. It was faithfully summarising evidence its own retriever judged irrelevant.

So I moved the check into code — score candidates, discard below 0.35, return a fixed refusal without calling the generator if nothing survives. Aggregate faithfulness dropped to 0.540. I'd argue that's correct, because the earlier number was partly measuring confident answers built on bad evidence."

**Common mistakes.** Narrating all three stages equally. Listing technologies. No numbers. Claiming the design was obviously right.

**Follow-ups to expect.** Why 0.35? What's the false-refusal rate? Why not have the LLM judge evidence sufficiency? What did faithfulness measure before the gate?

---

### Q2. Draw your architecture. Where does a request spend its time?

**Model answer.** "Streamlit frontend calls a FastAPI backend over REST. The RAG path: request hits `ask()` in `rag/pipeline.py`, passes a two-stage scope guard, then retrieval against a FAISS index of 642,692 MiniLM-L6-v2 embeddings at 384 dimensions, then generation via Llama 3.1 8B on Groq at temperature 0.2.

On latency, retrieval is not the bottleneck. Retrieval is 21.71ms; end-to-end grounded answer is 1,582.7ms, so retrieval is roughly one per cent. The extra ~849ms over the ungrounded baseline comes from the larger prompt once five reviews are inserted, plus an additional LLM call from the history-aware retriever that rewrites follow-up questions into standalone queries.

The interesting case is a rejected question — 112ms, because both guards return a fixed string without reaching generation. So the cost of grounding is paid in generation tokens, except on questions the system declines."

**Follow-ups.** Which of those numbers did you measure vs infer? What's your p99, not mean? Why is the history-aware retriever an extra call and could you avoid it?

**Honest gap to admit.** You measured mean latency over five questions, not p99 under load. Say so — inventing a p99 is worse.

---

### Q3. Why three separate stages rather than one end-to-end model?

**Model answer.** "Because they're different problems with different supervision. Sentiment classification is supervised with labels. Complaint discovery is unsupervised — the categories don't exist before the model finds them. Question answering is generation, where the failure mode is a fluent answer with no evidential basis.

They're also judged by incommensurable metrics. Macro F1, topic coherence and RAGAS faithfulness cannot be combined into one number meaningfully. That's not a convenience argument — it's the reason my evaluation is stage-wise, and it's what let me locate the pipeline's weakness in retrieval rather than generation. A single composite score would have pointed me at the prompt."

**Follow-ups.** Could an LLM do all three now? What would you lose? How do errors propagate between stages?

**Strong addition.** "Errors do propagate, and I documented it: discovery runs on negative reviews only, so a review misclassified as negative enters the topic model as noise. Each stage inherits the one before it."

---

### Q4. Your sentiment stage has five models. Justify five rather than two.

**Model answer.** "Each represents a distinct answer to the same problem rather than a variation. VADER is the untrained floor — it learns nothing, so it measures how far fixed rules go, and the roughly 40-point macro F1 gap to the fine-tuned model quantifies what learning from data buys. Naive Bayes and logistic regression are the classical family, and having both mattered more than I expected: they trail each other by 4.5 points on identical features, which means roughly two thirds of the distance from Naive Bayes to DistilBERT is recovered by swapping one linear model for a better-specified one. The transformer architecture accounts for less of that gap than it appears to.

RoBERTa is pretrained-unmodified, representing what a practitioner gets off the shelf. DistilBERT is fine-tuned. Without both you can't separate 'transformers are better' from 'task-specific training is better'."

**Follow-ups.** Was five worth the compute? Which would you drop? Why not SVM or gradient boosting?

**Good answer to "which would you drop":** "None, and the Naive Bayes finding is why — dropping it would have let me attribute the whole gap to the transformer, which the evidence doesn't support."

---

### Q5. Tell me about a bug that took you a long time to find.

**Why asked.** Standard, and you have an unusually good answer. Do not waste it.

**Model answer.** "One of five evaluation questions scored zero on three RAGAS metrics. I wrote it up as the grounding threshold working exactly as designed — declining to answer rather than inventing something. It fitted my design argument neatly.

It was wrong. The text the pipeline actually returned was the out-of-scope message, not the no-evidence message. That meant the scope classifier had rejected 'What customer service issues appear most often?' before retrieval ever ran — a legitimate question for a customer-feedback system. The metrics were identical either way: zero faithfulness, zero context precision, zero recall. Only the returned string distinguished a safety feature working from a component misfiring.

The correction surfaced because later ablation runs reproduced the same rejection twice more, making the pattern too consistent to ignore. What I took from it is narrower than 'be careful': when a result agrees with what you designed for, that agreement is the moment to check the artefact, not the moment to write the sentence you were hoping to write."

**Follow-ups.** Why didn't your metrics catch it? What would you build so it's caught automatically? How would you test the scope guard properly?

**Have this ready:** log the refusal *reason* as a structured enum, not a string; alert on refusal-rate by reason; build a labelled in-scope/out-of-scope probe set of ~100 questions and track guard precision/recall in CI.

---

### Q6. What would you do differently if you started again?

**Model answer.** "Fix the experimental controls first. I generated data splits separately in different scripts as I needed them, and found out late that this left the fine-tuned model evaluated on data overlapping its training set. Worse, the corpus version it trained on had been overwritten, so I couldn't quantify the overlap after the fact — only disclose it.

One split file, generated once, saved to disk, read by every script. That's a fifteen-minute change at the start of a project and it would have removed the single most consequential caveat in my results.

Second thing: I'd budget for paid API access. Free-tier rate limits capped my generation evaluation at five questions, prevented an independent judge model, and meant three of four ablation conditions were never built. I treated the constraint as fixed for too long instead of costing it."

**Follow-ups.** How would you detect leakage in a production pipeline? What's the CI check?

**Strong close.** "In production I'd hash the training IDs and assert empty intersection with the eval set as a test, so the pipeline fails rather than silently reports an inflated number."

---

### Q7. Who is this system for, and how would you know it worked?

**Model answer.** "Customer experience and product analysts who currently read samples of feedback by hand. The value isn't automation of a metric — it's that every answer arrives with the reviews that produced it, so a claim can be checked in seconds before it informs a decision.

I'm careful about what I claim on usefulness, though. I never ran a user study, so I framed the research question around traceability rather than business value. Traceability I can measure — every answer returns review identifiers, and faithfulness of 0.540 quantifies how far answers follow from the evidence. Whether analysts trust it more, or make better decisions, is unestablished and I'd want a study before claiming it.

If I were productising, the metric I'd want is the rate at which an analyst clicks through to a source review and then *changes* their conclusion. That measures whether the citations do work rather than whether they exist."

**Follow-ups.** What's the business case for the 849ms and per-query API cost? When would you not use RAG here?

**Good answer:** "For exploratory analysis where a wrong answer is noticed immediately, the ungrounded model is faster, cheaper and adequate. The grounding earns its cost where acting on a wrong conclusion is expensive and the analyst is accountable for the recommendation."

---

### Q8. What's the weakest part of your project?

**Why asked.** Testing self-awareness. Candidates who claim nothing is weak fail this.

**Model answer.** "The generation evaluation. Five questions, and the model judging the answers is the model that produced them. Per-question faithfulness ranged from 0.000 to 1.000 and context precision from 0.000 to 1.000, so the specific values shouldn't be quoted as measurements of system quality. What the evidence supports is qualitative — that relevancy and faithfulness diverge, that grounding is measurable for one system and undefined for the other, that retrieval is the weaker component.

The judge's reliability failed visibly too: RAGAS raised an OutputParserException on one question, so answer relevancy is averaged over four questions rather than five. That's not a theoretical concern about self-grading bias, it's the bias's cousin showing up in my own run.

Fixing it needs an independent judge and a question set in the hundreds. Both were blocked by free-tier rate limits, not by anything methodological."

**Follow-ups.** How many questions would you need? How would you pick them? Who annotates ground truth?

---

### Q9. Your Chapter 5 says RAG scored *worse* than the baseline. Why keep the RAG system?

**Model answer.** "On mean answer relevancy, yes — 0.610 against the baseline's 0.753, in under half the time. On that one metric retrieval made the system worse, and I reported it that way.

The per-question distribution is the actual finding. The baseline's scores are 0.988, 0.000, 0.961, 0.964, 0.853 — it wins four of five and fails one completely. Its mean is dragged down by that single zero; excluding it puts the baseline near 0.941. So the baseline has the higher ceiling *and* the lower floor. On the question where it collapsed to zero, the grounded pipeline scored 0.740.

A practitioner choosing between them is choosing between higher average responsiveness with occasional total failure, and lower average responsiveness with a floor. And faithfulness is undefined for the baseline — there's no context for its answers to be faithful to. That absence is the finding, not a gap in my evaluation. A study reporting only answer relevancy would have concluded retrieval augmentation degrades performance, which is precisely why I used four metrics."

**Follow-ups.** Isn't n=5 too small to claim a distribution? (Yes — say so.) Would you ship the baseline for some use cases?

---

### Q10. How is this reproducible?

**Model answer.** "Seeds fixed at 42 throughout, dependency versions pinned in requirements.txt, and every figure and table in the results chapter is written to disk by the script that computed it — so no number was transcribed by hand. The scripts are named alongside each table.

Experiment tracking is MLflow, which preserves parameters, metrics and artefacts per run so an individual result traces back to its configuration.

The gap I'd flag: the FAISS index and the fine-tuned model are ~4GB of artefacts not in version control, so a reader can reproduce the pipeline but must rebuild the index. And the split contamination means one model's shared-test-set number isn't reproducible in the controlled sense — that's disclosed rather than fixed."

**Follow-ups.** Where do the artefacts live? How would you version a 990MB index? DVC? Model registry?
