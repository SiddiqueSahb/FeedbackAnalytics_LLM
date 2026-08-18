# Escalation Chains

Thirty questions you are most likely to be asked, each with the follow-up ladder a strong interviewer will actually walk you down.

**How to use this.** Read level 1, answer out loud, then read level 2 and answer that, and so on. Keep going until you can't. **That point is the thing to find** — better here than in the room. Note it, go and learn it, come back.

Still questions only. The *"bottom of the ladder"* line tells you what depth is expected and where most candidates stop, without giving you the answer.

---

## SECTION 1 — YOUR PROJECT

### C1. Walk me through your project.
1. Walk me through your project in two minutes.
2. Why three separate stages rather than one end-to-end model?
3. What would break first if I doubled your traffic tomorrow?
4. Which of the three stages would you delete if you had to ship next week, and what do you lose?
5. If the fine-tuned model and the RAG pipeline disagree about a review, which do you trust and why?

**Bottom of the ladder:** whether you understand the system as a set of coupled decisions or as a list of components you assembled. Most candidates can describe; few can defend deletions.

---

### C2. Your fine-tuned DistilBERT beats logistic regression by two points. Justify the transformer.
1. Two points. Why not just use logistic regression?
2. Show me where the two-point gap comes from — which class, which platform?
3. What's the inference cost difference, concretely — latency, memory, hardware?
4. At what traffic volume does the transformer's cost outweigh the accuracy gain?
5. If I told you the production traffic was 95% Amazon, what would you deploy?

**Bottom of the ladder:** whether you can turn a model comparison into a deployment decision with numbers attached. The right answer changes with the traffic mix.

---

### C3. Your test set is contaminated.
1. Explain the contamination.
2. Why can't you quantify how much overlap there was?
3. Which of your conclusions survive it and which don't?
4. Walk me through the control that would have caught it. Where exactly does it run?
5. You have a script that claims the overlap *is* measurable. Which is true?

**Bottom of the ladder:** whether you can bound the damage of your own mistake precisely instead of either dismissing it or over-apologising. Level 5 is real — resolve it before you interview.

---

### C4. Tell me about a bug that took a long time to find.
1. Tell me about a bug that took you a long time to find.
2. How did you eventually notice it?
3. Why didn't your metrics catch it?
4. What class of bug are you still exposed to, given that mechanism?
5. What would you change structurally so this category becomes visible?

**Bottom of the ladder:** whether you generalise from an incident to a class of failure. "I fixed it" is a level-2 answer.

---

## SECTION 2 — TRANSFORMERS AND FINE-TUNING

### C5. Explain self-attention.
1. Explain self-attention.
2. Write the scaled dot-product attention formula. Why divide by √d_k?
3. What's the time and memory complexity in sequence length? Why does that matter for your 512-token limit?
4. What do multiple heads buy you that one wide head doesn't?
5. How would you reduce that complexity — what do flash attention or sparse attention actually change?

**Bottom of the ladder:** most candidates stop at level 2. Level 3 onwards separates people who've read the paper from people who've read a blog post.

---

### C6. Your validation loss rose after epoch 2 while training loss kept falling.
1. What's happening there?
2. Why does validation loss rise while accuracy stays roughly flat?
3. So which do you select the checkpoint on, and why?
4. Would early stopping have changed your final model? Why did you train four epochs anyway?
5. What would you change — learning rate schedule, weight decay, dropout — and which would you try first?

**Bottom of the ladder:** the loss/accuracy divergence at level 2 is the discriminating question. It's about confidence calibration, not correctness.

---

### C7. Your pretrained RoBERTa scored below logistic regression.
1. A 125M-parameter transformer lost to logistic regression. Explain.
2. How did you establish that it was the neutral class specifically?
3. Why does the pretraining domain cause that, mechanically?
4. Your binary re-evaluation moved it from 0.667 to 0.934. What does that prove and what does it not?
5. If you had to deploy that pretrained model unchanged, how would you get usable performance out of it?

**Bottom of the ladder:** level 4 is the trap — the re-evaluation shows the *capability* is there, not that the model is fit for a three-class task. Say so.

---

### C8. Why DistilBERT rather than a larger model?
1. Why DistilBERT rather than RoBERTa or DeBERTa-v3?
2. What does distillation actually do during training?
3. You used DeBERTa-v3 elsewhere in the project. Justify the inconsistency.
4. What would you expect DeBERTa-v3 to buy you here, in points, and at what training cost?
5. How would you serve a model twice the size under the same latency budget?

**Bottom of the ladder:** level 5 pulls in quantisation, batching and hardware. Most candidates have never thought about serving cost at all.

---

## SECTION 3 — EVALUATION AND STATISTICS

### C9. Why macro F1 rather than accuracy?
1. Why is macro F1 your headline metric?
2. Construct a case where accuracy is 90% and macro F1 is 40%.
3. When would weighted F1 be the *correct* choice instead?
4. Your classes aren't equally important to the business. How does that change the metric?
5. Design a single metric for this system that a product manager would accept.

**Bottom of the ladder:** level 5 is where it stops being a textbook question. There's no clean answer — they want to see you reason about cost asymmetry.

---

### C10. Explain McNemar's test.
1. What is McNemar's test and why did you use it?
2. Why a paired test rather than comparing F1 scores directly?
3. Write the contingency table. Which cells does the statistic actually use, and why are the others ignored?
4. Two of your models gave p = 0.82. State precisely what that does and does not license you to claim.
5. You ran ten pairwise comparisons. What's your family-wise error rate, and did you correct for it?

**Bottom of the ladder:** level 5 is the one almost nobody volunteers. Ten comparisons at α = 0.05 is roughly a 40% chance of at least one false positive.

---

### C11. Five evaluation questions is not an evaluation.
1. You evaluated generation on five questions. Defend that.
2. What's the confidence interval on a mean of five bimodal scores?
3. How many questions would you need to detect a 0.05 difference in faithfulness?
4. What did the small n let you do that a larger, sloppier evaluation wouldn't have?
5. Design the evaluation you'd run with a £200 budget. Be specific about the split.

**Bottom of the ladder:** level 3 requires an actual power calculation, not a vibe. Level 4 is your genuine counter — you inspected every output and found a bug.

---

### C12. Your ablation improved one metric and hurt another.
1. Faithfulness up 0.118, relevancy down 0.202. Do you ship it?
2. How do you know the difference came from the prompt and not from retrieval noise?
3. Both metrics come from an LLM judge. How much of that delta is real?
4. What experiment would tell you whether users prefer the tagged or untagged version?
5. Design the online test — metric, unit of randomisation, sample size, guardrails.

**Bottom of the ladder:** level 2 is where your identical context precision on both arms is the evidence. Use it.

---

## SECTION 4 — RETRIEVAL AND VECTOR SEARCH

### C13. Why L2 rather than inner product?
1. You use IndexFlatL2 with normalised vectors. Why L2 rather than inner product?
2. Derive the relationship between squared L2 distance and cosine similarity for unit vectors.
3. What breaks if the vectors aren't normalised?
4. At 384 dimensions in float32, where does numerical precision start to matter?
5. How does FAISS actually compute this — and what's the bound, compute or memory bandwidth?

**Bottom of the ladder:** level 2 is expected. Level 5 is where hardware-aware candidates separate themselves.

---

### C14. Flat index means exhaustive search.
1. What's the complexity of your flat index search?
2. At 642,692 vectors and 384 dimensions, how much memory and how long per query?
3. At what corpus size does this stop working for a 100ms budget?
4. Compare IVF-Flat, IVF-PQ and HNSW for that point. Which do you pick?
5. HNSW's M and efSearch — what does each control, and how do you tune them against a recall target?

**Bottom of the ladder:** level 2 wants arithmetic out loud. Level 5 wants a tuning methodology, not a definition.

---

### C15. Explain your grounded retriever.
1. Walk me through your two-pass grounded retriever.
2. Why wasn't MMR alone sufficient?
3. What does lambda_mult control and what happens at 0 and 1?
4. You fetch 20 and keep 5. Why those numbers?
5. Your context precision is 0.416. Is that a retrieval problem, a chunking problem or an embedding problem — and what experiment distinguishes them?

**Bottom of the ladder:** level 5 is the real one. Naming three hypotheses and an experiment to separate them is a senior answer.

---

### C16. Your threshold is a magic number.
1. Where did 0.35 come from?
2. What's the cost of setting it too high? Too low?
3. Design the calibration experiment — dataset, labels, sweep, decision rule.
4. What's the cost ratio between a false refusal and a hallucination for this product?
5. Should it be a single global threshold at all?

**Bottom of the ladder:** level 4 forces you to state a business judgement explicitly. Level 5 opens per-query-type or learned thresholds — most candidates never question the single-constant framing.

---

## SECTION 5 — RAG AND LLMs

### C17. Your RAG scored worse than the baseline.
1. RAG lost on relevancy, 0.610 against 0.753. Why keep it?
2. Was the baseline actually better, or just more confident-sounding?
3. Its advantage came from four strong answers and one total failure. What does that tell you about the variance?
4. What use case makes the grounded version obviously correct despite the lower score?
5. How would you measure "verifiability" as a metric a business would accept?

**Bottom of the ladder:** level 5 has no standard answer. They want to see you invent a measurement, not recite one.

---

### C18. Explain your two scope guards.
1. Explain the two guards and why the order matters.
2. Your keyword guard misses most out-of-scope questions. Why keep it?
3. What protection remains if the LLM API is down?
4. How would you measure each guard's precision and recall separately?
5. Design the guard you'd actually ship — what does it cost per query and what's its failure mode?

**Bottom of the ladder:** level 3 is the systems question hiding inside an ML question. Degraded-mode behaviour is what separates engineers from modellers.

---

### C19. Explain RAGAS.
1. Explain the four RAGAS metrics.
2. Which two grade retrieval and which two grade generation? Why does that split matter?
3. Why does context recall need a reference answer, and what does that make it vulnerable to?
4. Your context precision was 0.9999 and recall 0.0 on the same question. Explain.
5. Your judge is the same model family that wrote the answers. Quantify that bias — how?

**Bottom of the ladder:** level 4 is specific to your data and it's a genuinely good story. Level 5 wants an agreement study, not just an acknowledgement.

---

### C20. What is prompt injection?
1. What is prompt injection?
2. Your retrieved reviews are user-generated text going into a prompt. What's the attack?
3. Your prompt tells the model to treat reviews as data, not instructions. Is that sufficient?
4. What's the structural defence rather than the prompt-level one?
5. How would you test for it — build me the test suite.

**Bottom of the ladder:** level 3 is a trap. If you defend the prompt instruction as sufficient, you've failed the question.

---

### C21. When should a RAG system refuse?
1. When should your system refuse to answer?
2. How do you distinguish "correctly refused" from "wrongly refused" in your metrics?
3. Right now both produce identical zeros. What's the fix?
4. What would you monitor in production to catch a guard regression?
5. What's your alert threshold on refusal rate, and how did you choose it?

**Bottom of the ladder:** this is your scope-guard bug generalised. You should own this ladder completely.

---

## SECTION 6 — SYSTEM DESIGN

### C22. Scale this to 100 million reviews.
1. Scale this to 100M reviews, 500 QPS, p99 under 300ms.
2. Do the memory arithmetic out loud. How many shards?
3. Split your 300ms budget across the components. What doesn't fit?
4. Generation alone takes 700ms today. How do you meet the budget at all?
5. Now make it multi-tenant with per-tenant isolation and cost attribution.

**Bottom of the ladder:** if you don't produce numbers at level 2 unprompted, the round is already lost. Level 4 forces you to admit the budget requires either caching or dropping generation.

---

### C23. Your index is 990MB and loaded per process.
1. What's wrong with `lru_cache(maxsize=1)` on the vectorstore?
2. How much memory at four workers? Eight?
3. Give me three fixes and their trade-offs.
4. How do you deploy a new index version with zero downtime?
5. How do you add a single review without rebuilding 990MB?

**Bottom of the ladder:** level 5 opens incremental indexing, delta indexes and periodic compaction — the pattern behind most production vector stores.

---

### C24. Design monitoring for this system.
1. What would you monitor?
2. What pages someone at 3am, and what waits until morning?
3. You have no ground-truth labels in production. How do you detect quality degradation?
4. Your answers suddenly become nonsense. Walk me through triage in order.
5. What's the silent failure mode that no metric would catch?

**Bottom of the ladder:** level 5 — embedding-model mismatch between index build and query time. Every similarity becomes meaningless and nothing errors.

---

### C25. How would you A/B test a model change?
1. How would you A/B test the metadata-tagging change?
2. What's your unit of randomisation, and why not the request?
3. What's your primary metric when quality has no ground truth?
4. How do you size the experiment?
5. Treatment wins on your primary metric and loses on latency. Ship or not?

**Bottom of the ladder:** level 3 is the hard one for LLM products. Proxy metrics, thumbs, session-level behaviour — and their biases.

---

## SECTION 7 — ENGINEERING

### C26. Why two Docker containers?
1. Why two containers rather than one?
2. Why mount data and models as volumes rather than baking them in?
3. How does that design fail on Cloud Run?
4. What's your cold-start time and how do you reduce it?
5. Design the deployment so a bad release rolls back in under a minute.

**Bottom of the ladder:** level 3 is a real gap in your setup — own it rather than discovering it live.

---

### C27. Security-review your API.
1. What's wrong with your API from a security standpoint?
2. Order those by severity and justify the ordering.
3. Why is missing rate limiting an availability problem and not just a cost problem?
4. Your API key is a single shared secret. What's the migration path to per-tenant auth?
5. Review text goes to a third-party API. What are the data-protection implications?

**Bottom of the ladder:** level 2 — ordering by severity — is where judgement shows. Listing everything flat is a junior answer.

---

### C28. Your tests directory has no assertions.
1. Walk me through your test suite.
2. Those are scripts, not tests. Concede or defend.
3. What would you test first, and why that?
4. How do you test a component whose output is non-deterministic?
5. What test would have caught your split contamination? Write it.

**Bottom of the ladder:** level 2 is a character question as much as a technical one. Defending it is worse than conceding.

---

### C29. Why async, and what's the trap?
1. Why did you use FastAPI's async support?
2. What happens if you write `async def` around synchronous FAISS search?
3. How does `asyncio.to_thread` help, and what's its limit?
4. You have one worker and a threadpool. What's your actual concurrency ceiling?
5. Where does the GIL bite here, and where doesn't it?

**Bottom of the ladder:** level 5 — FAISS releases the GIL in C++; pure-Python work doesn't. Most candidates don't know which of their code is which.

---

### C30. How do you version data and models?
1. How do you version your data and models today?
2. MLflow tracks your classical models. What doesn't it track?
3. Your 3GB of artefacts aren't in version control. What broke because of that?
4. Design the versioning scheme — what's the identity of a "dataset version"?
5. How do you make a result from six months ago reproducible on a fresh machine?

**Bottom of the ladder:** level 3 connects directly to your contamination. Level 4 wants content hashing, not "we'd use DVC."

---

# PRACTICE PROTOCOL

**Round one — find your floor.** Walk each chain, out loud, timed. Stop at the first level you can't answer. Write down the chain number and level. Do not look anything up yet.

**Round two — close the gaps.** You'll likely have 20–30 stopping points. Most will cluster in two or three areas — that's your actual revision list, and it's far shorter than "revise everything."

**Round three — go again cold, a week later.** Target: reaching level 4 on every chain and level 5 on the ten closest to your project.

**The chains to own completely,** because they're about your own work and there's no excuse for stopping early: C1, C3, C4, C7, C12, C15, C16, C19, C21, C28.

**The chains where stopping at level 3 is acceptable** for a first MLE role: C5, C13, C14, C22, C29.

One last thing. When you hit the end of a ladder, "I don't know, but here's how I'd find out" is a real answer and interviewers respect it. Bluffing past your edge is the only response that actively costs you.
