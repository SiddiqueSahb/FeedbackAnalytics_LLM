# ROUND 2 — RAG, Retrieval and Vector Search

This is where an MLE interview will spend the most time on your project, and where shallow answers are most obvious.

### Q11. Why FAISS? Why not Pinecone, Weaviate, Chroma or pgvector?

**Model answer.** "Three reasons, and one of them is about reproducibility rather than performance. FAISS runs locally with no external service, so the index can be rebuilt by anyone reading the dissertation — an index depending on a hosted service can't be reconstructed once that service changes its API or pricing. Second, cost: this was an MSc project with no budget, and managed vector databases charge per index. Third, at 642,692 vectors I don't need distributed serving; that's a single-node problem.

What I'd lose at production scale is real: no managed replication, no filtered-search optimisation, no zero-downtime reindexing. If this served 500 QPS multi-tenant I'd move to a managed store or run FAISS behind a sharded service with a read-replica pattern, because rebuilding a 990MB index in-process means a cold start on every deploy."

**Follow-ups.** How large is your index? What happens on deploy? How do you add a review without a full rebuild?

**Have ready.** 642,692 × 384 × 4 bytes ≈ 990MB raw. `IndexFlatL2` supports `add()` for incremental additions, but there's no delete without rebuild — you'd tombstone in metadata and filter post-retrieval, then rebuild on a schedule.

---

### Q12. You use IndexFlatL2 with normalised vectors. Why L2 rather than inner product?

**Why asked.** This separates people who chose an index from people who copied one.

**Model answer.** "For normalised vectors they're rank-equivalent — ‖a−b‖² = 2 − 2·a·b, so ordering by ascending L2 gives the same ranking as descending cosine. I used L2 because LangChain's FAISS wrapper defaults to it and its relevance-score conversion assumes Euclidean: it maps distance to a [0,1] score via 1 − distance/√2, which is what my 0.35 threshold is expressed in.

If I were writing the retriever directly I'd use `IndexFlatIP` with normalised vectors, because the score is then cosine similarity directly and the threshold is interpretable without a conversion step. The conversion is a place where a subtle bug could hide."

**Follow-ups.** Derive that identity. What's √2 doing there? What if vectors weren't normalised?

**Answer to the last one.** "L2 and cosine diverge — L2 becomes sensitive to magnitude, so a long review with a larger norm could rank differently than its semantic similarity warrants. Normalisation is what makes them equivalent, and I set `normalize_embeddings=True` in the encoder for exactly that reason."

---

### Q13. Flat index means exhaustive search. What's your complexity, and when does it break?

**Model answer.** "O(N·d) per query — 642,692 × 384 multiply-accumulates, roughly 247 million FLOPs, which is why retrieval measures 21.71ms on CPU. Memory is O(N·d) at ~990MB for the vectors, plus the docstore for text and metadata.

It breaks on two axes. Latency: exhaustive scan is linear, so 10× the corpus is roughly 10× the retrieval time — about 220ms, at which point retrieval stops being one per cent of the request and starts mattering. Memory: 10× is ~10GB, past comfortable single-instance RAM.

First move is IVF-Flat — cluster into √N ≈ 800 centroids, probe nprobe of them, so you scan roughly nprobe/nlist of the corpus. With nprobe=16 that's 2% of vectors for maybe 95–98% recall@10. If memory is the binding constraint rather than latency, IVF-PQ compresses vectors via product quantisation at a further recall cost. HNSW gives better recall-latency trade-off than IVF but its graph structure roughly doubles memory."

**Follow-ups.** How would you measure the recall you lost? What's your recall target and why?

**Strong answer.** "Measure exhaustive Flat as ground truth on a held-out query set, then recall@k of the approximate index against it. I'd target 95%+ recall@5 — but I'd want to know how much answer quality actually degrades, because recall@5 loss may not translate proportionally into faithfulness loss. That's an experiment, not an assumption."

---

### Q14. Why MiniLM-L6-v2? Why not mpnet, BGE, E5 or an OpenAI embedding?

**Model answer.** "384 dimensions and it's fast on CPU, which matters when the index is ~990MB rather than ~2GB. Moving to `all-mpnet-base-v2` at 768 dims doubles the index footprint and roughly doubles query time.

The important part is that I tested whether that trade-off cost me anything rather than assuming. I compared dense retrieval against a TF-IDF baseline on 30 queries: Precision@3 was **identical at 0.90**, while dense was 5.9× faster — 21.71ms against 127.61ms. So on my data, dense retrieval's case rests on latency, not retrieval quality.

That result also bounds the upside of a bigger embedding model. If MiniLM already matches sparse retrieval on precision, a larger encoder has limited headroom to demonstrate, and I'd want that measured before paying 2× memory and latency for it."

**Follow-ups.** Thirty queries is small — could a real difference hide? (Yes. Say so.) How did you judge relevance for Precision@3?

**Answer.** "Human judgement — the evaluation script prints each retrieved review and prompts for a y/n relevance call. It's genuine human evaluation, but single-annotator with no inter-rater agreement, so I'd treat 0.90 as indicative."

---

### Q15. Walk me through your chunking strategy.

**Model answer.** "There isn't one, deliberately. Reviews are indexed as whole documents because they're already short — median 19 words on Twitter, 29 on Amazon, 85 on Yelp. Chunking exists to fit long documents into a context window and to keep retrieved passages topically tight; neither problem applies to a 29-word product review. Splitting one would destroy the unit of meaning.

I do cap text entering the prompt at 1,000 characters per review, but that's a context-overflow guard rather than chunking — it doesn't bind on current data and exists in case the index is extended with longer-form content.

If I indexed support tickets or call transcripts I'd need real chunking, and I'd start with semantic or sentence-window splitting rather than fixed tokens, because fixed-size splits cut mid-argument and a complaint's resolution often sits in the last sentence."

**Follow-ups.** What overlap would you use and why? How does chunk size interact with your threshold?

**Sharp answer.** "It interacts directly — smaller chunks concentrate similarity, so the same 0.35 threshold would admit more chunks. Any chunking change invalidates the threshold and it would need recalibrating."

---

### Q16. Why MMR rather than plain similarity search?

**Model answer.** "Domain-specific reason. Complaint questions retrieve near-duplicate reviews expressing the same grievance — ask about battery life and you get five variants of 'battery drains fast.' Five restatements of one complaint provide weaker grounding than five distinct ones, while scoring identically on similarity. MMR with lambda 0.5 balances query relevance against inter-document diversity, so the context covers more of the complaint space.

Implementation detail worth flagging: MMR only controls diversity among whatever top-k FAISS returns. It has no notion of whether any of it is relevant. So I run two passes — score candidates and keep those above threshold, run MMR over the same query, then intersect. Without the intersection, MMR's diversity re-ranking could smuggle in a sub-threshold review purely because it was topically distinct from the others."

**Follow-ups.** What happens when MMR returns 5 and only 2 pass the threshold?

**Honest answer, and this is a real gap.** "You get 2 documents where the prompt expects 5. There's no top-up logic — the pipeline proceeds with whatever survives. The airline question in my evaluation hit exactly this: only two reviews cleared, both thin, and faithfulness fell to 0.400. In production I'd either widen fetch_k and retry, or surface low-evidence-count as a signal to the user rather than silently answering from two documents."

---

### Q17. Explain your two guards and why the order matters.

**Model answer.** "Two gates before generation, addressing different failures. The scope guard rejects questions that aren't about customer feedback — a keyword filter naming nine off-topic subjects, then a small LLM classifier for cases keywords miss, like 'Who is the CEO of Apple?' It returns a fixed out-of-scope message without touching the retriever. The similarity threshold then gates retrieval output: candidates below 0.35 are discarded, and if nothing survives, a fixed no-evidence message replaces generation. A post-generation check repeats the test for multi-turn queries, where the retrieval query is an LLM-reformulated question the pre-check never saw.

The order matters for cost and for diagnosis. Cost: the scope guard runs first, so an off-topic question never pays for embedding or retrieval — 112ms versus 1,583ms. Diagnosis: because they return *different* messages, I could tell which fired. That's how I found the scope guard was falsely rejecting a valid question — the metrics were identical for both failure modes, and only the returned string distinguished them."

**Follow-ups.** Why is the keyword filter first? What's the LLM classifier's precision? What if the scope guard's LLM call fails?

**Answers.** Keyword filter is free and instant, so it short-circuits the obvious cases without an API call. The classifier's precision is unmeasured — that's a real gap, and I know of at least one false positive. On failure it falls back to the keyword-only check, so it can't crash the pipeline or leave a question unclassified.

---

### Q18. How would you calibrate the 0.35 threshold properly?

**Why asked.** You shipped an unvalidated constant. Own it and show the method.

**Model answer.** "I'd build a probe set of about 100 questions in three strata: clearly in-scope with good evidence in the corpus, clearly out-of-scope, and a deliberate band of borderline phrasings — abstract or broad in-scope questions like the one my scope guard wrongly rejected. Each labelled for whether the system *should* answer.

Then sweep the threshold and plot two curves: false-refusal rate on answerable questions, and hallucination rate on unanswerable ones, where hallucination is judged by whether the answer contains claims absent from retrieved context.

On the cost ratio — it depends on deployment, and I'd want the product owner's input rather than assuming. For an analyst-facing tool informing decisions, a hallucinated answer is worse than a refusal, because a refusal is visible and self-correcting while a plausible wrong answer propagates. I'd weight hallucination maybe 5:1 over false refusal and pick the threshold minimising that weighted cost. For a low-stakes exploratory tool the ratio inverts, because a system that refuses constantly gets abandoned.

The reason I state a method rather than a number is that 0.35 came from inspecting my index's score distribution, not from optimising anything."

**Follow-ups.** Is one global threshold right? Should it vary by query type?

**Strong answer.** "Probably not global. Open-ended questions like 'summarise the major complaints' have no specific retrieval target, and my context recall was 0.000 on exactly that question. A single threshold treats it like a specific question when it structurally isn't. Query-type-conditional thresholds, or routing open-ended questions to a different path entirely, would be my next experiment."

---

### Q19. Your context precision is 0.416 and faithfulness 0.540. What does that tell you?

**Model answer.** "That retrieval is the weaker component, not generation — which is not where prompt-engineering attention usually goes. Precision fell to zero on two of five questions, and on the airline one faithfulness fell with it, to 0.400.

That co-movement is the important part. In the earlier un-gated pipeline, a comparable zero-precision question still scored 0.917 on faithfulness — the system was being faithful to evidence its own retrieval had judged irrelevant. After the gate, faithfulness tracks evidence quality instead of staying artificially high. So the gate is doing what it was designed to do.

This is exactly the diagnostic Yu et al. argue aggregate RAG scores can't provide. A single composite score would have sent me to revise the prompt. The component metrics say the prompt is roughly fine and the retrieval is the constraint — so effort belongs in retrieval depth, reranking, or query reformulation for open-ended questions."

**Follow-ups.** So what would you fix first? Why didn't you?

**Answer.** "Retrieval depth — a k-sweep over {3, 5, 10, 20} would say directly whether five reviews is the right amount of evidence or just the number I chose. It was one of four ablation conditions I designed; three were never built because free-tier rate limits made the call volume unreachable. Retrieval depth is the one I regret most, given that the component metrics point straight at it."

---

### Q20. Add a reranker to your pipeline. Where does it go and what does it cost?

**Model answer.** "Between retrieval and the threshold gate. Retrieve a wider candidate set — fetch_k of 50 rather than 20 — then rerank with a cross-encoder and apply the threshold to reranked scores.

The reason it should help here specifically is the failure mode I actually observed. My retriever is a bi-encoder: query and document are embedded independently, so it can't model term interaction. That's how a Yelp review about waiting to get a *battery replaced in-store* ranked highly for a question about battery *life* — the embedding matched on 'battery' without modelling what was being asked about it. A cross-encoder scores the pair jointly and would likely have demoted it.

Cost is the problem. A cross-encoder is a forward pass per candidate, so 50 candidates is 50 inferences, not one vector lookup. On CPU with a MiniLM cross-encoder that's realistically 200–500ms — an order of magnitude above my 21.71ms retrieval. Given generation already dominates at ~1.5s it might be acceptable, but I'd measure rather than assume, and I'd want the recall/precision gain quantified against that latency before shipping it."

**Follow-ups.** Bi-encoder vs cross-encoder — explain the architectural difference. Why can't you use a cross-encoder for first-stage retrieval?

**Answer.** "Because it needs the query paired with each document, so there's nothing to precompute — you'd score all 642,692 pairs per query. Bi-encoders let you embed documents once, offline, and reduce query time to a vector search. That's the whole reason two-stage retrieval exists."

---

### Q21. Why LangChain rather than writing the chain yourself?

**Model answer.** "For the history-aware retriever mainly — `create_history_aware_retriever` handles reformulating a follow-up into a standalone query before retrieval, and that's fiddly logic I'd otherwise write and test myself.

Honestly though, my usage is thin enough that the dependency is arguable. I use the FAISS wrapper, the retrieval chain, and the condense-question prompt. I also had to fight it: I ended up writing a custom `RunnableLambda` retriever because `as_retriever(search_type='similarity_score_threshold')` didn't give me the two-pass threshold-then-MMR-intersect behaviour I wanted. So the abstraction stopped fitting at the point my requirements got specific.

If I rebuilt it, I'd probably drop LangChain for direct FAISS plus explicit prompt templates — fewer layers between me and the behaviour, and easier to debug. The scope-guard bug took longer to find partly because the returned message was several layers deep."

**Follow-ups.** What does LangChain's abstraction cost you in debugging? Version churn?

---

### Q22. Why Groq and Llama 3.1 8B? Why not GPT-4 or Claude?

**Model answer.** "Cost — the free tier permitted the evaluation call volume I needed, and I had no budget. Groq's inference is also genuinely fast, which mattered for an interactive chatbot.

But I'd want to be straight that this was a resource-driven decision whose consequences propagated further than I anticipated. Rate limits capped the generation evaluation at five questions, prevented using an independent judge model, and meant three of four ablation conditions were never built. In hindsight, a small paid budget would have strengthened the evaluation more than any modelling change available to me.

Temperature 0.2 was deliberate rather than default: the task is evidence-grounded answering where output variability is a defect, and a reproducible evaluation needs responses that don't swing between runs."

**Follow-ups.** Why not temperature 0? What breaks if you swap model providers? Is your prompt portable?

**Good answer on 0.** "Arguably it should be 0 for this task. 0.2 leaves a little variation with no benefit I can point to for grounded answering. I'd change it."
