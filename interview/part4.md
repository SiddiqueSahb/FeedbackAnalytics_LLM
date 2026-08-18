# ROUND 4 — Design Decisions Under Challenge

Short answers here. The interviewer is testing whether you chose or copied.

### Q37. Why FastAPI and not Flask or Django?

"Async support without extra machinery, which matters because my handlers do blocking work — FAISS search and an external LLM call. I wrap those in `asyncio.to_thread` so they don't block the event loop. Pydantic request/response models give validation and typed schemas for free, and automatic OpenAPI docs meant the frontend contract was self-documenting. Flask would need Quart or gunicorn workers for the same concurrency story; Django is a full framework where I needed four endpoints and no ORM."

**Follow-up:** What actually happens if you *don't* use `to_thread` here? — "A single FAISS search blocks the event loop, so concurrent requests serialise and your async framework gives you nothing. It's the classic mistake of writing `async def` around synchronous work."

---

### Q38. Why Streamlit and not React?

"Because the frontend isn't the contribution and I had five months. Streamlit gave me five pages — dashboard, analyse, search, chatbot, upload — in a fraction of the time, and it keeps the frontend a thin HTTP client over the backend.

I'd be clear about what that costs: no component reuse, limited control over rendering, and it re-runs the whole script on interaction, so it doesn't scale to many concurrent users. For a real product it's a prototype layer. But it's deliberately decoupled — the frontend only calls the API, so replacing it with React is a frontend rewrite, not a system rewrite."

---

### Q39. Why two Docker containers and not one?

"Separation of concerns and resource profile. The backend needs torch, faiss and transformers — a heavy image. The frontend needs Streamlit and `requests`, which is light. One container would force the frontend to carry ~2GB of ML dependencies it never imports, and you'd scale them together when their load profiles differ entirely: retrieval is CPU-and-memory bound, the UI is I/O bound.

Data and models are mounted as volumes rather than baked in — they're multi-GB and change independently of code, so baking them in would make every code change a multi-GB image rebuild."

**Follow-up:** How does that work on Cloud Run, which has no volumes? — "It doesn't, and that's a real gap. On Cloud Run I'd need either a persistent-disk-backed service, or fetch the index from GCS on cold start, which adds startup latency. Worth being honest that my Docker setup assumes a host filesystem."

---

### Q40. Why REST and not gRPC?

"Request volume is low and interactive, the client is a browser-based UI, and REST plus JSON needs no client codegen or proxy for browser support. gRPC earns its keep on high-throughput service-to-service calls with strict schemas and streaming — none of which describes an analyst asking a question every few minutes.

If this became an internal service called by other backends at high QPS, gRPC's binary framing and HTTP/2 multiplexing would matter. Streaming would also let me stream generation tokens rather than waiting 1.5s for a complete answer, which is a genuine UX argument for it."

---

### Q41. Why DistilBERT and not DeBERTa-v3 or RoBERTa for fine-tuning?

"Hardware. Distillation retains ~97% of BERT-base performance at ~40% fewer parameters, and fine-tuning took 2h56m on freely available hardware. DeBERTa-v3 would likely score higher — its disentangled attention is stronger on classification benchmarks — but roughly doubles training time and served model size.

Worth noting I *did* use DeBERTa-v3 elsewhere: it's the zero-shot classifier for category assignment, chosen over `bart-large-mnli` for stronger reported zero-shot performance with larger label sets. So the choice was task-conditional rather than a blanket preference."

---

### Q42. Why the two-stage shortlist-then-entailment design for category assignment?

"Cost and evidence. Cost: a flat zero-shot pass scores every one of 24 categories per review, and that scales badly as the taxonomy grows. The embedding shortlist cuts it to six.

Evidence: Schopf et al. benchmarked similarity-based against zero-shot approaches across four datasets and found similarity-based significantly better in most cases. So shortlisting by similarity before applying entailment isn't just cheaper, it's using each method where the literature says it's stronger.

One more detail — the hypothesis passed to the entailment model is the category's full LLM-generated *description*, not a bare category name, because label verbalisation quality matters as much as model choice. Which creates a dependency I'd flag: the naming step directly determines assignment quality, and it failed silently on one cluster."

---

### Q43. Why per-platform BERTopic rather than pooling the corpus?

"Yelp is 70% of the corpus. Pooling lets it dominate the topic space, and you get a taxonomy describing restaurant service while claiming to describe customer feedback generally.

The cost is a harder consolidation problem — 121 topics needing reconciliation across platforms. I took that trade deliberately.

It also produced the most informative structural result. Amazon gave 69 topics from 57,931 documents; Yelp gave 18 from 149,838. That's not a parameter artefact — minimum topic size was scaled to each platform's document count specifically to prevent that. It's semantic breadth: Amazon spans electronics, groceries, clothing and cosmetics, where a failing battery and a stale flavour share almost no vocabulary, so HDBSCAN finds many small dense regions. Yelp is narrow by comparison. Which means the merge to 24 does genuine consolidation for Amazon and near-relabelling for Yelp — compression ratios differing by almost 4× — so the taxonomy fits Yelp's structure better than Amazon's."

---

### Q44. Why HDBSCAN rather than k-means for clustering?

"Two reasons. It doesn't need k specified in advance, which matters when you don't know how many complaint types exist. And it leaves points unassigned as noise rather than forcing every document into a cluster — which is the right behaviour here, because not every negative review expresses a categorisable complaint.

That noise-handling turned out to matter a lot in my results. It's the same philosophy as my 0.35 confidence threshold: recording a weak match as unclassified is more informative than forcing it into the nearest bucket. The 27.6% unclassified rate is an unflattering number that's trustworthy precisely because the design allows it to appear."

---

# ROUND 5 — Where I'd Attack Your Design

Read this as the interviewer trying to reject you. Each has the attack and your best defence.

### Attack 1: "Your test set is contaminated. Why should I trust any of your numbers?"

**Defence.** "One model's number on one test set, and I'd separate that from the rest. The fine-tuned model was developed against a separately generated corpus version, so a portion of the shared test set overlaps data it saw. The extent can't be quantified because the earlier corpus was overwritten.

What that affects: DistilBERT's 0.8244 on the shared set. What it doesn't affect: the other four models, the per-platform findings, the diagnostic experiments, or the taxonomy and RAG stages.

And I report 0.8375 as definitive throughout — that's from the model's *own* held-out partition, measured at fine-tuning time on data it provably never saw. So the caveated number and the clean number are both stated, with the difference explained.

In production this is a leaked-feature incident. The control is a single canonical split generated once, saved to disk, read by every script, with a CI assertion that training and eval ID sets don't intersect."

---

### Attack 2: "You shipped a magic number. 0.35 isn't engineering, it's guessing."

**Defence.** "Correct, and my own code comment says so — 'a starting point, not a validated constant.' I won't defend the number. I'll defend two things.

First, that having an architectural gate at all is better than not having one, and I have evidence: pre-gate, a question with 0.000 context precision produced an answer scoring 0.917 faithfulness. The gate changed that behaviour measurably.

Second, that I know exactly how to calibrate it and why I didn't — a ~100-question probe set stratified into in-scope, out-of-scope and deliberately borderline, sweeping the threshold against false-refusal and hallucination rates with an explicit cost ratio. It's the cheapest experiment on my future-work list and it addresses a defect my evaluation found by accident."

---

### Attack 3: "Five evaluation questions is not an evaluation."

**Defence.** "Agreed, and I say that in the dissertation rather than defending it. Per-question faithfulness spans 0.000 to 1.000. No confidence interval is reported because none would be meaningful.

What I'd push back on is the implication that it's therefore worthless. It demonstrated the measurement approach, and it surfaced two things a larger but sloppier evaluation might have missed: the divergence between relevancy and faithfulness, and a scope-classifier bug I found by reading individual outputs. With five questions I could inspect every one. At n=500 I'd have seen a mean and missed it.

The right framing is that n=5 supports qualitative findings and cannot support precise values, which is exactly how I framed the RQ2 conclusion."

---

### Attack 4: "Your system takes 1.5 seconds and costs an API call per query. Nobody will use it."

**Defence.** "For some use cases, correct. The ungrounded baseline is faster, cheaper and adequate where an analyst will notice a wrong answer immediately.

The grounding earns its cost where acting on a wrong conclusion is expensive and someone is accountable for the recommendation. And the cost profile isn't uniform — a rejected question completes in 112ms because both guards return before generation.

If latency were the binding constraint I'd stream tokens rather than waiting for a complete answer, which changes perceived latency substantially without changing total time. And I'd cache: identical or near-identical questions are common in analyst workflows, and a semantic cache on the embedded query would eliminate the generation call entirely for repeats."

---

### Attack 5: "Flat FAISS with a single-process load and a threading lock. This doesn't scale past one instance."

**Defence.** "It doesn't, and it wasn't built to. `lru_cache(maxsize=1)` plus a lock loads the index once per process and makes that load thread-safe — appropriate for a single-instance research deployment, wrong for production.

The problems, named precisely: 990MB loaded per process means memory scales with worker count, not with load. Cold start pays the full index load. No sharding, no replication, and no way to add a review without touching the in-process index.

Production shape: index build as an offline job producing a versioned artefact in object storage; a retrieval service loading it read-only with N replicas behind a load balancer; IVF or HNSW instead of Flat once corpus growth makes exhaustive scan too slow; Redis for a semantic query cache; and index updates as a blue-green swap rather than in-place mutation."

---

### Attack 6: "Your scope guard rejected a valid question and you only noticed by accident. What else is broken that you haven't noticed?"

**Defence.** "That's the right question and I don't have a complete answer, which is itself informative.

What I can say: the reason it was invisible is that both failure modes — correct refusal and false rejection — produced identical metrics. Zero faithfulness, zero precision, zero recall. Only the returned string differed.

So the class of bug I'm exposed to is any failure that produces metrically-indistinguishable-but-semantically-different outcomes. The fix is structural rather than a patch: log the refusal *reason* as an enum rather than a message string, monitor refusal rate segmented by reason, and put a labelled probe set in CI so guard precision and recall are tracked rather than assumed.

The honest general answer is that my evaluation is too small to have found everything, and the scope-guard case is evidence for that rather than an isolated incident."
